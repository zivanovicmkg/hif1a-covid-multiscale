"""
===============================================================================
Scale A — Population-level baseline model on Mexico CDC data
===============================================================================
Purpose: establish a pure demographic/comorbidity baseline AUC. This quantifies
how much predictive power is carried by demographics alone, vs. what Scale C
and D add through laboratory biomarkers.

Design:
    - N = 62,169 hospitalized patients (TIPO_PACIENTE=2)
    - 11 features: demographics + 8 binary comorbidities + pneumonia
    - 2 outcomes: ICU admission (UCI=1), in-hospital mortality (FECHA_DEF)
    - 3 models each: Logistic Regression, Random Forest, XGBoost
    - Stratified 80/20 split + 5-fold CV on training set
    - Bootstrap 95% CI on test set

Mexico dataset encoding:
    1 = YES
    2 = NO
    97 = Not applicable
    98 = Ignored
    99 = Unknown

Output (in /home/marko-b2/COVID_AI_Project/04_Scale_A_Mexico/):
    - mexico_hospitalized_cleaned.csv
    - scale_a_metrics.csv
    - scale_a_coefficients.csv (LR OR and CI)
    - figures/A1_ROC_comparison.png
    - figures/A2_feature_importance.png
    - figures/A3_OR_forest_plot.png
    - scale_a_log.txt
===============================================================================
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import xgboost as xgb
from scipy import stats

# ============================================================================
# CONFIG
# ============================================================================
MEXICO_CSV = Path('/home/marko-b2/COVID_DATASETS/mexico_covid19.csv')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/04_Scale_A_Mexico')
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'figures').mkdir(exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

RANDOM_STATE = 42

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE A — MEXICO POPULATION-LEVEL BASELINE MODEL')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load and filter to hospitalized
# ============================================================================
log('\n[STEP 1] Loading Mexico CSV...')
df = pd.read_csv(MEXICO_CSV, low_memory=False)
log(f'  Total rows: {len(df):,}')

# Filter to hospitalized (TIPO_PACIENTE == 2) and SARS-CoV-2 positive (RESULTADO == 1)
log('\n[STEP 2] Filtering to hospitalized SARS-CoV-2 positive patients...')
log(f'  RESULTADO distribution: {df["RESULTADO"].value_counts().to_dict()}')
# RESULTADO: 1=positive, 2=negative, 3=pending
df = df[df['TIPO_PACIENTE'] == 2].copy()
log(f'  After TIPO_PACIENTE==2 (hospitalized): {len(df):,}')
df = df[df['RESULTADO'] == 1].copy()
log(f'  After RESULTADO==1 (SARS-CoV-2 positive): {len(df):,}')

# ============================================================================
# STEP 3: Build outcome variables
# ============================================================================
log('\n[STEP 3] Building outcome variables...')

# ICU outcome: UCI == 1 → positive, 2 → negative, 97/99 → NaN (drop)
df['ICU'] = df['UCI'].map({1: 1, 2: 0}).astype('Int64')
log(f'  ICU outcome:')
log(f'    ICU=1 (admitted): {(df["ICU"]==1).sum():,} ({(df["ICU"]==1).mean()*100:.1f}%)')
log(f'    ICU=0 (not admitted): {(df["ICU"]==0).sum():,} ({(df["ICU"]==0).mean()*100:.1f}%)')
log(f'    Missing (97/99): {df["ICU"].isna().sum():,}')

# Mortality outcome: FECHA_DEF != '9999-99-99' → died
df['MORTALITY'] = (df['FECHA_DEF'] != '9999-99-99').astype(int)
log(f'\n  Mortality outcome:')
log(f'    Died: {(df["MORTALITY"]==1).sum():,} ({(df["MORTALITY"]==1).mean()*100:.1f}%)')
log(f'    Survived: {(df["MORTALITY"]==0).sum():,} ({(df["MORTALITY"]==0).mean()*100:.1f}%)')

# ============================================================================
# STEP 4: Build feature matrix
# ============================================================================
log('\n[STEP 4] Building feature matrix (demographics + comorbidities)...')

# Map 1=YES, 2=NO to 1/0; 97/98/99 to NaN (treat as unknown)
def clean_binary(series):
    return series.map({1: 1, 2: 0})

# Sex: 1=M, 2=F → 1=M, 0=F
df['SEX_MALE'] = df['SEXO'].map({1: 1, 2: 0})

# Age
df['AGE'] = pd.to_numeric(df['EDAD'], errors='coerce')
log(f'  AGE: range {df["AGE"].min()}-{df["AGE"].max()}, mean {df["AGE"].mean():.1f}')

# Comorbidities — all binarized with clean_binary
comorbid_cols = {
    'DIABETES': 'diabetes',
    'HIPERTENSION': 'hypertension',
    'OBESIDAD': 'obesity',
    'EPOC': 'copd',
    'ASMA': 'asthma',
    'CARDIOVASCULAR': 'cardiovascular',
    'RENAL_CRONICA': 'chronic_renal',
    'INMUSUPR': 'immunosuppressed',
}

for src, tgt in comorbid_cols.items():
    df[tgt] = clean_binary(df[src])
    n_yes = (df[tgt] == 1).sum()
    pct = n_yes / df[tgt].notna().sum() * 100
    log(f'  {tgt:20s}: {n_yes:6,} yes ({pct:.1f}%), missing: {df[tgt].isna().sum():,}')

# Pneumonia on admission (NEUMONIA) — critical predictor
df['pneumonia'] = clean_binary(df['NEUMONIA'])
n_pneu = (df['pneumonia'] == 1).sum()
log(f'  {"pneumonia":20s}: {n_pneu:6,} yes ({n_pneu/df["pneumonia"].notna().sum()*100:.1f}%)')

# Smoking
df['smoking'] = clean_binary(df['TABAQUISMO'])

# Final feature list
features = ['AGE', 'SEX_MALE', 'diabetes', 'hypertension', 'obesity', 'copd',
            'asthma', 'cardiovascular', 'chronic_renal', 'immunosuppressed',
            'pneumonia', 'smoking']
log(f'\n  Total features: {len(features)}')

# Build final working dataset — drop rows with too many NaN
work = df[features + ['ICU', 'MORTALITY']].copy()
work = work.dropna(subset=['AGE'])  # must have age
log(f'  After dropping missing age: {len(work):,}')

# Impute binary features (median, which for binary is majority)
for f in features:
    if f == 'AGE':
        continue
    if work[f].isna().any():
        med = work[f].median()
        n_imp = work[f].isna().sum()
        work[f] = work[f].fillna(med)

log(f'  Final dataset: {len(work):,} patients, {len(features)} features')

# Save cleaned dataset
work.to_csv(OUT_DIR / 'mexico_hospitalized_cleaned.csv', index=False)
log(f'  Saved: mexico_hospitalized_cleaned.csv')

# ============================================================================
# STEP 5: Train models for BOTH outcomes
# ============================================================================
log('\n[STEP 5] Training models for BOTH outcomes...')

X = work[features].values
all_results = {}
all_probas = {}

for outcome_name in ['ICU', 'MORTALITY']:
    log(f'\n{"="*60}')
    log(f'OUTCOME: {outcome_name}')
    log('='*60)
    
    # Subset to patients with known outcome
    mask = work[outcome_name].notna()
    Xo = X[mask]
    yo = work.loc[mask, outcome_name].astype(int).values
    log(f'  N patients with known outcome: {len(yo):,}')
    log(f'  Event rate: {yo.mean()*100:.2f}%')
    
    # Train/test split (80/20 stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xo, yo, test_size=0.2, stratify=yo, random_state=RANDOM_STATE
    )
    log(f'  Train: {len(X_tr):,}, Test: {len(X_te):,}')
    
    # Scale for LR
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    # 5-fold CV setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # --------- Logistic Regression ---------
    log(f'\n  [{outcome_name}] Training Logistic Regression...')
    lr = LogisticRegression(max_iter=2000, class_weight='balanced',
                             random_state=RANDOM_STATE, C=1.0)
    cv_aucs = cross_val_score(lr, X_tr_sc, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
    lr.fit(X_tr_sc, y_tr)
    lr_proba = lr.predict_proba(X_te_sc)[:, 1]
    lr_auc = roc_auc_score(y_te, lr_proba)
    log(f'    CV AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f} | Test AUC: {lr_auc:.3f}')
    
    # --------- Random Forest ---------
    log(f'\n  [{outcome_name}] Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=20,
                                  min_samples_leaf=10, class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)
    cv_aucs_rf = cross_val_score(rf, X_tr, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_proba = rf.predict_proba(X_te)[:, 1]
    rf_auc = roc_auc_score(y_te, rf_proba)
    log(f'    CV AUC: {cv_aucs_rf.mean():.3f} ± {cv_aucs_rf.std():.3f} | Test AUC: {rf_auc:.3f}')
    
    # --------- XGBoost ---------
    log(f'\n  [{outcome_name}] Training XGBoost...')
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
                                    random_state=RANDOM_STATE, eval_metric='logloss',
                                    verbosity=0)
    cv_aucs_xgb = cross_val_score(xgb_model, X_tr, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
    xgb_model.fit(X_tr, y_tr)
    xgb_proba = xgb_model.predict_proba(X_te)[:, 1]
    xgb_auc = roc_auc_score(y_te, xgb_proba)
    log(f'    CV AUC: {cv_aucs_xgb.mean():.3f} ± {cv_aucs_xgb.std():.3f} | Test AUC: {xgb_auc:.3f}')
    
    # Store results
    all_results[outcome_name] = {
        'LR': {'model': lr, 'scaler': scaler, 'proba': lr_proba,
               'cv_auc': cv_aucs.mean(), 'cv_std': cv_aucs.std(), 'test_auc': lr_auc,
               'brier': brier_score_loss(y_te, lr_proba)},
        'RF': {'model': rf, 'proba': rf_proba,
               'cv_auc': cv_aucs_rf.mean(), 'cv_std': cv_aucs_rf.std(), 'test_auc': rf_auc,
               'brier': brier_score_loss(y_te, rf_proba)},
        'XGBoost': {'model': xgb_model, 'proba': xgb_proba,
                     'cv_auc': cv_aucs_xgb.mean(), 'cv_std': cv_aucs_xgb.std(),
                     'test_auc': xgb_auc,
                     'brier': brier_score_loss(y_te, xgb_proba)},
        'y_te': y_te, 'X_tr_sc': X_tr_sc, 'X_tr': X_tr, 'y_tr': y_tr,
        'X_te_sc': X_te_sc, 'X_te': X_te
    }

# ============================================================================
# STEP 6: Bootstrap 95% CI for best model AUC
# ============================================================================
log('\n[STEP 6] Bootstrap 95% CI for each model...')
rng = np.random.RandomState(42)

def boot_ci(y_true, y_proba, n_iter=500):
    aucs = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
        except:
            pass
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

metrics_rows = []
for outcome_name in ['ICU', 'MORTALITY']:
    y_te = all_results[outcome_name]['y_te']
    for model_name in ['LR', 'RF', 'XGBoost']:
        r = all_results[outcome_name][model_name]
        ci_l, ci_h = boot_ci(y_te, r['proba'])
        log(f'  [{outcome_name}] {model_name}: AUC = {r["test_auc"]:.3f} [95% CI {ci_l:.3f}-{ci_h:.3f}]')
        metrics_rows.append({
            'Outcome': outcome_name, 'Model': model_name,
            'CV_AUC': round(r['cv_auc'], 3),
            'CV_std': round(r['cv_std'], 3),
            'Test_AUC': round(r['test_auc'], 3),
            'CI_low': round(ci_l, 3),
            'CI_high': round(ci_h, 3),
            'Brier': round(r['brier'], 3)
        })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / 'scale_a_metrics.csv', index=False)
log(f'\n  Saved: scale_a_metrics.csv')

# ============================================================================
# STEP 7: Logistic regression coefficients (with SE and CI) — for paper
# ============================================================================
log('\n[STEP 7] Computing LR coefficients with 95% CI...')

coef_rows = []
for outcome_name in ['ICU', 'MORTALITY']:
    r = all_results[outcome_name]
    X_tr_sc = r['X_tr_sc']
    y_tr = r['y_tr']
    lr = r['LR']['model']
    coefs = lr.coef_[0]
    
    # Standard errors via Fisher information
    p_hat = lr.predict_proba(X_tr_sc)[:, 1]
    X_int = np.hstack([np.ones((X_tr_sc.shape[0], 1)), X_tr_sc])
    W = np.diag(p_hat * (1 - p_hat))
    try:
        cov = np.linalg.inv(X_int.T @ W @ X_int)
        se_all = np.sqrt(np.diag(cov))
        se = se_all[1:]
    except np.linalg.LinAlgError:
        se = np.full_like(coefs, np.nan)
    
    for i, feat in enumerate(features):
        c, s = coefs[i], se[i]
        or_ = np.exp(c)
        if np.isnan(s):
            lo = hi = p_val = np.nan
        else:
            lo = np.exp(c - 1.96 * s)
            hi = np.exp(c + 1.96 * s)
            z = c / s
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        coef_rows.append({
            'Outcome': outcome_name, 'Feature': feat,
            'Coef': round(c, 3),
            'OR': round(or_, 2),
            'CI_low': round(lo, 2) if not np.isnan(lo) else 'NA',
            'CI_high': round(hi, 2) if not np.isnan(hi) else 'NA',
            'p_value': round(p_val, 4) if not np.isnan(p_val) else 'NA'
        })

coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(OUT_DIR / 'scale_a_coefficients.csv', index=False)
log(f'  Saved: scale_a_coefficients.csv')

# Print top coefficients for each outcome
for outcome_name in ['ICU', 'MORTALITY']:
    sub = coef_df[coef_df['Outcome']==outcome_name].copy()
    sub['abs_coef'] = sub['Coef'].abs()
    sub = sub.sort_values('abs_coef', ascending=False)
    log(f'\n  Top predictors for {outcome_name}:')
    for _, row in sub.head(7).iterrows():
        direction = '↑' if row['Coef'] > 0 else '↓'
        p_str = f"p={row['p_value']}" if row['p_value'] != 'NA' else 'p=NA'
        log(f"    {direction} {row['Feature']:20s}  OR={row['OR']}  [{row['CI_low']}-{row['CI_high']}]  {p_str}")

# ============================================================================
# STEP 8: Figure A1 — ROC curves comparison
# ============================================================================
log('\n[STEP 8] Figure A1 — ROC curves...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, outcome_name in zip(axes, ['ICU', 'MORTALITY']):
    y_te = all_results[outcome_name]['y_te']
    colors = {'LR': '#1f77b4', 'RF': '#2ca02c', 'XGBoost': '#E24A4A'}
    for model_name in ['LR', 'RF', 'XGBoost']:
        r = all_results[outcome_name][model_name]
        fpr, tpr, _ = roc_curve(y_te, r['proba'])
        ci_row = metrics_df[(metrics_df['Outcome']==outcome_name) & (metrics_df['Model']==model_name)].iloc[0]
        ax.plot(fpr, tpr, color=colors[model_name], lw=2,
                label=f'{model_name} (AUC={r["test_auc"]:.3f} [{ci_row["CI_low"]:.3f}-{ci_row["CI_high"]:.3f}])')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    event_rate = y_te.mean() * 100
    ax.set_title(f'{outcome_name} (N={len(y_te):,}, event rate={event_rate:.1f}%)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Figure A1 — Scale A (Mexico) baseline demographic model ROC curves',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'A1_ROC_comparison.png', dpi=300)
plt.close()
log('  Saved: A1_ROC_comparison.png')

# ============================================================================
# STEP 9: Figure A2 — Feature importance (XGBoost + LR)
# ============================================================================
log('\n[STEP 9] Figure A2 — feature importance...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, outcome_name in zip(axes, ['ICU', 'MORTALITY']):
    xgb_model = all_results[outcome_name]['XGBoost']['model']
    imp = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax.barh(imp['feature'], imp['importance'], color='#E24A4A', edgecolor='black')
    ax.set_xlabel('XGBoost feature importance', fontsize=11)
    ax.set_title(f'{outcome_name}', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Figure A2 — Scale A feature importance by XGBoost', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'A2_feature_importance.png', dpi=300)
plt.close()
log('  Saved: A2_feature_importance.png')

# ============================================================================
# STEP 10: Figure A3 — Forest plot of ORs (LR, both outcomes)
# ============================================================================
log('\n[STEP 10] Figure A3 — Forest plots of ORs...')
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, outcome_name in zip(axes, ['ICU', 'MORTALITY']):
    sub = coef_df[coef_df['Outcome']==outcome_name].copy()
    # Only include those with valid CI
    sub = sub[sub['CI_low'] != 'NA'].copy()
    sub['OR'] = pd.to_numeric(sub['OR'])
    sub['CI_low'] = pd.to_numeric(sub['CI_low'])
    sub['CI_high'] = pd.to_numeric(sub['CI_high'])
    sub = sub.sort_values('OR')
    
    y_pos = np.arange(len(sub))
    for i, row in enumerate(sub.itertuples()):
        color = '#E24A4A' if row.OR > 1 else '#4A90E2'
        ax.plot([row.CI_low, row.CI_high], [i, i], color='k', lw=1.4)
        ax.plot([row.CI_low, row.CI_low], [i-0.12, i+0.12], color='k', lw=1.4)
        ax.plot([row.CI_high, row.CI_high], [i-0.12, i+0.12], color='k', lw=1.4)
        ax.scatter(row.OR, i, s=80, c=color, edgecolor='black', zorder=3)
    
    ax.axvline(1, color='gray', linestyle='--', lw=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub['Feature'].tolist(), fontsize=10)
    ax.set_xlabel('Odds Ratio (95% CI) — standardized', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(f'{outcome_name}', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle(f'Figure A3 — Scale A logistic regression odds ratios',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'A3_OR_forest_plot.png', dpi=300)
plt.close()
log('  Saved: A3_OR_forest_plot.png')

# ============================================================================
# STEP 11: Save models
# ============================================================================
log('\n[STEP 11] Saving models...')
for outcome_name in ['ICU', 'MORTALITY']:
    r = all_results[outcome_name]
    for model_name in ['LR', 'RF', 'XGBoost']:
        with open(OUT_DIR / f'{outcome_name}_{model_name}.pkl', 'wb') as f:
            pickle.dump(r[model_name]['model'], f)
    with open(OUT_DIR / f'{outcome_name}_scaler.pkl', 'wb') as f:
        pickle.dump(r['LR']['scaler'], f)

with open(OUT_DIR / 'feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)
log(f'  Saved all models as .pkl')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
log('\n' + '='*75)
log('SCALE A COMPLETE')
log(f'Finished: {datetime.now().isoformat()}')
log('='*75)

log('\n=== SCALE A SUMMARY TABLE ===')
log(metrics_df.to_string(index=False))

log('\n=== KEY INTERPRETATION ===')
for outcome_name in ['ICU', 'MORTALITY']:
    best = metrics_df[metrics_df['Outcome']==outcome_name].sort_values('Test_AUC', ascending=False).iloc[0]
    log(f'\n  {outcome_name}: best model = {best["Model"]}, AUC = {best["Test_AUC"]}')
    log(f'  Event rate: {(all_results[outcome_name]["y_te"].mean()*100):.1f}%')

log('\n  Scale A provides a "demographic + comorbidity" baseline AUC.')
log('  The gap between Scale A and Scale C/D AUC quantifies the incremental')
log('  predictive value of laboratory biomarkers.')

with open(OUT_DIR / 'scale_a_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
