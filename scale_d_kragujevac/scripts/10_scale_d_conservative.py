"""
===============================================================================
Scale D — CONSERVATIVE ML model on Kragujevac cohort
===============================================================================
After detecting leakage (RTG_score = last measurement), we restrict features
to guaranteed BASELINE (pre-outcome) variables only:
    - Demographics: age, sex
    - Admission vitals: SpO2, pO2 on admission (explicitly baseline in colname)
    - Comorbidities: binary indicators of chronic conditions
    - Vaccination status

NOT used (potential leakage or treatment):
    - All lab values (CRP, LDH, urea...) — unknown timing protocol
    - RTG_score (confirmed leakage)
    - All treatments (confounded by indication)

Design:
    - N=93, 30 events
    - 11 features, 2.7 events/feature (conservative)
    - LOOCV + Bootstrap 95% CI
    - LR (primary) + XGBoost (sensitivity)

Output:
    /home/marko-b2/COVID_AI_Project/03_External_Validation/conservative/
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
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

# ============================================================================
# CONFIG
# ============================================================================
KRAGUJEVAC_FILE = Path('/home/marko-b2/genetika_COVID19_v2.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/03_External_Validation/conservative')
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
log('SCALE D — CONSERVATIVE ML (BASELINE FEATURES ONLY)')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load and prepare
# ============================================================================
log('\n[STEP 1] Loading Kragujevac data...')
df = pd.read_excel(KRAGUJEVAC_FILE)
df = df.dropna(subset=['godine']).reset_index(drop=True)
df['komorbiditeti_bin'] = df['komorbiditeti'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '0' or str(x).strip().lower() == 'ne') else 1
)
df['severe_outcome'] = ((df['HFV'] == 1) | (df['MV'] == 1) | (df['smrtni_ishod'] == 1)).astype(int)

log(f'  N = {len(df)} patients')
log(f'  Severe outcomes: {df["severe_outcome"].sum()} ({df["severe_outcome"].mean()*100:.1f}%)')

# ============================================================================
# STEP 2: Select ONLY guaranteed baseline features
# ============================================================================
log('\n[STEP 2] Selecting guaranteed baseline features...')

baseline_features = [
    # Demographics
    'godine',           # age
    'pol',              # sex
    # Admission vitals (names explicitly say "na prijemu" = on admission)
    'SAT_O2',           # SpO2 on admission
    'pO2',              # pO2 on admission
    # Chronic comorbidities (not acute)
    'komorbiditeti_bin',
    'DM',
    'HTA',
    'HOBP',
    'neuroloska_dg',
    'maligna_Dg',
    'HRI',
    'vakcinacija',
]

available = [c for c in baseline_features if c in df.columns]
log(f'  Baseline features used ({len(available)}):')
for c in available:
    pct_miss = 100 * df[c].isna().sum() / len(df)
    log(f'    - {c:25s} ({pct_miss:.0f}% missing)')

X = df[available].copy()
y = df['severe_outcome'].values

# Simple median imputation for the few missing values
for col in available:
    if X[col].isna().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        log(f'    Imputed {col} with median = {median_val:.2f}')

# ============================================================================
# STEP 3: Train models with LOOCV
# ============================================================================
log('\n[STEP 3] Training models with LOOCV...')

loo = LeaveOneOut()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

# Primary: Logistic Regression with L2 regularization
log('\n  === Logistic Regression (L2, C=1.0) ===')
lr = LogisticRegression(max_iter=2000, class_weight='balanced',
                         random_state=RANDOM_STATE, C=1.0)
lr_proba = cross_val_predict(lr, X_scaled, y, cv=loo, method='predict_proba')[:, 1]
lr_auc = roc_auc_score(y, lr_proba)
lr_brier = brier_score_loss(y, lr_proba)
log(f'  LOOCV AUC: {lr_auc:.3f}, Brier: {lr_brier:.3f}')

# Sensitivity: XGBoost (shallow)
log('\n  === XGBoost (shallow, regularized) ===')
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=2, learning_rate=0.05,
    reg_alpha=1.0, reg_lambda=2.0,  # Strong regularization
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0
)
xgb_proba = cross_val_predict(xgb_model, X.values, y, cv=loo, method='predict_proba')[:, 1]
xgb_auc = roc_auc_score(y, xgb_proba)
xgb_brier = brier_score_loss(y, xgb_proba)
log(f'  LOOCV AUC: {xgb_auc:.3f}, Brier: {xgb_brier:.3f}')

# Ultra-minimal baseline: age + SpO2 only
log('\n  === Ultra-minimal (age + SpO2 only) ===')
X_min = X[['godine', 'SAT_O2']].values
X_min_scaled = StandardScaler().fit_transform(X_min)
lr_min = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr_min_proba = cross_val_predict(lr_min, X_min_scaled, y, cv=loo, method='predict_proba')[:, 1]
lr_min_auc = roc_auc_score(y, lr_min_proba)
log(f'  LOOCV AUC: {lr_min_auc:.3f}')

# ============================================================================
# STEP 4: Bootstrap 95% CI
# ============================================================================
log('\n[STEP 4] Bootstrap 95% CI...')
rng = np.random.RandomState(42)

def bootstrap_ci(y_true, y_proba, n_iter=1000):
    aucs = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
        except:
            pass
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

lr_ci_low, lr_ci_high = bootstrap_ci(y, lr_proba)
xgb_ci_low, xgb_ci_high = bootstrap_ci(y, xgb_proba)
min_ci_low, min_ci_high = bootstrap_ci(y, lr_min_proba)

log(f'  LR (11 features):        AUC = {lr_auc:.3f} [95% CI {lr_ci_low:.3f}-{lr_ci_high:.3f}]')
log(f'  XGBoost (11 features):   AUC = {xgb_auc:.3f} [95% CI {xgb_ci_low:.3f}-{xgb_ci_high:.3f}]')
log(f'  Ultra-minimal (2 feat):  AUC = {lr_min_auc:.3f} [95% CI {min_ci_low:.3f}-{min_ci_high:.3f}]')

# ============================================================================
# STEP 5: Feature importance (LR coefficients)
# ============================================================================
log('\n[STEP 5] Feature importance (LR, trained on all data)...')
lr_final = LogisticRegression(max_iter=2000, class_weight='balanced',
                               random_state=RANDOM_STATE, C=1.0)
lr_final.fit(X_scaled, y)

coef_df = pd.DataFrame({
    'feature': available,
    'coefficient': lr_final.coef_[0],
    'odds_ratio': np.exp(lr_final.coef_[0]),
    'abs_coef': np.abs(lr_final.coef_[0])
}).sort_values('abs_coef', ascending=False)

log('\n  Feature coefficients (standardized, sorted by |coef|):')
for _, r in coef_df.iterrows():
    direction = '↑' if r['coefficient'] > 0 else '↓'
    log(f'    {direction} {r["feature"]:20s}  coef={r["coefficient"]:+.3f}  OR={r["odds_ratio"]:.2f}')

coef_df.to_csv(OUT_DIR / 'lr_coefficients.csv', index=False)

# ============================================================================
# STEP 6: Save metrics
# ============================================================================
metrics_rows = [
    {'Model': 'LR (11 baseline features)', 'AUC': round(lr_auc, 3),
     'CI_low': round(lr_ci_low, 3), 'CI_high': round(lr_ci_high, 3),
     'Brier': round(lr_brier, 3), 'N_features': len(available)},
    {'Model': 'XGBoost (11 baseline features)', 'AUC': round(xgb_auc, 3),
     'CI_low': round(xgb_ci_low, 3), 'CI_high': round(xgb_ci_high, 3),
     'Brier': round(xgb_brier, 3), 'N_features': len(available)},
    {'Model': 'LR (age + SpO2 only)', 'AUC': round(lr_min_auc, 3),
     'CI_low': round(min_ci_low, 3), 'CI_high': round(min_ci_high, 3),
     'Brier': 'NA', 'N_features': 2},
]
pd.DataFrame(metrics_rows).to_csv(OUT_DIR / 'conservative_metrics.csv', index=False)

# ============================================================================
# STEP 7: Save predictions
# ============================================================================
preds = df[['godine', 'pol', 'rs11549465', 'rs41508050', 'severe_outcome',
            'HFV', 'MV', 'smrtni_ishod']].copy()
preds['lr_proba'] = lr_proba
preds['xgb_proba'] = xgb_proba
preds['lr_min_proba'] = lr_min_proba
preds.to_csv(OUT_DIR / 'conservative_predictions.csv', index=False)

# ============================================================================
# STEP 8: Figure D10 — ROC curves
# ============================================================================
log('\n[STEP 6] Figure D10 — Conservative ROC curves...')
fig, ax = plt.subplots(figsize=(8, 7))

for proba, auc_val, ci_l, ci_h, name, color in [
    (lr_proba, lr_auc, lr_ci_low, lr_ci_high, 'LR (11 features)', '#1f77b4'),
    (xgb_proba, xgb_auc, xgb_ci_low, xgb_ci_high, 'XGBoost (11 features)', '#E24A4A'),
    (lr_min_proba, lr_min_auc, min_ci_low, min_ci_high, 'LR (age + SpO2)', '#2ca02c'),
]:
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{name} (AUC={auc_val:.3f} [{ci_l:.3f}-{ci_h:.3f}])')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'Figure D10 — Conservative Scale D model (N={len(df)}, baseline features only)', fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D10_conservative_ROC.png', dpi=300)
plt.close()
log('  Saved: D10_conservative_ROC.png')

# ============================================================================
# STEP 9: Figure D11 — coefficient plot
# ============================================================================
log('\n[STEP 7] Figure D11 — Feature coefficients...')
fig, ax = plt.subplots(figsize=(9, 6))
top = coef_df.head(len(available))
colors_bar = ['#E24A4A' if c > 0 else '#4A90E2' for c in top['coefficient']]
ax.barh(top['feature'][::-1], top['coefficient'][::-1], color=colors_bar[::-1],
        edgecolor='black')
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Logistic regression coefficient (standardized)', fontsize=11)
ax.set_title(f'Figure D11 — Baseline predictors of severe COVID-19 outcome (N={len(df)})', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E24A4A', label='↑ Risk (coef > 0)'),
    Patch(facecolor='#4A90E2', label='↓ Risk (coef < 0)')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D11_coefficients.png', dpi=300)
plt.close()
log('  Saved: D11_coefficients.png')

# ============================================================================
# STEP 10: HIF1A stratification check
# ============================================================================
log('\n[STEP 8] Performance stratified by HIF1A genotype...')
df['lr_proba'] = lr_proba

strat_rows = []
for snp in ['rs11549465', 'rs41508050']:
    for gen_val, gen_label in [(0, 'CC'), (1, 'CT')]:
        sub = df[df[snp] == gen_val]
        if len(sub) < 5:
            continue
        if sub['severe_outcome'].nunique() < 2:
            log(f'  {snp} {gen_label}: only 1 outcome class (n={len(sub)})')
            continue
        sub_auc = roc_auc_score(sub['severe_outcome'], sub['lr_proba'])
        log(f'  {snp} {gen_label} (n={len(sub)}): AUC={sub_auc:.3f}, event rate={sub["severe_outcome"].mean():.3f}')
        strat_rows.append({'SNP': snp, 'genotype': gen_label, 'N': len(sub),
                          'AUC': round(sub_auc, 3),
                          'event_rate': round(sub['severe_outcome'].mean(), 3)})

pd.DataFrame(strat_rows).to_csv(OUT_DIR / 'stratified_performance.csv', index=False)

# ============================================================================
# DONE
# ============================================================================
log('\n' + '='*75)
log(f'SCALE D CONSERVATIVE COMPLETE: {datetime.now().isoformat()}')
log(f'Output: {OUT_DIR}')
log('='*75)

log('\n=== FINAL SUMMARY ===')
log(f'Primary model: LR with 11 baseline features')
log(f'  AUC = {lr_auc:.3f} [95% CI {lr_ci_low:.3f}-{lr_ci_high:.3f}]')
log(f'  Compared to chance (0.50): +{(lr_auc-0.5):.3f}')
log(f'')
log(f'All features used are PRE-OUTCOME (no leakage possible):')
for c in available:
    log(f'  - {c}')
log(f'')
if lr_auc < 0.65:
    log('INTERPRETATION: Limited signal in baseline features alone.')
    log('  Expected — severe COVID-19 outcome is driven by dynamic lab changes')
    log('  that we cannot use until we verify timing protocol with coauthors.')
elif lr_auc < 0.75:
    log('INTERPRETATION: Modest signal — acceptable for publication as baseline model.')
    log('  Lab values would likely improve this further if timing is validated.')
else:
    log('INTERPRETATION: Strong baseline signal. Suspicious — double-check for')
    log('  hidden leakage (e.g., "vakcinacija" might correlate with later events).')

with open(OUT_DIR / 'conservative_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
