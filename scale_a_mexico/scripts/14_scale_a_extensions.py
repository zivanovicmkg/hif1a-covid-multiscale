"""
===============================================================================
Scale A — Extension analyses
===============================================================================
Three supplementary analyses to resolve paradoxes identified in 13_scale_a:
    1. Composite severe outcome (ICU OR mortality) — eliminates competing risks
    2. Age-stratified analysis (<50, 50-64, 65+) — check if paradoxes persist
    3. Model without pneumonia — check if pneumonia absorbs other signals
===============================================================================
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from scipy import stats

DATA_FILE = Path('/home/marko-b2/COVID_AI_Project/04_Scale_A_Mexico/mexico_hospitalized_cleaned.csv')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/04_Scale_A_Mexico')
(OUT_DIR / 'figures').mkdir(exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE A EXTENSIONS — composite outcome, age strata, drop-pneumonia')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# Load cleaned data
df = pd.read_csv(DATA_FILE)
log(f'\nLoaded: N={len(df):,} patients')

features_full = ['AGE', 'SEX_MALE', 'diabetes', 'hypertension', 'obesity', 'copd',
                  'asthma', 'cardiovascular', 'chronic_renal', 'immunosuppressed',
                  'pneumonia', 'smoking']

# =============================================================================
# ANALYSIS 1: Composite outcome (ICU OR mortality)
# =============================================================================
log('\n' + '='*75)
log('[ANALYSIS 1] COMPOSITE SEVERE OUTCOME (ICU OR Mortality)')
log('='*75)

# Composite: severe = 1 if EITHER ICU OR died
df['SEVERE'] = ((df['ICU'] == 1) | (df['MORTALITY'] == 1)).astype(int)
# Drop rows where ICU is missing (can't fully assess severity)
df_s = df.dropna(subset=['ICU']).copy()
log(f'N with complete outcomes: {len(df_s):,}')
log(f'SEVERE=1: {(df_s["SEVERE"]==1).sum():,} ({df_s["SEVERE"].mean()*100:.1f}%)')
log(f'  - ICU only: {((df_s["ICU"]==1) & (df_s["MORTALITY"]==0)).sum():,}')
log(f'  - Mortality only: {((df_s["ICU"]==0) & (df_s["MORTALITY"]==1)).sum():,}')
log(f'  - Both ICU + died: {((df_s["ICU"]==1) & (df_s["MORTALITY"]==1)).sum():,}')

X = df_s[features_full].values
y = df_s['SEVERE'].values

# 80/20 split + scale + train LR
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
cv_aucs = cross_val_score(lr, X_tr_sc, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
lr.fit(X_tr_sc, y_tr)
test_proba = lr.predict_proba(X_te_sc)[:, 1]
test_auc = roc_auc_score(y_te, test_proba)
log(f'\nComposite LR — CV AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f} | Test AUC: {test_auc:.3f}')

# Bootstrap CI
rng = np.random.RandomState(42)
boots = []
for _ in range(500):
    idx = rng.choice(len(y_te), len(y_te), replace=True)
    if len(np.unique(y_te[idx])) > 1:
        boots.append(roc_auc_score(y_te[idx], test_proba[idx]))
ci_l, ci_h = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
log(f'  95% CI: [{ci_l:.3f}-{ci_h:.3f}]')

# Coefficients with CIs
coefs = lr.coef_[0]
p_hat = lr.predict_proba(X_tr_sc)[:, 1]
X_int = np.hstack([np.ones((X_tr_sc.shape[0], 1)), X_tr_sc])
W = np.diag(p_hat * (1 - p_hat))
try:
    cov = np.linalg.inv(X_int.T @ W @ X_int)
    se = np.sqrt(np.diag(cov))[1:]
except:
    se = np.full_like(coefs, np.nan)

rows = []
for i, feat in enumerate(features_full):
    c, s = coefs[i], se[i]
    or_ = np.exp(c)
    ci_l_f = np.exp(c - 1.96*s) if not np.isnan(s) else np.nan
    ci_h_f = np.exp(c + 1.96*s) if not np.isnan(s) else np.nan
    z = c/s if not np.isnan(s) else np.nan
    p_val = 2*(1 - stats.norm.cdf(abs(z))) if not np.isnan(s) else np.nan
    rows.append({'Feature': feat, 'Coef': round(c,3), 'OR': round(or_,2),
                 'CI_low': round(ci_l_f,2) if not np.isnan(ci_l_f) else 'NA',
                 'CI_high': round(ci_h_f,2) if not np.isnan(ci_h_f) else 'NA',
                 'p_value': round(p_val,4) if not np.isnan(p_val) else 'NA'})

comp_df = pd.DataFrame(rows).sort_values('Coef', key=abs, ascending=False)
log('\nTop predictors for COMPOSITE SEVERE outcome:')
for _, r in comp_df.head(8).iterrows():
    dir_ = '↑' if r['Coef'] > 0 else '↓'
    log(f"  {dir_} {r['Feature']:20s}  OR={r['OR']}  [{r['CI_low']}-{r['CI_high']}]  p={r['p_value']}")

comp_df.to_csv(OUT_DIR / 'scale_a_composite_coefficients.csv', index=False)

# =============================================================================
# ANALYSIS 2: Age-stratified
# =============================================================================
log('\n' + '='*75)
log('[ANALYSIS 2] AGE-STRATIFIED ANALYSIS (SEVERE outcome)')
log('='*75)

df_s['age_group'] = pd.cut(df_s['AGE'], bins=[0, 50, 65, 150],
                            labels=['<50', '50-64', '65+'])
log(f'\nAge distribution:')
for g in ['<50', '50-64', '65+']:
    sub = df_s[df_s['age_group']==g]
    log(f'  {g:>6s}: N={len(sub):,}, SEVERE rate={sub["SEVERE"].mean()*100:.1f}%')

strata_results = []
for age_grp in ['<50', '50-64', '65+']:
    sub = df_s[df_s['age_group'] == age_grp].copy()
    if len(sub) < 500:
        log(f'\n  Skipping {age_grp}: too few patients (N={len(sub)})')
        continue
    
    features_no_age = [f for f in features_full if f != 'AGE']
    X_s = sub[features_no_age].values
    y_s = sub['SEVERE'].values
    
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2,
                                                        stratify=y_s, random_state=42)
    sc = StandardScaler()
    X_tr_s_sc = sc.fit_transform(X_tr_s)
    X_te_s_sc = sc.transform(X_te_s)
    
    lr_s = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
    cv_s = cross_val_score(lr_s, X_tr_s_sc, y_tr_s, cv=skf, scoring='roc_auc', n_jobs=-1)
    lr_s.fit(X_tr_s_sc, y_tr_s)
    test_auc_s = roc_auc_score(y_te_s, lr_s.predict_proba(X_te_s_sc)[:, 1])
    log(f'\n  [{age_grp}] N={len(sub):,}, CV AUC: {cv_s.mean():.3f}±{cv_s.std():.3f}, Test AUC: {test_auc_s:.3f}')
    
    # Save top 3 predictors
    coefs_s = lr_s.coef_[0]
    top3 = sorted(zip(features_no_age, coefs_s), key=lambda x: -abs(x[1]))[:5]
    for f, c in top3:
        or_ = np.exp(c)
        dir_ = '↑' if c > 0 else '↓'
        log(f'    {dir_} {f:20s}  OR={or_:.2f}')
        strata_results.append({'age_group': age_grp, 'feature': f,
                                'OR': round(or_, 2), 'coef': round(c, 3)})

pd.DataFrame(strata_results).to_csv(OUT_DIR / 'scale_a_age_stratified.csv', index=False)

# =============================================================================
# ANALYSIS 3: Model without pneumonia
# =============================================================================
log('\n' + '='*75)
log('[ANALYSIS 3] MODEL WITHOUT PNEUMONIA (to check signal absorption)')
log('='*75)

features_no_pneu = [f for f in features_full if f != 'pneumonia']
X_np = df_s[features_no_pneu].values
y_np = df_s['SEVERE'].values

X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(X_np, y_np, test_size=0.2,
                                                         stratify=y_np, random_state=42)
sc_np = StandardScaler()
X_tr_np_sc = sc_np.fit_transform(X_tr_np)
X_te_np_sc = sc_np.transform(X_te_np)

lr_np = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
cv_np = cross_val_score(lr_np, X_tr_np_sc, y_tr_np, cv=skf, scoring='roc_auc', n_jobs=-1)
lr_np.fit(X_tr_np_sc, y_tr_np)
test_auc_np = roc_auc_score(y_te_np, lr_np.predict_proba(X_te_np_sc)[:, 1])
log(f'\nModel WITHOUT pneumonia — CV AUC: {cv_np.mean():.3f}, Test AUC: {test_auc_np:.3f}')

log(f'\n  AUC difference: with pneumonia {test_auc:.3f} vs without {test_auc_np:.3f}')
log(f'  ΔAUC = {test_auc - test_auc_np:+.3f} (pneumonia contribution)')

log('\n  Top predictors WITHOUT pneumonia:')
coefs_np = lr_np.coef_[0]
p_hat_np = lr_np.predict_proba(X_tr_np_sc)[:, 1]
X_int_np = np.hstack([np.ones((X_tr_np_sc.shape[0], 1)), X_tr_np_sc])
W_np = np.diag(p_hat_np * (1 - p_hat_np))
try:
    cov_np = np.linalg.inv(X_int_np.T @ W_np @ X_int_np)
    se_np = np.sqrt(np.diag(cov_np))[1:]
except:
    se_np = np.full_like(coefs_np, np.nan)

rows_np = []
for i, feat in enumerate(features_no_pneu):
    c, s = coefs_np[i], se_np[i]
    or_ = np.exp(c)
    ci_l_f = np.exp(c - 1.96*s) if not np.isnan(s) else np.nan
    ci_h_f = np.exp(c + 1.96*s) if not np.isnan(s) else np.nan
    rows_np.append({'Feature': feat, 'Coef': round(c,3), 'OR': round(or_,2),
                     'CI_low': round(ci_l_f,2) if not np.isnan(ci_l_f) else 'NA',
                     'CI_high': round(ci_h_f,2) if not np.isnan(ci_h_f) else 'NA'})

no_pneu_df = pd.DataFrame(rows_np).sort_values('Coef', key=abs, ascending=False)
for _, r in no_pneu_df.head(6).iterrows():
    dir_ = '↑' if r['Coef'] > 0 else '↓'
    log(f"    {dir_} {r['Feature']:20s}  OR={r['OR']}  [{r['CI_low']}-{r['CI_high']}]")

no_pneu_df.to_csv(OUT_DIR / 'scale_a_no_pneumonia_coefficients.csv', index=False)

# =============================================================================
# FIGURE A4 — Composite outcome forest plot + ΔAUC comparison
# =============================================================================
log('\n[FIGURE A4] Composite outcome forest plot + ΔAUC...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: composite forest plot
fp_df = comp_df[comp_df['CI_low'] != 'NA'].copy()
fp_df['OR'] = pd.to_numeric(fp_df['OR'])
fp_df['CI_low'] = pd.to_numeric(fp_df['CI_low'])
fp_df['CI_high'] = pd.to_numeric(fp_df['CI_high'])
fp_df = fp_df.sort_values('OR')

ax = axes[0]
for i, row in enumerate(fp_df.itertuples()):
    color = '#E24A4A' if row.OR > 1 else '#4A90E2'
    ax.plot([row.CI_low, row.CI_high], [i, i], color='k', lw=1.4)
    ax.plot([row.CI_low]*2, [i-0.12, i+0.12], color='k', lw=1.4)
    ax.plot([row.CI_high]*2, [i-0.12, i+0.12], color='k', lw=1.4)
    ax.scatter(row.OR, i, s=80, c=color, edgecolor='black', zorder=3)
ax.axvline(1, color='gray', linestyle='--', lw=1, alpha=0.7)
ax.set_yticks(range(len(fp_df)))
ax.set_yticklabels(fp_df['Feature'].tolist(), fontsize=10)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
ax.set_xscale('log')
ax.set_title(f'(A) Composite severe outcome (N={len(df_s):,}, ICU OR mortality)', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Right: ΔAUC comparison
ax = axes[1]
cats = ['ICU\nonly', 'Mortality\nonly', 'Composite\n(ICU OR died)', 'Composite\n(no pneumonia)']
aucs_vals = [0.661, 0.649, test_auc, test_auc_np]
colors = ['#5dade2', '#58d68d', '#e74c3c', '#f39c12']
bars = ax.bar(cats, aucs_vals, color=colors, edgecolor='black', linewidth=0.8)
ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6, label='Chance')
ax.set_ylabel('Test AUC', fontsize=11)
ax.set_ylim(0.45, max(aucs_vals)+0.05)
ax.set_title('(B) AUC comparison across outcome definitions', fontsize=11)
for bar, v in zip(bars, aucs_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v+0.005, f'{v:.3f}',
             ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Figure A4 — Scale A extension analyses', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'A4_extensions.png', dpi=300)
plt.close()
log('  Saved: A4_extensions.png')

# =============================================================================
# DONE
# =============================================================================
log('\n' + '='*75)
log('SCALE A EXTENSIONS COMPLETE')
log(f'Finished: {datetime.now().isoformat()}')
log('='*75)

log('\n=== SUMMARY ===')
log(f'1. Composite SEVERE outcome (ICU OR died): AUC = {test_auc:.3f} [{ci_l:.3f}-{ci_h:.3f}]')
log(f'   Event rate: {df_s["SEVERE"].mean()*100:.1f}% (N={len(df_s):,})')
log(f'2. Age strata AUCs: see scale_a_age_stratified.csv')
log(f'3. Without pneumonia: AUC = {test_auc_np:.3f}, ΔAUC = {test_auc - test_auc_np:+.3f}')

with open(OUT_DIR / 'scale_a_extensions_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
