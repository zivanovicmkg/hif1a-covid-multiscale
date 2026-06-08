"""
===============================================================================
Scale D — VIF + Drop-HTA sanity check
===============================================================================
Investigates collinearity among baseline features, particularly the
paradoxical negative HTA coefficient. If HTA is confounded with age,
its removal should yield similar AUC.
===============================================================================
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================================
KRAGUJEVAC_FILE = Path('/home/marko-b2/genetika_COVID19_v2.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/03_External_Validation/conservative')
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

print('='*75)
print('SCALE D — VIF + DROP-HTA SANITY CHECK')
print('='*75)

# ============================================================================
# Load data
# ============================================================================
df = pd.read_excel(KRAGUJEVAC_FILE)
df = df.dropna(subset=['godine']).reset_index(drop=True)
df['komorbiditeti_bin'] = df['komorbiditeti'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '0' or str(x).strip().lower() == 'ne') else 1
)
df['severe_outcome'] = ((df['HFV'] == 1) | (df['MV'] == 1) | (df['smrtni_ishod'] == 1)).astype(int)

baseline_features = [
    'godine', 'pol', 'SAT_O2', 'pO2', 'komorbiditeti_bin',
    'DM', 'HTA', 'HOBP', 'neuroloska_dg', 'maligna_Dg', 'HRI', 'vakcinacija',
]
X = df[baseline_features].copy()
y = df['severe_outcome'].values

# Impute missing
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

print(f'\nN = {len(df)}, Events = {int(y.sum())}')
print(f'Features = {len(baseline_features)}')

# ============================================================================
# STEP 1: VIF (without statsmodels — manual implementation)
# ============================================================================
print('\n[STEP 1] Computing VIF for each feature...')
print('VIF interpretation:')
print('  VIF < 5    = No problematic collinearity')
print('  VIF 5-10   = Moderate (acceptable, mention in Methods)')
print('  VIF > 10   = Serious collinearity (consider removal)')

def compute_vif(X_df):
    """Compute VIF manually: VIF_i = 1 / (1 - R²_i) where R² is from regressing feature i on others."""
    from sklearn.linear_model import LinearRegression
    vifs = {}
    for col in X_df.columns:
        y_reg = X_df[col].values
        X_reg = X_df.drop(columns=[col]).values
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        r2 = lr.score(X_reg, y_reg)
        if r2 >= 0.9999:
            vifs[col] = float('inf')
        else:
            vifs[col] = 1.0 / (1.0 - r2)
    return vifs

vifs = compute_vif(X)
vif_df = pd.DataFrame([
    {'feature': k, 'VIF': round(v, 2),
     'interpretation': 'OK' if v < 5 else ('MODERATE' if v < 10 else 'HIGH COLLINEARITY')}
    for k, v in sorted(vifs.items(), key=lambda x: -x[1])
])

print('\n' + vif_df.to_string(index=False))
vif_df.to_csv(OUT_DIR / 'vif_analysis.csv', index=False)
print(f'\nSaved: vif_analysis.csv')

# ============================================================================
# STEP 2: Correlation matrix figure
# ============================================================================
print('\n[STEP 2] Generating correlation matrix...')
corr = X.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={'label': 'Pearson correlation'}, ax=ax)
ax.set_title('Correlation matrix — Kragujevac baseline features (N=93)',
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D12_correlation_matrix.png', dpi=300)
plt.close()
print('  Saved: D12_correlation_matrix.png')

# Report highest correlations with HTA
print('\n  HTA correlations (top 5 by |r|):')
hta_corr = corr['HTA'].drop('HTA').abs().sort_values(ascending=False)
for feat, r in hta_corr.head(5).items():
    actual_r = corr.loc['HTA', feat]
    print(f'    HTA ↔ {feat:20s}: r = {actual_r:+.3f}')

# ============================================================================
# STEP 3: Drop-HTA sensitivity analysis
# ============================================================================
print('\n[STEP 3] Sensitivity analysis: LR without HTA...')

loo = LeaveOneOut()
scaler = StandardScaler()

# Original (with HTA)
X_full = scaler.fit_transform(X.values)
lr_full = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
proba_full = cross_val_predict(lr_full, X_full, y, cv=loo, method='predict_proba')[:, 1]
auc_full = roc_auc_score(y, proba_full)

# Without HTA
X_no_hta = X.drop(columns=['HTA'])
X_no_hta_scaled = scaler.fit_transform(X_no_hta.values)
lr_no_hta = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
proba_no_hta = cross_val_predict(lr_no_hta, X_no_hta_scaled, y, cv=loo, method='predict_proba')[:, 1]
auc_no_hta = roc_auc_score(y, proba_no_hta)

# Without komorbiditeti_bin (also collinear with DM/HTA individually)
X_no_comorb = X.drop(columns=['komorbiditeti_bin'])
X_no_comorb_scaled = scaler.fit_transform(X_no_comorb.values)
lr_no_comorb = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
proba_no_comorb = cross_val_predict(lr_no_comorb, X_no_comorb_scaled, y, cv=loo, method='predict_proba')[:, 1]
auc_no_comorb = roc_auc_score(y, proba_no_comorb)

# Without both HTA and komorbiditeti_bin
X_clean = X.drop(columns=['HTA', 'komorbiditeti_bin'])
X_clean_scaled = scaler.fit_transform(X_clean.values)
lr_clean = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
proba_clean = cross_val_predict(lr_clean, X_clean_scaled, y, cv=loo, method='predict_proba')[:, 1]
auc_clean = roc_auc_score(y, proba_clean)

# Bootstrap CIs
rng = np.random.RandomState(42)
def boot_ci(y_true, y_proba):
    aucs = []
    for _ in range(1000):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
        except:
            pass
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

ci_full = boot_ci(y, proba_full)
ci_no_hta = boot_ci(y, proba_no_hta)
ci_no_comorb = boot_ci(y, proba_no_comorb)
ci_clean = boot_ci(y, proba_clean)

print('\n=== SENSITIVITY ANALYSIS RESULTS ===')
print(f'{"Model":40s} {"AUC":>6s} {"95% CI":>18s}')
print('-' * 68)
print(f'{"Original (all 12 features)":40s} {auc_full:>6.3f} [{ci_full[0]:.3f}-{ci_full[1]:.3f}]')
print(f'{"Without HTA":40s} {auc_no_hta:>6.3f} [{ci_no_hta[0]:.3f}-{ci_no_hta[1]:.3f}]')
print(f'{"Without komorbiditeti_bin":40s} {auc_no_comorb:>6.3f} [{ci_no_comorb[0]:.3f}-{ci_no_comorb[1]:.3f}]')
print(f'{"Without HTA AND komorbiditeti_bin":40s} {auc_clean:>6.3f} [{ci_clean[0]:.3f}-{ci_clean[1]:.3f}]')

delta = auc_full - auc_no_hta
print(f'\n  ΔAUC (removing HTA) = {delta:+.3f}')
if abs(delta) < 0.02:
    print('  → INTERPRETATION: HTA removal changes AUC by <0.02')
    print('    HTA adds no real predictive value. Its paradoxical coefficient was')
    print('    a collinearity artifact. Safe to drop for Methods simplicity.')
elif abs(delta) < 0.05:
    print('  → INTERPRETATION: Small AUC change. HTA has marginal contribution.')
else:
    print('  → INTERPRETATION: HTA removal causes substantial AUC change.')
    print('    Coefficient direction may still be confounded, but variable matters.')

# ============================================================================
# STEP 4: Save results table
# ============================================================================
results = pd.DataFrame([
    {'Model': 'Original (all 12 features)', 'AUC': round(auc_full, 3),
     'CI_low': round(ci_full[0], 3), 'CI_high': round(ci_full[1], 3),
     'ΔAUC_vs_full': 0.0, 'N_features': 12},
    {'Model': 'Without HTA', 'AUC': round(auc_no_hta, 3),
     'CI_low': round(ci_no_hta[0], 3), 'CI_high': round(ci_no_hta[1], 3),
     'ΔAUC_vs_full': round(auc_no_hta - auc_full, 3), 'N_features': 11},
    {'Model': 'Without komorbiditeti_bin', 'AUC': round(auc_no_comorb, 3),
     'CI_low': round(ci_no_comorb[0], 3), 'CI_high': round(ci_no_comorb[1], 3),
     'ΔAUC_vs_full': round(auc_no_comorb - auc_full, 3), 'N_features': 11},
    {'Model': 'Without HTA AND komorbiditeti_bin', 'AUC': round(auc_clean, 3),
     'CI_low': round(ci_clean[0], 3), 'CI_high': round(ci_clean[1], 3),
     'ΔAUC_vs_full': round(auc_clean - auc_full, 3), 'N_features': 10},
])
results.to_csv(OUT_DIR / 'sensitivity_analysis.csv', index=False)
print(f'\nSaved: sensitivity_analysis.csv')

# Re-fit to see coefficients without HTA
print('\n[STEP 5] Coefficients without HTA (standardized):')
lr_no_hta_final = LogisticRegression(max_iter=2000, class_weight='balanced',
                                       C=1.0, random_state=42)
lr_no_hta_final.fit(X_no_hta_scaled, y)

coef_df = pd.DataFrame({
    'feature': X_no_hta.columns,
    'coefficient': lr_no_hta_final.coef_[0],
    'odds_ratio': np.exp(lr_no_hta_final.coef_[0]),
}).sort_values('coefficient', key=abs, ascending=False)

for _, r in coef_df.iterrows():
    direction = '↑' if r['coefficient'] > 0 else '↓'
    print(f'  {direction} {r["feature"]:20s}  coef={r["coefficient"]:+.3f}  OR={r["odds_ratio"]:.2f}')

coef_df.to_csv(OUT_DIR / 'coefficients_no_hta.csv', index=False)

print('\n' + '='*75)
print('DONE')
print('='*75)
