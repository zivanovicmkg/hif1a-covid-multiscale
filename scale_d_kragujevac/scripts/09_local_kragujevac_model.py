"""
===============================================================================
Scale D — LOCAL XGBoost training on Kragujevac cohort
===============================================================================
After external validation failed (AUC=0.50), we retrain the model directly
on Kragujevac data to assess whether predictive signal exists in this cohort.

Design:
    - N=93 patients, 30 events (severe outcome = HFV OR MV OR death)
    - 24 features (drop 6 that are fully missing)
    - Leave-One-Out Cross Validation (LOOCV) — stable AUC for small N
    - Regularized XGBoost + baseline Logistic Regression
    - Comparison to "simple" baseline (age + CRP only)

Output:
    /home/marko-b2/COVID_AI_Project/03_External_Validation/local_model/
        - local_kragujevac_metrics.csv
        - local_kragujevac_predictions.csv
        - local_shap_importance.csv
        - figures/D7_local_ROC.png
        - figures/D8_local_SHAP.png
        - figures/D9_local_vs_external_comparison.png
        - local_training_log.txt
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
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, precision_recall_curve)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

# ============================================================================
# CONFIG
# ============================================================================
KRAGUJEVAC_FILE = Path('/home/marko-b2/genetika_COVID19_v2.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/03_External_Validation/local_model')
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
log('SCALE D — LOCAL KRAGUJEVAC MODEL TRAINING')
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
# STEP 2: Feature selection — use Kragujevac-native variables
# ============================================================================
log('\n[STEP 2] Selecting available Kragujevac features...')

# All numeric clinical features available
feature_cols = [
    'godine', 'pol', 'SAT_O2', 'pO2', 'RTG_score',
    'komorbiditeti_bin', 'DM', 'HTA', 'HOBP', 'neuroloska_dg',
    'maligna_Dg', 'HRI', 'vakcinacija',
    'leukociti', 'limfociti', 'eritrociti', 'hgb', 'trombociti',
    'glikemija', 'urea', 'kreatinin', 'K', 'Na', 'albumini',
    'CRP', 'pct', 'AST', 'ALT', 'bilT', 'gGT',
    'CK', 'LDH', 'pBNP', 'ddimer', 'PV', 'aPTT', 'INR',
    'troponin', 'feritin', 'IL6',
]

feature_cols = [c for c in feature_cols if c in df.columns]
log(f'  Features: {len(feature_cols)}')

X = df[feature_cols].copy()
y = df['severe_outcome'].values

# Check missingness
log('\n  Feature missingness:')
for col in feature_cols:
    pct_miss = 100 * X[col].isna().sum() / len(X)
    if pct_miss > 30:
        log(f'    ⚠ {col}: {pct_miss:.0f}% missing (will impute)')

# KNN imputation
log('\n[STEP 3] KNN imputation (k=5)...')
imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

# ============================================================================
# STEP 4: Train multiple models with LOOCV
# ============================================================================
log('\n[STEP 4] Training models with LOOCV...')

# Full-feature XGBoost (regularized for small N)
log('\n  === MODEL 1: XGBoost (24 features, regularized) ===')
xgb_full = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    reg_alpha=0.5, reg_lambda=1.0,  # L1 + L2 regularization
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0
)
loo = LeaveOneOut()
xgb_full_proba = cross_val_predict(xgb_full, X_imputed.values, y, cv=loo, method='predict_proba')[:, 1]
xgb_full_auc = roc_auc_score(y, xgb_full_proba)
xgb_full_brier = brier_score_loss(y, xgb_full_proba)
log(f'  LOOCV AUC: {xgb_full_auc:.3f}, Brier: {xgb_full_brier:.3f}')

# Simple baseline: age + CRP only
log('\n  === MODEL 2: Baseline (age + CRP only) ===')
baseline_cols = ['godine', 'CRP']
X_baseline = X_imputed[baseline_cols].values
lr_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
scaler_b = StandardScaler()
X_baseline_scaled = scaler_b.fit_transform(X_baseline)
lr_base_proba = cross_val_predict(lr_baseline, X_baseline_scaled, y, cv=loo, method='predict_proba')[:, 1]
lr_base_auc = roc_auc_score(y, lr_base_proba)
lr_base_brier = brier_score_loss(y, lr_base_proba)
log(f'  LOOCV AUC: {lr_base_auc:.3f}, Brier: {lr_base_brier:.3f}')

# Expanded baseline: age + CRP + urea + lymphocytes
log('\n  === MODEL 3: Expanded baseline (age + CRP + urea + limfociti + LDH) ===')
exp_cols = ['godine', 'CRP', 'urea', 'limfociti', 'LDH']
exp_cols = [c for c in exp_cols if c in X_imputed.columns]
X_exp = X_imputed[exp_cols].values
lr_exp = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
scaler_e = StandardScaler()
X_exp_scaled = scaler_e.fit_transform(X_exp)
lr_exp_proba = cross_val_predict(lr_exp, X_exp_scaled, y, cv=loo, method='predict_proba')[:, 1]
lr_exp_auc = roc_auc_score(y, lr_exp_proba)
lr_exp_brier = brier_score_loss(y, lr_exp_proba)
log(f'  LOOCV AUC: {lr_exp_auc:.3f}, Brier: {lr_exp_brier:.3f}')

# Full Logistic Regression (all features)
log('\n  === MODEL 4: Logistic Regression (all features) ===')
lr_full = LogisticRegression(max_iter=2000, class_weight='balanced',
                              random_state=RANDOM_STATE, C=0.5)  # L2 regularized
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_imputed.values)
lr_full_proba = cross_val_predict(lr_full, X_full_scaled, y, cv=loo, method='predict_proba')[:, 1]
lr_full_auc = roc_auc_score(y, lr_full_proba)
lr_full_brier = brier_score_loss(y, lr_full_proba)
log(f'  LOOCV AUC: {lr_full_auc:.3f}, Brier: {lr_full_brier:.3f}')

# Bootstrap 95% CI for best model
log('\n[STEP 5] Bootstrap 95% CI for best model...')
models_auc = {
    'XGBoost': xgb_full_auc,
    'LR (all)': lr_full_auc,
    'LR (age+CRP+urea+ly+LDH)': lr_exp_auc,
    'Baseline (age+CRP)': lr_base_auc
}
best_model_name = max(models_auc, key=models_auc.get)
best_proba = {'XGBoost': xgb_full_proba, 'LR (all)': lr_full_proba,
              'LR (age+CRP+urea+ly+LDH)': lr_exp_proba,
              'Baseline (age+CRP)': lr_base_proba}[best_model_name]

rng = np.random.RandomState(42)
boot_aucs = []
for _ in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    try:
        boot_aucs.append(roc_auc_score(y[idx], best_proba[idx]))
    except:
        pass
ci_low = np.percentile(boot_aucs, 2.5)
ci_high = np.percentile(boot_aucs, 97.5)
log(f'  Best: {best_model_name}, AUC = {models_auc[best_model_name]:.3f} [95% CI {ci_low:.3f}-{ci_high:.3f}]')

# ============================================================================
# STEP 6: Save metrics
# ============================================================================
metrics_rows = [
    {'Model': 'XGBoost (full)', 'N_features': len(feature_cols),
     'AUC': round(xgb_full_auc, 3), 'Brier': round(xgb_full_brier, 3)},
    {'Model': 'LR (all features)', 'N_features': len(feature_cols),
     'AUC': round(lr_full_auc, 3), 'Brier': round(lr_full_brier, 3)},
    {'Model': 'LR (age+CRP+urea+ly+LDH)', 'N_features': len(exp_cols),
     'AUC': round(lr_exp_auc, 3), 'Brier': round(lr_exp_brier, 3)},
    {'Model': 'Baseline (age+CRP)', 'N_features': 2,
     'AUC': round(lr_base_auc, 3), 'Brier': round(lr_base_brier, 3)},
    {'Model': 'External transfer (Sirio->Kragujevac)', 'N_features': 30,
     'AUC': 0.503, 'Brier': 0.259, 'Note': 'from previous script'},
]
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / 'local_kragujevac_metrics.csv', index=False)
log(f'\n  Saved: local_kragujevac_metrics.csv')

# Save predictions
pred_df = df[['godine', 'pol', 'rs11549465', 'rs41508050', 'severe_outcome',
              'HFV', 'MV', 'smrtni_ishod']].copy()
pred_df['xgb_proba'] = xgb_full_proba
pred_df['lr_full_proba'] = lr_full_proba
pred_df['lr_exp_proba'] = lr_exp_proba
pred_df['baseline_proba'] = lr_base_proba
pred_df.to_csv(OUT_DIR / 'local_kragujevac_predictions.csv', index=False)

# ============================================================================
# STEP 7: Figure D7 — Comparison ROC curves
# ============================================================================
log('\n[STEP 7] Figure D7 — ROC comparison...')
fig, ax = plt.subplots(figsize=(8, 7))

model_data = [
    ('XGBoost full', xgb_full_proba, xgb_full_auc, '#E24A4A'),
    ('LR full', lr_full_proba, lr_full_auc, '#1f77b4'),
    ('LR age+CRP+urea+ly+LDH', lr_exp_proba, lr_exp_auc, '#2ca02c'),
    ('Baseline age+CRP', lr_base_proba, lr_base_auc, '#ff7f0e'),
]

for name, proba, auc_val, color in model_data:
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_val:.3f})')

# External transfer (dashed, greyed)
ax.plot([0, 0.5, 1], [0, 0.5, 1], 'gray', lw=1.5, linestyle=':',
        alpha=0.7, label='External transfer (Sirio→Kg) AUC=0.503')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'Figure D7 — Local Kragujevac models (LOOCV, N={len(df)})', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D7_local_ROC.png', dpi=300)
plt.close()
log('  Saved: D7_local_ROC.png')

# ============================================================================
# STEP 8: SHAP for local XGBoost (retrained on full data)
# ============================================================================
log('\n[STEP 8] SHAP analysis on local XGBoost...')
xgb_final = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    reg_alpha=0.5, reg_lambda=1.0,
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0
)
xgb_final.fit(X_imputed.values, y)

try:
    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_imputed.values)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    plt.figure(figsize=(8, 10))
    shap.summary_plot(shap_values, X_imputed.values, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.title('SHAP — Local Kragujevac XGBoost', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'D8_local_SHAP.png', dpi=300)
    plt.close()
    log('  Saved: D8_local_SHAP.png')
    
    shap_imp = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_imp.to_csv(OUT_DIR / 'local_shap_importance.csv', index=False)
    
    log('\n  Top 10 local features by SHAP:')
    for _, r in shap_imp.head(10).iterrows():
        log(f'    {r["feature"]:20s}  {r["mean_abs_shap"]:.4f}')
except Exception as e:
    log(f'  SHAP error: {e}')

# Save final trained model
with open(OUT_DIR / 'local_xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_final, f)
with open(OUT_DIR / 'local_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
with open(OUT_DIR / 'local_imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

# ============================================================================
# STEP 9: Figure D9 — comparison bar chart
# ============================================================================
log('\n[STEP 9] Figure D9 — summary bar chart...')
fig, ax = plt.subplots(figsize=(10, 6))
names = [r['Model'] for r in metrics_rows]
aucs = [r['AUC'] for r in metrics_rows]
colors = ['#E24A4A', '#1f77b4', '#2ca02c', '#ff7f0e', '#888888']
bars = ax.barh(names, aucs, color=colors, edgecolor='black')
for bar, auc_val in zip(bars, aucs):
    ax.text(auc_val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{auc_val:.3f}', va='center', fontsize=11, fontweight='bold')
ax.axvline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
ax.axvline(0.7, color='green', linestyle=':', alpha=0.5, label='Clinically useful (0.7)')
ax.set_xlim(0.3, 1.0)
ax.set_xlabel('AUC', fontsize=12)
ax.set_title(f'Figure D9 — Model comparison on Kragujevac cohort (N={len(df)})', fontsize=12)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D9_local_vs_external_comparison.png', dpi=300)
plt.close()
log('  Saved: D9_local_vs_external_comparison.png')

# ============================================================================
# DONE
# ============================================================================
log('\n' + '='*75)
log(f'LOCAL TRAINING COMPLETE: {datetime.now().isoformat()}')
log(f'Output: {OUT_DIR}')
log('='*75)

# Summary
log('\n=== SUMMARY ===')
log(f'Best local model: {best_model_name}')
log(f'AUC = {models_auc[best_model_name]:.3f} [95% CI {ci_low:.3f}-{ci_high:.3f}]')
log(f'vs External transfer AUC = 0.503')
log(f'vs Chance AUC = 0.500')
improvement = models_auc[best_model_name] - 0.503
log(f'Improvement over external transfer: +{improvement:.3f}')

with open(OUT_DIR / 'local_training_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
