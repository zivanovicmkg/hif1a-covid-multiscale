"""
===============================================================================
Scale D — External Validation of Scale C XGBoost Model on Kragujevac Cohort
===============================================================================
Applies the Scale C XGBoost (trained on Sirio-Libanes) to the Kragujevac
cohort to assess cross-population generalization.

Variables missing in Kragujevac (lactate, DBP, SBP, HR, temperature, resp_rate)
are filled with population medians — this effectively neutralizes their
contribution and is reported as a limitation.

Outputs:
    /home/marko-b2/COVID_AI_Project/03_External_Validation/
        - kragujevac_predictions.csv
        - external_validation_metrics.csv
        - stratified_performance.csv
        - shap_importance_kragujevac.csv
        - figures/D3_ROC_external.png
        - figures/D4_calibration_external.png
        - figures/D5_predictions_by_genotype.png
        - figures/D6_SHAP_kragujevac.png
        - external_validation_log.txt
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
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, brier_score_loss,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.impute import KNNImputer
from scipy import stats
import shap

# ============================================================================
# CONFIGURATION
# ============================================================================
# ADJUST IF YOUR FILE IS ELSEWHERE
KRAGUJEVAC_FILE = Path('/home/marko-b2/genetika_COVID19_v2.xlsx')

MODELS_DIR = Path('/home/marko-b2/COVID_AI_Project/02_ML_Model/sirio/models')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/03_External_Validation')
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
log('SCALE D — EXTERNAL VALIDATION ON KRAGUJEVAC COHORT')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load saved models
# ============================================================================
log('\n[STEP 1] Loading saved Scale C models...')
with open(MODELS_DIR / 'XGBoost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / 'calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)
with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

log(f'  XGBoost model: {type(xgb_model).__name__}')
log(f'  Features required ({len(feature_names)}):')
for i, feat in enumerate(feature_names):
    log(f'    [{i:2d}] {feat}')

# ============================================================================
# STEP 2: Load and clean Kragujevac data
# ============================================================================
log('\n[STEP 2] Loading Kragujevac cohort...')
df = pd.read_excel(KRAGUJEVAC_FILE)
df = df.dropna(subset=['godine']).reset_index(drop=True)
log(f'  Loaded: {len(df)} patients')

# Fix komorbiditeti (object type)
df['komorbiditeti_bin'] = df['komorbiditeti'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '0' or str(x).strip().lower() == 'ne') else 1
)

# ============================================================================
# STEP 3: Build composite outcome (Sirio's ICU_LATER ≈ HFV OR MV OR death)
# ============================================================================
log('\n[STEP 3] Building composite severe outcome (HFV OR MV OR death)...')
df['severe_outcome'] = ((df['HFV'] == 1) | (df['MV'] == 1) | (df['smrtni_ishod'] == 1)).astype(int)
log(f'  Severe outcome (+): {(df["severe_outcome"]==1).sum()}')
log(f'  Severe outcome (-): {(df["severe_outcome"]==0).sum()}')
log(f'  Event rate: {df["severe_outcome"].mean()*100:.1f}%')

# ============================================================================
# STEP 4: Harmonize variables — map Kragujevac to Sirio feature names
# ============================================================================
log('\n[STEP 4] Harmonizing variables (Kragujevac -> Sirio feature names)...')

feature_map = {
    'age_65plus': lambda df: (df['godine'] >= 65).astype(int),
    'age_percentile': lambda df: df['godine'].rank(pct=True) * 100,
    'sex': lambda df: df['pol'],
    'htn': lambda df: df['HTA'],
    'immunocomp': lambda df: df['maligna_Dg'],
    'leukocytes': lambda df: df['leukociti'],
    'hgb': lambda df: df['hgb'],
    'lymphocytes': lambda df: df['limfociti'],
    'urea': lambda df: df['urea'],
    'creatinine': lambda df: df['kreatinin'],
    'd_dimer': lambda df: df['ddimer'],
    'crp': lambda df: df['CRP'],
    'platelets': lambda df: df['trombociti'],
    'neutrophiles': lambda df: df['leukociti'] - df['limfociti'],
    'potassium': lambda df: df['K'],
    'sodium': lambda df: df['Na'],
    'ast': lambda df: df['AST'],
    'alt': lambda df: df['ALT'],
    'inr': lambda df: df['INR'],
    'aptt': lambda df: df['aPTT'],
    'glucose': lambda df: df['glikemija'],
    'albumin': lambda df: df['albumini'],
    'lactate': lambda df: pd.Series([np.nan] * len(df)),  # NOT AVAILABLE
    'bilirubin': lambda df: df['bilT'],
    'dbp': lambda df: pd.Series([np.nan] * len(df)),  # NOT AVAILABLE
    'sbp': lambda df: pd.Series([np.nan] * len(df)),  # NOT AVAILABLE
    'heart_rate': lambda df: pd.Series([np.nan] * len(df)),  # NOT AVAILABLE
    'spo2': lambda df: df['SAT_O2'],
    'temperature': lambda df: pd.Series([np.nan] * len(df)),  # NOT AVAILABLE
    'resp_rate': lambda df: pd.Series([np.nan] * len(df)),  # NOT AVAILABLE
}

X_kragujevac = pd.DataFrame(index=df.index)
for feat in feature_names:
    try:
        X_kragujevac[feat] = feature_map[feat](df)
    except KeyError:
        log(f'  ⚠ Feature "{feat}" not in feature_map — filling with NaN')
        X_kragujevac[feat] = np.nan

X_kragujevac = X_kragujevac[feature_names]

log(f'\n  Feature availability in Kragujevac cohort:')
for col in X_kragujevac.columns:
    n_present = X_kragujevac[col].notna().sum()
    pct = 100 * n_present / len(X_kragujevac)
    flag = '⚠️' if pct < 50 else '✓'
    log(f'    {flag} {col:20s}: {n_present}/{len(X_kragujevac)} ({pct:.0f}%)')

# ============================================================================
# STEP 5: Hybrid imputation — constant fill for 100% missing, KNN for rest
# ============================================================================
log('\n[STEP 5] Hybrid imputation (constant for fully-missing, KNN for partial)...')

fully_missing = [c for c in X_kragujevac.columns if X_kragujevac[c].isna().all()]
partial_missing = [c for c in X_kragujevac.columns if not X_kragujevac[c].isna().all()]
log(f'  Fully missing columns (filling with Sirio-like medians): {fully_missing}')
log(f'  Partial columns (KNN imputation): {len(partial_missing)}')

sirio_medians = {
    'lactate': 1.6,
    'dbp': 75.0,
    'sbp': 130.0,
    'heart_rate': 85.0,
    'temperature': 36.8,
    'resp_rate': 20.0,
}

X_filled = X_kragujevac.copy()
for col in fully_missing:
    fill_value = sirio_medians.get(col, 0)
    X_filled[col] = fill_value
    log(f'    Filled {col} = {fill_value} (constant)')

imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed_arr = imputer.fit_transform(X_filled.values)
X_imputed = pd.DataFrame(X_imputed_arr, columns=feature_names, index=X_kragujevac.index)
log(f'  Imputed shape: {X_imputed.shape}')

# ============================================================================
# STEP 6: Apply XGBoost model
# ============================================================================
log('\n[STEP 6] Applying XGBoost model to Kragujevac cohort...')
y_pred_proba = xgb_model.predict_proba(X_imputed.values)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)
y_cal_proba = calibrator.predict_proba(X_imputed.values)[:, 1]

# ============================================================================
# STEP 7: Compute performance metrics
# ============================================================================
log('\n[STEP 7] Computing performance metrics...')
y_true = df['severe_outcome'].values

auc = roc_auc_score(y_true, y_pred_proba)
ap = average_precision_score(y_true, y_pred_proba)
brier = brier_score_loss(y_true, y_pred_proba)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cal_auc = roc_auc_score(y_true, y_cal_proba)
cal_brier = brier_score_loss(y_true, y_cal_proba)

log(f'\n  === RAW XGBoost on Kragujevac ===')
log(f'  AUC:       {auc:.3f}')
log(f'  AP:        {ap:.3f}')
log(f'  Brier:     {brier:.3f}')
log(f'  Accuracy:  {acc:.3f}')
log(f'  Precision: {prec:.3f}')
log(f'  Recall:    {rec:.3f}')
log(f'  F1:        {f1:.3f}')

log(f'\n  === CALIBRATED XGBoost (Sirio-trained Platt) ===')
log(f'  AUC:   {cal_auc:.3f}')
log(f'  Brier: {cal_brier:.3f}')

# Bootstrap 95% CI for AUC
log('\n[STEP 8] Bootstrap 95% CI for AUC (1000 iterations)...')
rng = np.random.RandomState(42)
bootstrap_aucs = []
for i in range(1000):
    idx = rng.choice(len(y_true), len(y_true), replace=True)
    if len(np.unique(y_true[idx])) < 2:
        continue
    try:
        bootstrap_aucs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))
    except:
        pass
ci_low = np.percentile(bootstrap_aucs, 2.5)
ci_high = np.percentile(bootstrap_aucs, 97.5)
log(f'  AUC = {auc:.3f} [95% CI {ci_low:.3f}-{ci_high:.3f}]')

# ============================================================================
# STEP 9: Stratified performance by HIF1A genotype
# ============================================================================
log('\n[STEP 9] Stratified performance by HIF1A genotype...')
df['predicted_proba'] = y_pred_proba
df['predicted_class'] = y_pred

stratified_results = []
for snp in ['rs11549465', 'rs41508050']:
    log(f'\n  --- {snp} ---')
    for genotype_val, genotype_label in [(0, 'CC'), (1, 'CT')]:
        sub = df[df[snp] == genotype_val]
        if len(sub) < 5:
            continue
        if sub['severe_outcome'].nunique() < 2:
            log(f'    {genotype_label}: only 1 outcome class, cannot compute AUC')
            stratified_results.append({
                'SNP': snp, 'genotype': genotype_label, 'N': len(sub),
                'AUC': 'NA (single outcome class)',
                'mean_predicted_prob': round(sub["predicted_proba"].mean(), 3),
                'actual_event_rate': round(sub["severe_outcome"].mean(), 3)
            })
            continue
        sub_auc = roc_auc_score(sub['severe_outcome'], sub['predicted_proba'])
        log(f'    {genotype_label} (n={len(sub)}): AUC={sub_auc:.3f}, '
            f'mean pred={sub["predicted_proba"].mean():.3f}, '
            f'actual event rate={sub["severe_outcome"].mean():.3f}')
        stratified_results.append({
            'SNP': snp, 'genotype': genotype_label, 'N': len(sub),
            'AUC': round(sub_auc, 3),
            'mean_predicted_prob': round(sub['predicted_proba'].mean(), 3),
            'actual_event_rate': round(sub['severe_outcome'].mean(), 3)
        })

pd.DataFrame(stratified_results).to_csv(OUT_DIR / 'stratified_performance.csv', index=False)

# ============================================================================
# STEP 10: Save metrics and predictions
# ============================================================================
metrics = {
    'Cohort': 'Kragujevac',
    'N': len(df),
    'Event_rate': round(y_true.mean(), 3),
    'AUC': round(auc, 3),
    'AUC_CI_low': round(ci_low, 3),
    'AUC_CI_high': round(ci_high, 3),
    'AUC_calibrated': round(cal_auc, 3),
    'Brier': round(brier, 3),
    'Brier_calibrated': round(cal_brier, 3),
    'AP': round(ap, 3),
    'Accuracy': round(acc, 3),
    'Precision': round(prec, 3),
    'Recall': round(rec, 3),
    'F1': round(f1, 3),
}
pd.DataFrame([metrics]).to_csv(OUT_DIR / 'external_validation_metrics.csv', index=False)
log(f'\n  Saved: external_validation_metrics.csv')

preds = df[['godine', 'pol', 'rs11549465', 'rs41508050', 'severe_outcome',
            'HFV', 'MV', 'smrtni_ishod']].copy()
preds['predicted_proba'] = y_pred_proba
preds['predicted_proba_calibrated'] = y_cal_proba
preds['predicted_class'] = y_pred
preds.to_csv(OUT_DIR / 'kragujevac_predictions.csv', index=False)
log(f'  Saved: kragujevac_predictions.csv')

# ============================================================================
# STEP 11: Figure D3 — ROC curve external
# ============================================================================
log('\n[STEP 11] Figure D3 — External validation ROC...')
fig, ax = plt.subplots(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
ax.plot(fpr, tpr, color='#E24A4A', lw=2.5,
        label=f'Kragujevac (AUC = {auc:.3f} [{ci_low:.3f}-{ci_high:.3f}])')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'Figure D3 — External Validation on Kragujevac (N={len(df)})', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D3_ROC_external.png', dpi=300)
plt.close()
log('  Saved: D3_ROC_external.png')

# ============================================================================
# STEP 12: Figure D4 — Calibration
# ============================================================================
log('\n[STEP 12] Figure D4 — Calibration...')
fig, ax = plt.subplots(figsize=(7, 6))
try:
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_pred_proba, n_bins=5, strategy='quantile')
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_cal_proba, n_bins=5, strategy='quantile')
    ax.plot(prob_pred_raw, prob_true_raw, 'o-', color='#E24A4A', lw=2,
            label=f'Raw XGBoost (Brier={brier:.3f})', markersize=10)
    ax.plot(prob_pred_cal, prob_true_cal, 's-', color='purple', lw=2,
            label=f'Calibrated (Brier={cal_brier:.3f})', markersize=10)
except Exception as e:
    log(f'  Calibration curve error: {e} (skipping)')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean predicted probability', fontsize=12)
ax.set_ylabel('Fraction of positives', fontsize=12)
ax.set_title(f'Figure D4 — Calibration on Kragujevac (N={len(df)})', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D4_calibration_external.png', dpi=300)
plt.close()
log('  Saved: D4_calibration_external.png')

# ============================================================================
# STEP 13: Figure D5 — Predicted probabilities by genotype
# ============================================================================
log('\n[STEP 13] Figure D5 — Predictions by genotype...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, snp in zip(axes, ['rs11549465', 'rs41508050']):
    sub = df.dropna(subset=[snp, 'severe_outcome']).copy()
    sub['genotype'] = sub[snp].map({0: 'CC', 1: 'CT'})
    sub['outcome_label'] = sub['severe_outcome'].map({0: 'Non-severe', 1: 'Severe'})
    try:
        sns.violinplot(data=sub, x='genotype', y='predicted_proba', hue='outcome_label',
                       split=True, ax=ax,
                       palette={'Non-severe': '#4A90E2', 'Severe': '#E24A4A'})
    except Exception as e:
        sns.stripplot(data=sub, x='genotype', y='predicted_proba', hue='outcome_label',
                       ax=ax, palette={'Non-severe': '#4A90E2', 'Severe': '#E24A4A'})
    ax.set_title(f'{snp}', fontsize=12)
    ax.set_ylabel('Predicted probability of severe outcome')
    ax.set_xlabel('Genotype')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Figure D5 — Predicted severity by HIF1A genotype and actual outcome', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D5_predictions_by_genotype.png', dpi=300)
plt.close()
log('  Saved: D5_predictions_by_genotype.png')

# ============================================================================
# STEP 14: SHAP on Kragujevac
# ============================================================================
log('\n[STEP 14] SHAP analysis on Kragujevac cohort...')
try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_imputed.values)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_imputed.values, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title('SHAP — Kragujevac cohort (external validation)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'D6_SHAP_kragujevac.png', dpi=300)
    plt.close()
    log('  Saved: D6_SHAP_kragujevac.png')
    
    shap_imp = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap_kragujevac': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap_kragujevac', ascending=False)
    shap_imp.to_csv(OUT_DIR / 'shap_importance_kragujevac.csv', index=False)
    
    log('\n  Top 10 features by SHAP on Kragujevac:')
    for _, r in shap_imp.head(10).iterrows():
        log(f'    {r["feature"]:20s}  {r["mean_abs_shap_kragujevac"]:.4f}')
except Exception as e:
    log(f'  SHAP error: {e}')

# ============================================================================
# DONE
# ============================================================================
log('\n' + '='*75)
log(f'EXTERNAL VALIDATION COMPLETE: {datetime.now().isoformat()}')
log(f'Output: {OUT_DIR}')
log('='*75)

with open(OUT_DIR / 'external_validation_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
