"""
===============================================================================
Scale C — Polish deliverables
===============================================================================
Four additions to complete Scale C for manuscript:
    1. Bootstrap 95% CI on AUC for all models (was: point estimate only)
    2. Baseline demographic-only model (age + comorbidities, no labs)
       to quantify incremental value of laboratory biomarkers
    3. Decision Curve Analysis (net benefit across thresholds)
    4. Cleaned calibration figure (2-curve focused plot, no clutter)

Uses saved models from previous Scale C training run.
===============================================================================
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              average_precision_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# ============================================================================
# CONFIG
# ============================================================================
DATA_CSV = Path('/home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/sirio_processed/sirio_early_ml_ready.csv')
MODELS_DIR = Path('/home/marko-b2/COVID_AI_Project/02_ML_Model/sirio/models')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/02_ML_Model/sirio/polish')
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'figures').mkdir(exist_ok=True)

RANDOM_STATE = 42

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE C — POLISH DELIVERABLES')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load data and saved models
# ============================================================================
log('\n[STEP 1] Loading preprocessed data and saved models...')

df = pd.read_csv(DATA_CSV)
log(f'  Data shape: {df.shape}')
log(f'  Columns (first 10): {list(df.columns[:10])}')

# Load feature names from saved pkl
with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
log(f'  N features: {len(feature_names)}')
log(f'  Features: {feature_names}')

# Identify outcome column (likely 'ICU' or 'ICU_later' or similar)
outcome_candidates = [c for c in df.columns if 'ICU' in c.upper() or 'OUTCOME' in c.upper() or 'TARGET' in c.upper() or c.lower() == 'y']
log(f'  Potential outcome columns: {outcome_candidates}')

if len(outcome_candidates) == 0:
    log('  ERROR: cannot identify outcome column. Columns are:')
    for c in df.columns:
        log(f'    {c}')
    raise ValueError('Outcome column not found')

# Use the first candidate (manual adjust if wrong)
outcome_col = outcome_candidates[0]
if 'ICU_LATER' in [c.upper() for c in df.columns]:
    outcome_col = [c for c in df.columns if c.upper() == 'ICU_LATER'][0]
log(f'  Using outcome column: {outcome_col}')

# Build X, y
X = df[feature_names].values
y = df[outcome_col].astype(int).values
log(f'  N patients: {len(y)}, event rate: {y.mean()*100:.1f}%')

# Load models
with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / 'XGBoost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open(MODELS_DIR / 'calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)
with open(MODELS_DIR / 'LogisticRegression.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open(MODELS_DIR / 'RandomForest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open(MODELS_DIR / 'LightGBM.pkl', 'rb') as f:
    lgb_model = pickle.load(f)

log(f'  All 5 models loaded successfully')

# Recreate train/test split (same random_state as original)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_tr_sc = scaler.transform(X_tr)
X_te_sc = scaler.transform(X_te)
log(f'  Train: {len(X_tr)}, Test: {len(X_te)}')

# Get predictions from all models
probas = {
    'Logistic Regression': lr_model.predict_proba(X_te_sc)[:, 1],
    'Random Forest': rf_model.predict_proba(X_te)[:, 1],
    'XGBoost': xgb_model.predict_proba(X_te)[:, 1],
    'LightGBM': lgb_model.predict_proba(X_te)[:, 1],
    'XGBoost (calibrated)': calibrator.predict_proba(X_te)[:, 1],
}

# ============================================================================
# STEP 2: Bootstrap 95% CI for all models
# ============================================================================
log('\n[STEP 2] Bootstrap 95% CI for AUC (1000 iterations)...')
rng = np.random.RandomState(42)

def boot_ci(y_true, y_proba, n_iter=1000):
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
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

ci_rows = []
for model_name, proba in probas.items():
    auc = roc_auc_score(y_te, proba)
    mean_auc, ci_l, ci_h = boot_ci(y_te, proba)
    brier = brier_score_loss(y_te, proba)
    ap = average_precision_score(y_te, proba)
    log(f'  {model_name:28s}: AUC = {auc:.3f} [95% CI {ci_l:.3f}-{ci_h:.3f}], Brier = {brier:.3f}, AP = {ap:.3f}')
    ci_rows.append({
        'Model': model_name,
        'Test_AUC': round(auc, 3),
        'CI_low': round(ci_l, 3),
        'CI_high': round(ci_h, 3),
        'Brier': round(brier, 3),
        'AvgPrecision': round(ap, 3)
    })

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT_DIR / 'scale_c_bootstrap_CI.csv', index=False)
log(f'\n  Saved: scale_c_bootstrap_CI.csv')

# ============================================================================
# STEP 3: Baseline model — demographics + comorbidities only (no labs)
# ============================================================================
log('\n[STEP 3] Training baseline model (demographics + comorbidities only)...')

# Identify demographic/comorbidity features vs. lab features
# In Sirio dataset, features named AGE_PERCENTIL, GENDER, DISEASE GROUPING 1-6 are demog/comorb
# Everything else is likely a lab value or vital

# Conservative heuristic: match features by name pattern
demog_patterns = ['age', 'percentil', 'gender', 'sex', 'disease', 'group']
demog_features = []
for f in feature_names:
    if any(pat in f.lower() for pat in demog_patterns):
        demog_features.append(f)

log(f'  Identified {len(demog_features)} demographic/comorbidity features: {demog_features}')

if len(demog_features) < 2:
    log(f'\n  WARNING: only {len(demog_features)} demographic features found.')
    log(f'  Using first 4 columns as baseline (demographic proxy).')
    demog_features = feature_names[:4]

demog_idx = [feature_names.index(f) for f in demog_features]
X_tr_demog = X_tr[:, demog_idx]
X_te_demog = X_te[:, demog_idx]

# Scale and train LR on demographic features only
sc_demog = StandardScaler()
X_tr_demog_sc = sc_demog.fit_transform(X_tr_demog)
X_te_demog_sc = sc_demog.transform(X_te_demog)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lr_base = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=RANDOM_STATE)
cv_base = cross_val_score(lr_base, X_tr_demog_sc, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
lr_base.fit(X_tr_demog_sc, y_tr)
base_proba = lr_base.predict_proba(X_te_demog_sc)[:, 1]
base_auc = roc_auc_score(y_te, base_proba)
_, base_ci_l, base_ci_h = boot_ci(y_te, base_proba)

log(f'\n  Baseline LR (demographics only, {len(demog_features)} features):')
log(f'    CV AUC: {cv_base.mean():.3f} ± {cv_base.std():.3f}')
log(f'    Test AUC: {base_auc:.3f} [95% CI {base_ci_l:.3f}-{base_ci_h:.3f}]')

# Incremental value calculation
best_full = max(ci_rows, key=lambda x: x['Test_AUC'])
delta_auc = best_full['Test_AUC'] - base_auc
log(f'\n  Best full model ({best_full["Model"]}): AUC = {best_full["Test_AUC"]}')
log(f'  Δ AUC (labs add above demographics): +{delta_auc:.3f}')

# Save baseline result
pd.DataFrame([{
    'Model': 'LR baseline (demog only)',
    'N_features': len(demog_features),
    'Test_AUC': round(base_auc, 3),
    'CI_low': round(base_ci_l, 3),
    'CI_high': round(base_ci_h, 3),
    'vs_best_full_AUC': round(best_full['Test_AUC'], 3),
    'Delta_AUC_labs_contribute': round(delta_auc, 3)
}]).to_csv(OUT_DIR / 'scale_c_baseline_comparison.csv', index=False)

probas['Baseline (demog only)'] = base_proba  # add to proba dict for plotting

# ============================================================================
# STEP 4: Decision Curve Analysis
# ============================================================================
log('\n[STEP 4] Decision Curve Analysis...')

def decision_curve(y_true, y_proba, thresholds=None):
    """
    Compute net benefit at each threshold.
    Net benefit = (TP/N) - (FP/N) * (threshold/(1-threshold))
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    N = len(y_true)
    nbs = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        if t >= 1:
            nb = 0
        else:
            nb = (tp / N) - (fp / N) * (t / (1 - t))
        nbs.append(nb)
    return np.array(thresholds), np.array(nbs)

# Calculate for each model + "treat all" + "treat none"
thresholds = np.arange(0.01, 0.99, 0.01)
dca_models = ['XGBoost (calibrated)', 'XGBoost', 'Logistic Regression', 'Baseline (demog only)']

# Treat all: NB = prevalence - (1-prev) * (t/(1-t))
prev = y_te.mean()
treat_all_nb = prev - (1 - prev) * (thresholds / (1 - thresholds))
treat_none_nb = np.zeros_like(thresholds)

fig, ax = plt.subplots(figsize=(10, 7))
colors = {'XGBoost (calibrated)': '#E24A4A',
          'XGBoost': '#FF9500',
          'Logistic Regression': '#4A90E2',
          'Baseline (demog only)': '#8E8E93'}

for m in dca_models:
    if m in probas:
        t, nb = decision_curve(y_te, probas[m])
        ax.plot(t, nb, lw=2, color=colors.get(m, 'gray'), label=m)

ax.plot(thresholds, treat_all_nb, 'k--', lw=1.5, alpha=0.7, label='Treat all')
ax.plot(thresholds, treat_none_nb, 'k:', lw=1.5, alpha=0.7, label='Treat none (NB=0)')

ax.set_xlabel('Threshold probability', fontsize=12)
ax.set_ylabel('Net benefit', fontsize=12)
ax.set_title('Figure C5 — Decision Curve Analysis (Scale C, hold-out test set)', fontsize=12)
ax.set_xlim(0, 0.8)
ax.set_ylim(-0.05, prev + 0.02)
ax.axhline(0, color='k', lw=0.5, alpha=0.4)
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C5_decision_curve.png', dpi=300)
plt.close()
log('  Saved: C5_decision_curve.png')

# Interpretation: find threshold range where best model dominates
best_proba = probas['XGBoost (calibrated)']
best_t, best_nb = decision_curve(y_te, best_proba)
_, base_nb = decision_curve(y_te, probas['Baseline (demog only)'])
dominant_range = (best_t[best_nb > base_nb]).tolist()
if len(dominant_range) > 0:
    log(f'\n  Best (calibrated XGBoost) dominates baseline across thresholds {dominant_range[0]:.2f}–{dominant_range[-1]:.2f}')
else:
    log(f'\n  No clear dominance range identified')

# Save DCA data for possible re-use
dca_data = pd.DataFrame({'threshold': thresholds})
for m in dca_models:
    if m in probas:
        _, nb = decision_curve(y_te, probas[m])
        dca_data[m.replace(' ', '_')] = nb
dca_data['treat_all'] = treat_all_nb
dca_data['treat_none'] = treat_none_nb
dca_data.to_csv(OUT_DIR / 'scale_c_dca_data.csv', index=False)

# ============================================================================
# STEP 5: Cleaned calibration figure (2-curve focused)
# ============================================================================
log('\n[STEP 5] Cleaned calibration figure...')

# Get predictions for uncalibrated XGBoost and calibrated XGBoost
xgb_uncal = xgb_model.predict_proba(X_te)[:, 1]
xgb_cal = calibrator.predict_proba(X_te)[:, 1]

# Calibration curves
n_bins = 8
frac_uncal, mean_pred_uncal = calibration_curve(y_te, xgb_uncal, n_bins=n_bins, strategy='quantile')
frac_cal, mean_pred_cal = calibration_curve(y_te, xgb_cal, n_bins=n_bins, strategy='quantile')

brier_uncal = brier_score_loss(y_te, xgb_uncal)
brier_cal = brier_score_loss(y_te, xgb_cal)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                gridspec_kw={'width_ratios': [2, 1]})

# Left: calibration curves
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Perfect calibration')
ax1.plot(mean_pred_uncal, frac_uncal, 'o-', color='#FF9500', lw=2.5, markersize=10,
         label=f'XGBoost uncalibrated (Brier={brier_uncal:.3f})')
ax1.plot(mean_pred_cal, frac_cal, 's-', color='#E24A4A', lw=2.5, markersize=10,
         label=f'XGBoost Platt-calibrated (Brier={brier_cal:.3f})')
ax1.set_xlabel('Mean predicted probability', fontsize=12)
ax1.set_ylabel('Observed outcome frequency', fontsize=12)
ax1.set_title('Reliability plot (quantile bins)', fontsize=12)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)

# Right: histogram of predicted probabilities
ax2.hist(xgb_uncal, bins=20, alpha=0.5, color='#FF9500', label='Uncalibrated', edgecolor='black', linewidth=0.5)
ax2.hist(xgb_cal, bins=20, alpha=0.5, color='#E24A4A', label='Calibrated', edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Predicted probability', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Distribution of predictions', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(alpha=0.3)

plt.suptitle('Figure C4 (revised) — Calibration of XGBoost (hold-out test set)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C4_calibration_clean.png', dpi=300)
plt.close()
log(f'  Saved: C4_calibration_clean.png')
log(f'  Brier score: uncalibrated {brier_uncal:.3f} → calibrated {brier_cal:.3f}')

# ============================================================================
# STEP 6: Combined summary ROC figure with baseline overlay
# ============================================================================
log('\n[STEP 6] Combined ROC: XGBoost (best) vs baseline...')

fig, ax = plt.subplots(figsize=(8, 7))
for model_name, color in [('XGBoost (calibrated)', '#E24A4A'),
                           ('XGBoost', '#FF9500'),
                           ('Baseline (demog only)', '#8E8E93')]:
    proba = probas[model_name]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    ci_row = [r for r in ci_rows + [{'Model': 'Baseline (demog only)',
                                      'Test_AUC': base_auc,
                                      'CI_low': base_ci_l,
                                      'CI_high': base_ci_h}]
              if r['Model'] == model_name]
    if ci_row:
        r = ci_row[0]
        label = f'{model_name} (AUC={auc:.3f} [{r["CI_low"]:.3f}-{r["CI_high"]:.3f}])'
    else:
        label = f'{model_name} (AUC={auc:.3f})'
    ax.plot(fpr, tpr, color=color, lw=2.5, label=label)

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'Figure C6 — Incremental value of lab biomarkers (N={len(y_te)})',
             fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

# Annotate delta AUC
ax.text(0.55, 0.15,
         f'Δ AUC (labs vs demog): +{delta_auc:.3f}',
         fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FFF9E6', edgecolor='#FFD666'))

plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C6_incremental_value.png', dpi=300)
plt.close()
log('  Saved: C6_incremental_value.png')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
log('\n' + '='*75)
log('SCALE C POLISH COMPLETE')
log(f'Finished: {datetime.now().isoformat()}')
log('='*75)

log('\n=== FINAL METRICS SUMMARY ===')
log(ci_df.to_string(index=False))

log(f'\n=== INCREMENTAL VALUE ===')
log(f'  Baseline (demographics only): AUC = {base_auc:.3f} [{base_ci_l:.3f}-{base_ci_h:.3f}]')
log(f'  Best full model (with labs): AUC = {best_full["Test_AUC"]}')
log(f'  Δ AUC from laboratory biomarkers: +{delta_auc:.3f}')

log(f'\n=== CALIBRATION ===')
log(f'  Uncalibrated Brier: {brier_uncal:.3f}')
log(f'  Platt-calibrated Brier: {brier_cal:.3f}')
log(f'  Improvement: {brier_uncal - brier_cal:+.3f}')

log(f'\n=== FILES GENERATED ===')
log(f'  /polish/scale_c_bootstrap_CI.csv')
log(f'  /polish/scale_c_baseline_comparison.csv')
log(f'  /polish/scale_c_dca_data.csv')
log(f'  /polish/figures/C4_calibration_clean.png')
log(f'  /polish/figures/C5_decision_curve.png')
log(f'  /polish/figures/C6_incremental_value.png')

with open(OUT_DIR / 'scale_c_polish_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
