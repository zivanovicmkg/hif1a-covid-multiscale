"""
===============================================================================
Scale D — FINAL model and deliverables
===============================================================================
Final model (Opcija B): 11 baseline features, komorbiditeti_bin removed
(collinear with individual comorbidities).

Rationale for final choice:
    - VIF analysis: all features < 5 after removing komorbiditeti_bin
    - HTA retained despite negative coefficient: real effect in this cohort
      (likely ACE inhibitor/ARB protective effect, literature-supported)
    - Highest AUC among tested variants (0.840)
    - Clean clinical interpretation

Outputs (in /home/marko-b2/COVID_AI_Project/03_External_Validation/scale_d_final/):
    - scale_d_final_model.pkl + scaler.pkl
    - Table_D_final_coefficients.csv
    - Table_D_final_metrics.csv
    - figures/D10_final_ROC.png (updated)
    - figures/D11_final_coefficients.png (updated)
    - figures/D13_final_forest_plot.png (NEW)
    - scale_d_final_log.txt
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
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, precision_recall_curve,
                             confusion_matrix)
from scipy import stats

# ============================================================================
KRAGUJEVAC_FILE = Path('/home/marko-b2/genetika_COVID19_v2.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/03_External_Validation/scale_d_final')
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
log('SCALE D — FINAL MODEL AND DELIVERABLES')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# Load data
# ============================================================================
log('\n[STEP 1] Loading data...')
df = pd.read_excel(KRAGUJEVAC_FILE)
df = df.dropna(subset=['godine']).reset_index(drop=True)
df['severe_outcome'] = ((df['HFV'] == 1) | (df['MV'] == 1) | (df['smrtni_ishod'] == 1)).astype(int)

log(f'  N = {len(df)} patients')
log(f'  Severe outcomes: {int(df["severe_outcome"].sum())} ({df["severe_outcome"].mean()*100:.1f}%)')

# ============================================================================
# FINAL FEATURE SET — Opcija B
# ============================================================================
log('\n[STEP 2] Final feature set (11 features, no komorbiditeti_bin)...')
final_features = [
    'godine',         # Age
    'pol',            # Sex (1=M, 0=F)
    'SAT_O2',         # SpO2 on admission
    'pO2',            # pO2 on admission
    'DM',             # Diabetes
    'HTA',            # Hypertension
    'HOBP',           # COPD/Asthma
    'neuroloska_dg',  # Neurological disease
    'maligna_Dg',     # Malignancy
    'HRI',            # Chronic renal insufficiency
    'vakcinacija',    # COVID-19 vaccination
]
log(f'  Features: {final_features}')

X = df[final_features].copy()
y = df['severe_outcome'].values

# Median imputation for missing
for col in X.columns:
    if X[col].isna().any():
        med = X[col].median()
        X[col] = X[col].fillna(med)
        log(f'  Imputed {col} with median = {med:.2f}')

# ============================================================================
# LOOCV predictions and metrics
# ============================================================================
log('\n[STEP 3] LOOCV predictions...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

lr_final = LogisticRegression(max_iter=2000, class_weight='balanced',
                               C=1.0, random_state=42)
y_proba = cross_val_predict(lr_final, X_scaled, y,
                             cv=LeaveOneOut(), method='predict_proba')[:, 1]

# Metrics
auc = roc_auc_score(y, y_proba)
ap = average_precision_score(y, y_proba)
brier = brier_score_loss(y, y_proba)

# Optimal threshold by Youden J
fpr, tpr, thresholds = roc_curve(y, y_proba)
j_scores = tpr - fpr
opt_idx = np.argmax(j_scores)
opt_threshold = thresholds[opt_idx]
y_pred = (y_proba >= opt_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

log(f'  AUC = {auc:.3f}')
log(f'  Average Precision = {ap:.3f}')
log(f'  Brier = {brier:.3f}')
log(f'  Optimal threshold (Youden): {opt_threshold:.3f}')
log(f'  Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
log(f'  PPV: {ppv:.3f}, NPV: {npv:.3f}')
log(f'  Accuracy: {accuracy:.3f}')

# Bootstrap CI
log('\n[STEP 4] Bootstrap 95% CI (1000 iterations)...')
rng = np.random.RandomState(42)
boot_aucs = []
for _ in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    try:
        boot_aucs.append(roc_auc_score(y[idx], y_proba[idx]))
    except:
        pass
ci_low = np.percentile(boot_aucs, 2.5)
ci_high = np.percentile(boot_aucs, 97.5)
log(f'  AUC = {auc:.3f} [95% CI {ci_low:.3f}-{ci_high:.3f}]')

# ============================================================================
# Fit final model on all data for coefficients
# ============================================================================
log('\n[STEP 5] Fitting final model on all data for coefficients + CIs...')
lr_final.fit(X_scaled, y)
coefs = lr_final.coef_[0]
intercept = lr_final.intercept_[0]

# Standard errors via Fisher info
p_hat = lr_final.predict_proba(X_scaled)[:, 1]
X_int = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
W = np.diag(p_hat * (1 - p_hat))
try:
    cov = np.linalg.inv(X_int.T @ W @ X_int)
    se_all = np.sqrt(np.diag(cov))
    se = se_all[1:]  # skip intercept
except np.linalg.LinAlgError:
    se = np.full_like(coefs, np.nan)

coef_rows = []
for i, feat in enumerate(final_features):
    coef = coefs[i]
    s = se[i]
    or_ = np.exp(coef)
    if np.isnan(s):
        ci_l = ci_h = np.nan
        p_val = np.nan
    else:
        ci_l = np.exp(coef - 1.96 * s)
        ci_h = np.exp(coef + 1.96 * s)
        z = coef / s
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    
    direction = '↑ Risk' if coef > 0 else '↓ Risk' if coef < 0 else '≈'
    coef_rows.append({
        'Feature': feat,
        'Coefficient': round(coef, 3),
        'SE': round(s, 3) if not np.isnan(s) else 'NA',
        'OR': round(or_, 2),
        'CI_low': round(ci_l, 2) if not np.isnan(ci_l) else 'NA',
        'CI_high': round(ci_h, 2) if not np.isnan(ci_h) else 'NA',
        'p_value': round(p_val, 4) if not np.isnan(p_val) else 'NA',
        'Direction': direction,
    })

coef_df = pd.DataFrame(coef_rows)
coef_df_sorted = coef_df.copy()
coef_df_sorted['abs_coef'] = coef_df_sorted['Coefficient'].abs()
coef_df_sorted = coef_df_sorted.sort_values('abs_coef', ascending=False).drop(columns='abs_coef')

log('\n  Final model coefficients:')
log(coef_df_sorted.to_string(index=False))
coef_df_sorted.to_csv(OUT_DIR / 'Table_D_final_coefficients.csv', index=False)
log(f'\n  Saved: Table_D_final_coefficients.csv')

# ============================================================================
# Summary metrics table
# ============================================================================
log('\n[STEP 6] Saving final metrics table...')
metrics_summary = pd.DataFrame([{
    'Cohort': 'Kragujevac',
    'N': len(df),
    'N_events': int(y.sum()),
    'Event_rate': round(y.mean(), 3),
    'N_features': len(final_features),
    'AUC': round(auc, 3),
    'AUC_CI_low': round(ci_low, 3),
    'AUC_CI_high': round(ci_high, 3),
    'AP': round(ap, 3),
    'Brier': round(brier, 3),
    'Optimal_threshold': round(opt_threshold, 3),
    'Sensitivity': round(sensitivity, 3),
    'Specificity': round(specificity, 3),
    'PPV': round(ppv, 3),
    'NPV': round(npv, 3),
    'Accuracy': round(accuracy, 3),
    'Validation': 'LOOCV',
    'Bootstrap_iterations': 1000,
}])
metrics_summary.to_csv(OUT_DIR / 'Table_D_final_metrics.csv', index=False)
log(f'  Saved: Table_D_final_metrics.csv')

# ============================================================================
# Figure D10 FINAL — ROC curve
# ============================================================================
log('\n[STEP 7] Figure D10 final — ROC...')
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#1f77b4', lw=2.5,
        label=f'Final model, 11 features (AUC = {auc:.3f} [{ci_low:.3f}-{ci_high:.3f}])')
ax.scatter(fpr[opt_idx], tpr[opt_idx], s=120, c='red', zorder=5, edgecolor='black',
           label=f'Optimal (Youden): Sens={sensitivity:.2f}, Spec={specificity:.2f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title(f'Figure D10 — Final Scale D model, LOOCV (N={len(df)})', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D10_final_ROC.png', dpi=300)
plt.close()
log('  Saved: D10_final_ROC.png')

# ============================================================================
# Figure D11 FINAL — coefficients
# ============================================================================
log('\n[STEP 8] Figure D11 final — coefficients...')
plot_df = coef_df.copy()
plot_df = plot_df.sort_values('Coefficient', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#E24A4A' if c > 0 else '#4A90E2' for c in plot_df['Coefficient']]
ax.barh(plot_df['Feature'], plot_df['Coefficient'], color=colors,
        edgecolor='black', linewidth=0.6)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Logistic regression coefficient (standardized)', fontsize=11)
ax.set_title(f'Figure D11 — Final model coefficients (N={len(df)}, 11 features)',
             fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Annotate with OR values
for i, (feat, c, or_val) in enumerate(zip(plot_df['Feature'], plot_df['Coefficient'],
                                            plot_df['OR'])):
    x_pos = c + (0.05 if c > 0 else -0.05)
    ha = 'left' if c > 0 else 'right'
    ax.text(x_pos, i, f'OR={or_val:.2f}', va='center', ha=ha, fontsize=9)

from matplotlib.patches import Patch
legend = [Patch(facecolor='#E24A4A', label='↑ Risk'), Patch(facecolor='#4A90E2', label='↓ Risk')]
ax.legend(handles=legend, loc='lower right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D11_final_coefficients.png', dpi=300)
plt.close()
log('  Saved: D11_final_coefficients.png')

# ============================================================================
# Figure D13 — Forest plot with OR and CI
# ============================================================================
log('\n[STEP 9] Figure D13 — Forest plot with OR and 95% CI...')
# Sort by OR
fp_df = coef_df[coef_df['CI_low'] != 'NA'].copy()
fp_df['OR'] = pd.to_numeric(fp_df['OR'])
fp_df['CI_low'] = pd.to_numeric(fp_df['CI_low'])
fp_df['CI_high'] = pd.to_numeric(fp_df['CI_high'])
fp_df = fp_df.sort_values('OR')

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(fp_df))

for i, row in enumerate(fp_df.itertuples()):
    color = '#E24A4A' if row.OR > 1 else '#4A90E2'
    ax.plot([row.CI_low, row.CI_high], [i, i], color='k', lw=1.5)
    ax.plot([row.CI_low, row.CI_low], [i-0.1, i+0.1], color='k', lw=1.5)
    ax.plot([row.CI_high, row.CI_high], [i-0.1, i+0.1], color='k', lw=1.5)
    ax.scatter(row.OR, i, s=100, c=color, edgecolor='black', zorder=3)

ax.axvline(1, color='gray', linestyle='--', lw=1, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(fp_df['Feature'], fontsize=11)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
ax.set_xscale('log')
ax.set_title(f'Figure D13 — Final Scale D model: Odds Ratios (N={len(df)})',
             fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Annotate p-values
for i, row in enumerate(fp_df.itertuples()):
    p_val = row.p_value
    if p_val != 'NA':
        p_text = f'p={p_val:.3f}' if float(p_val) >= 0.001 else 'p<0.001'
        ax.text(max(fp_df['CI_high']) * 1.3, i, p_text,
                va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'D13_final_forest_plot.png', dpi=300)
plt.close()
log('  Saved: D13_final_forest_plot.png')

# ============================================================================
# Save predictions & model
# ============================================================================
log('\n[STEP 10] Saving model + predictions...')
preds_df = df[['godine', 'pol', 'rs11549465', 'rs41508050', 'severe_outcome',
                'HFV', 'MV', 'smrtni_ishod']].copy()
preds_df['predicted_proba'] = y_proba
preds_df['predicted_class'] = y_pred
preds_df.to_csv(OUT_DIR / 'scale_d_final_predictions.csv', index=False)

with open(OUT_DIR / 'scale_d_final_model.pkl', 'wb') as f:
    pickle.dump(lr_final, f)
with open(OUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(OUT_DIR / 'feature_names.pkl', 'wb') as f:
    pickle.dump(final_features, f)

log('  Saved: scale_d_final_model.pkl, scaler.pkl, feature_names.pkl')
log('  Saved: scale_d_final_predictions.csv')

# ============================================================================
# Stratified performance by HIF1A
# ============================================================================
log('\n[STEP 11] Stratified performance by HIF1A genotype...')
df['pred_proba'] = y_proba
strat = []
for snp in ['rs11549465', 'rs41508050']:
    for gv, gl in [(0, 'CC'), (1, 'CT')]:
        sub = df[df[snp] == gv]
        if len(sub) < 5:
            continue
        if sub['severe_outcome'].nunique() < 2:
            log(f'  {snp} {gl} (n={len(sub)}): only 1 outcome class')
            strat.append({'SNP': snp, 'Genotype': gl, 'N': len(sub),
                         'AUC': 'NA', 'event_rate': round(sub['severe_outcome'].mean(), 3)})
            continue
        sub_auc = roc_auc_score(sub['severe_outcome'], sub['pred_proba'])
        log(f'  {snp} {gl} (n={len(sub)}): AUC={sub_auc:.3f}, event rate={sub["severe_outcome"].mean():.3f}')
        strat.append({'SNP': snp, 'Genotype': gl, 'N': len(sub),
                     'AUC': round(sub_auc, 3),
                     'event_rate': round(sub['severe_outcome'].mean(), 3)})

pd.DataFrame(strat).to_csv(OUT_DIR / 'stratified_HIF1A_performance.csv', index=False)

# ============================================================================
# DONE
# ============================================================================
log('\n' + '='*75)
log('SCALE D FINAL — COMPLETE')
log(f'Finished: {datetime.now().isoformat()}')
log(f'Output: {OUT_DIR}')
log('='*75)

log('\n=== FINAL DELIVERABLES ===')
log(f'Primary model: LR, 11 baseline features')
log(f'  AUC = {auc:.3f} [95% CI {ci_low:.3f}-{ci_high:.3f}]')
log(f'  Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}')
log(f'  Event rate = {y.mean()*100:.1f}%')

log('\nTOP PREDICTORS (by |coef|):')
top = coef_df_sorted.head(6)
for _, r in top.iterrows():
    log(f'  {r["Direction"]} {r["Feature"]:20s}  OR={r["OR"]}  [{r["CI_low"]}-{r["CI_high"]}]  p={r["p_value"]}')

with open(OUT_DIR / 'scale_d_final_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
