"""
===============================================================================
Scale C ML Training — Early ICU Prediction on Sirio-Libanes Cohort
===============================================================================
Input:  sirio_early_ml_ready.csv (326 patients, 30 features, ICU_LATER outcome)
Design: Stratified 5-fold CV + 20% held-out test set
Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Output: Figures (ROC, PR, calibration, SHAP), metrics table, saved models
===============================================================================
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, brier_score_loss,
                             confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
import shap

# Plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
import seaborn as sns
sns.set_style('whitegrid')

# ============================================================================
# SETUP
# ============================================================================
INPUT = Path('/home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/sirio_processed/sirio_early_ml_ready.csv')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/02_ML_Model/sirio')
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'figures').mkdir(exist_ok=True)
(OUT_DIR / 'models').mkdir(exist_ok=True)
(OUT_DIR / 'results').mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE C ML TRAINING — EARLY ICU PREDICTION')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load data and split
# ============================================================================
log('\n[STEP 1] Loading ML-ready data...')
df = pd.read_csv(INPUT)
log(f'  Shape: {df.shape}')
log(f'  Outcome: {(df["ICU_LATER"]==1).sum()} positive / {(df["ICU_LATER"]==0).sum()} negative')

feature_cols = [c for c in df.columns if c not in ['PATIENT_VISIT_IDENTIFIER', 'ICU_LATER']]
X = df[feature_cols].values
y = df['ICU_LATER'].values

log(f'\n[STEP 2] Splitting into train (80%) / hold-out test (20%)...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
log(f'  Train: {len(X_train)} patients ({(y_train==1).sum()} positive, {(y_train==0).sum()} negative)')
log(f'  Test:  {len(X_test)} patients ({(y_test==1).sum()} positive, {(y_test==0).sum()} negative)')

# Scale for LR only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 3: Define and train 4 models
# ============================================================================
log('\n[STEP 3] Training 4 models with 5-fold stratified CV...')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced',
                                              random_state=RANDOM_STATE, solver='lbfgs'),
    'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=10,
                                            min_samples_split=5, min_samples_leaf=3,
                                            class_weight='balanced',
                                            random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                  random_state=RANDOM_STATE, eval_metric='logloss',
                                  use_label_encoder=False, verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    class_weight='balanced',
                                    random_state=RANDOM_STATE, verbose=-1)
}

cv_results = {}
test_results = {}
trained_models = {}

for name, model in models.items():
    log(f'\n  Training {name}...')
    
    # Use scaled data for LR, raw for tree-based
    X_tr = X_train_scaled if name == 'LogisticRegression' else X_train
    X_te = X_test_scaled if name == 'LogisticRegression' else X_test
    
    # CV AUC
    cv_aucs = cross_val_score(model, X_tr, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    log(f'    5-fold CV AUC: {cv_aucs.mean():.3f} (±{cv_aucs.std():.3f}) '
        f'[folds: {", ".join(f"{a:.3f}" for a in cv_aucs)}]')
    
    # Fit on full train
    model.fit(X_tr, y_train)
    
    # Predict on test
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_ap = average_precision_score(y_test, y_pred_proba)
    test_brier = brier_score_loss(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    log(f'    Test AUC: {test_auc:.3f} | AP: {test_ap:.3f} | Brier: {test_brier:.3f}')
    log(f'    Test Acc: {test_acc:.3f} | Prec: {test_prec:.3f} | Rec: {test_rec:.3f} | F1: {test_f1:.3f}')
    
    cv_results[name] = {
        'cv_auc_mean': cv_aucs.mean(), 'cv_auc_std': cv_aucs.std(),
        'cv_aucs': cv_aucs.tolist()
    }
    test_results[name] = {
        'auc': test_auc, 'avg_precision': test_ap, 'brier': test_brier,
        'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1,
        'y_pred_proba': y_pred_proba, 'y_pred': y_pred
    }
    trained_models[name] = model

# ============================================================================
# STEP 4: Save metrics table
# ============================================================================
log('\n[STEP 4] Saving metrics table...')
metrics_rows = []
for name in models.keys():
    metrics_rows.append({
        'Model': name,
        'CV_AUC_mean': round(cv_results[name]['cv_auc_mean'], 3),
        'CV_AUC_std': round(cv_results[name]['cv_auc_std'], 3),
        'Test_AUC': round(test_results[name]['auc'], 3),
        'Test_AP': round(test_results[name]['avg_precision'], 3),
        'Test_Brier': round(test_results[name]['brier'], 3),
        'Test_Accuracy': round(test_results[name]['accuracy'], 3),
        'Test_Precision': round(test_results[name]['precision'], 3),
        'Test_Recall': round(test_results[name]['recall'], 3),
        'Test_F1': round(test_results[name]['f1'], 3),
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / 'results' / 'model_comparison.csv', index=False)
log(f'\n{metrics_df.to_string(index=False)}')
log(f'\n  Saved: results/model_comparison.csv')

# Identify best model by test AUC
best_name = metrics_df.loc[metrics_df['Test_AUC'].idxmax(), 'Model']
log(f'\n  Best model by Test AUC: {best_name}')

# ============================================================================
# STEP 5: Figure 1 — ROC curves
# ============================================================================
log('\n[STEP 5] Generating Figure C1 — ROC curves...')
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'LogisticRegression': '#1f77b4', 'RandomForest': '#2ca02c',
          'XGBoost': '#ff7f0e', 'LightGBM': '#d62728'}
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, test_results[name]['y_pred_proba'])
    auc = test_results[name]['auc']
    ax.plot(fpr, tpr, color=colors[name], lw=2.2,
            label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Scale C Early ICU Prediction (Hold-out test)', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C1_ROC_curves.png', dpi=300)
plt.close()
log('  Saved: figures/C1_ROC_curves.png')

# ============================================================================
# STEP 6: Figure 2 — Precision-Recall curves
# ============================================================================
log('\n[STEP 6] Generating Figure C2 — Precision-Recall curves...')
fig, ax = plt.subplots(figsize=(7, 6))
for name in models.keys():
    prec, rec, _ = precision_recall_curve(y_test, test_results[name]['y_pred_proba'])
    ap = test_results[name]['avg_precision']
    ax.plot(rec, prec, color=colors[name], lw=2.2,
            label=f'{name} (AP = {ap:.3f})')
baseline = (y_test==1).mean()
ax.axhline(baseline, color='k', linestyle='--', lw=1, alpha=0.5,
           label=f'Chance (baseline = {baseline:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — Scale C Early ICU Prediction', fontsize=13)
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C2_PR_curves.png', dpi=300)
plt.close()
log('  Saved: figures/C2_PR_curves.png')

# ============================================================================
# STEP 7: Calibration — Platt scaling on best tree-based model
# ============================================================================
log(f'\n[STEP 7] Calibrating best model ({best_name}) with Platt scaling...')

# Re-train with calibration
X_tr_best = X_train_scaled if best_name == 'LogisticRegression' else X_train
X_te_best = X_test_scaled if best_name == 'LogisticRegression' else X_test

base_model = type(trained_models[best_name])(**trained_models[best_name].get_params())
calibrator = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrator.fit(X_tr_best, y_train)
y_cal_proba = calibrator.predict_proba(X_te_best)[:, 1]

cal_brier = brier_score_loss(y_test, y_cal_proba)
cal_auc = roc_auc_score(y_test, y_cal_proba)
log(f'  Before calibration — Brier: {test_results[best_name]["brier"]:.3f}, AUC: {test_results[best_name]["auc"]:.3f}')
log(f'  After calibration  — Brier: {cal_brier:.3f}, AUC: {cal_auc:.3f}')

# Figure 3 — Calibration curve
log('\n[STEP 8] Generating Figure C3 — Calibration curve...')
fig, ax = plt.subplots(figsize=(7, 6))
for name in models.keys():
    prob_true, prob_pred = calibration_curve(y_test, test_results[name]['y_pred_proba'],
                                              n_bins=8, strategy='quantile')
    ax.plot(prob_pred, prob_true, 'o-', color=colors[name], lw=2,
            label=f'{name} (Brier={test_results[name]["brier"]:.3f})', alpha=0.7)

# Add calibrated version
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_cal_proba, n_bins=8, strategy='quantile')
ax.plot(prob_pred_cal, prob_true_cal, 's-', color='purple', lw=2.5,
        label=f'{best_name} + Platt (Brier={cal_brier:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean predicted probability', fontsize=12)
ax.set_ylabel('Fraction of positives', fontsize=12)
ax.set_title('Calibration — Scale C Models', fontsize=13)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C3_calibration.png', dpi=300)
plt.close()
log('  Saved: figures/C3_calibration.png')

# ============================================================================
# STEP 9: SHAP analysis on best model
# ============================================================================
log(f'\n[STEP 9] Computing SHAP values for {best_name}...')

try:
    if best_name in ['RandomForest', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(trained_models[best_name])
        shap_values = explainer.shap_values(X_test)
        # For RF with binary classification, shap returns list or 3D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        # LR — linear explainer
        explainer = shap.LinearExplainer(trained_models[best_name], X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)

    # SHAP summary plot (bar)
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      plot_type='bar', show=False, max_display=20)
    plt.title(f'SHAP Feature Importance — {best_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'C4_SHAP_bar.png', dpi=300)
    plt.close()
    log('  Saved: figures/C4_SHAP_bar.png')

    # SHAP beeswarm
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.title(f'SHAP Summary — {best_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'C5_SHAP_beeswarm.png', dpi=300)
    plt.close()
    log('  Saved: figures/C5_SHAP_beeswarm.png')

    # Save mean |SHAP| per feature
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance.to_csv(OUT_DIR / 'results' / 'shap_importance.csv', index=False)
    log(f'\n  Top 10 features by SHAP importance:')
    for _, r in shap_importance.head(10).iterrows():
        log(f'    {r["feature"]:20s}  {r["mean_abs_shap"]:.4f}')
except Exception as e:
    log(f'  SHAP error: {e}')
    log('  Continuing without SHAP — will compute feature importance differently')

# ============================================================================
# STEP 10: Feature importance (fallback / complement)
# ============================================================================
log('\n[STEP 10] Plotting feature importance for tree models...')
for name in ['RandomForest', 'XGBoost', 'LightGBM']:
    model = trained_models[name]
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        fi.to_csv(OUT_DIR / 'results' / f'feature_importance_{name}.csv', index=False)

# Combined feature importance figure (top 15 by best model's SHAP or importance)
try:
    top_features = shap_importance.head(15)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top_features['feature'][::-1], top_features['mean_abs_shap'][::-1], color='steelblue')
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(f'Top 15 predictive features — {best_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'C6_top_features.png', dpi=300)
    plt.close()
    log('  Saved: figures/C6_top_features.png')
except:
    pass

# ============================================================================
# STEP 11: Confusion matrices
# ============================================================================
log('\n[STEP 11] Confusion matrices for all models...')
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, name in zip(axes, models.keys()):
    cm = confusion_matrix(y_test, test_results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No ICU', 'ICU'], yticklabels=['No ICU', 'ICU'], cbar=False)
    ax.set_title(f'{name}\nAUC={test_results[name]["auc"]:.3f}', fontsize=11)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(OUT_DIR / 'figures' / 'C7_confusion_matrices.png', dpi=300)
plt.close()
log('  Saved: figures/C7_confusion_matrices.png')

# ============================================================================
# STEP 12: Save models for later Scale D external validation
# ============================================================================
log('\n[STEP 12] Saving trained models for Scale D external validation...')
for name, model in trained_models.items():
    with open(OUT_DIR / 'models' / f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
with open(OUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(OUT_DIR / 'models' / 'calibrator.pkl', 'wb') as f:
    pickle.dump(calibrator, f)
with open(OUT_DIR / 'models' / 'feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
log(f'  Saved models to: {OUT_DIR}/models/')

# ============================================================================
# DONE
# ============================================================================
log('\n' + '='*75)
log(f'SCALE C TRAINING COMPLETE: {datetime.now().isoformat()}')
log(f'Output directory: {OUT_DIR}')
log(f'  - figures/: {len(list((OUT_DIR / "figures").glob("*.png")))} figures')
log(f'  - models/: {len(list((OUT_DIR / "models").glob("*.pkl")))} pickle files')
log(f'  - results/: metrics + importance tables')
log('='*75)

# Save log
with open(OUT_DIR / 'training_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
print(f'\nLog saved: {OUT_DIR}/training_log.txt')
