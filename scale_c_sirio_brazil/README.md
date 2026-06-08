# Scale C — Sírio-Libanês ICU Prediction (Brazil)

Calibrated XGBoost machine-learning model with SHAP interpretability for ICU transfer prediction in hospitalized COVID-19 patients from the Sírio-Libanês Hospital cohort (São Paulo, Brazil).

## Cohort

- **Source**: Sírio-Libanês Hospital, São Paulo, Brazil
- **Public dataset**: Kaggle — *COVID-19 — Clinical Data to assess diagnosis*
- **N**: 326 hospitalized patients
- **Outcome**: ICU transfer (binary)
- **Predictors**: Demographics, comorbidities, vital signs, biochemical parameters at admission

## Scripts

| Script | Function |
|---|---|
| `scripts/02_explore_sirio_libanes.py` | Initial data exploration and quality check |
| `scripts/05_preprocess_sirio.py` | Cleaning v1 |
| `scripts/06_preprocess_sirio_v2.py` | Cleaning v2 (final): admission-window extraction, feature engineering, missingness handling |
| `scripts/07_train_sirio_models.py` | Train multiple model families (LR, RF, XGBoost, LightGBM) with calibration |
| `scripts/15_scale_c_polish.py` | Final polished pipeline: bootstrap 95% CI, decision-curve analysis, SHAP global + local |

## Models compared

- Logistic Regression (baseline)
- Random Forest
- LightGBM
- XGBoost (final selected model)

All models are calibrated via isotonic / Platt scaling and evaluated with bootstrap 95% CI.

## Data access

The Sírio-Libanês data is freely available on Kaggle (free Kaggle account required). See [`data/README.md`](data/README.md) for the exact dataset URL and download steps.

## Models (`models/`)

Serialized models are included in the repository:
- `LogisticRegression.pkl`
- `RandomForest.pkl`
- `LightGBM.pkl`
- `XGBoost.pkl` — final selected model
- `calibrator.pkl` — calibration wrapper
- `scaler.pkl` — feature standardizer
- `feature_names.pkl` — feature ordering

These models were trained exclusively on the public Sírio-Libanês Kaggle dataset and contain no patient-identifying information.

## Results in this repository

- `results/model_comparison.csv` — performance across model families
- `results/feature_importance_XGBoost.csv`, `..._RandomForest.csv`, `..._LightGBM.csv`
- `results/shap_importance.csv` — SHAP global importance
- `results/polish/scale_c_baseline_comparison.csv`
- `results/polish/scale_c_bootstrap_CI.csv`
- `results/polish/scale_c_dca_data.csv` — decision-curve analysis
- Figures C1–C7 (ROC, PR, calibration, SHAP bar, SHAP beeswarm, top-features, confusion matrices) copied to `figures/main/`
