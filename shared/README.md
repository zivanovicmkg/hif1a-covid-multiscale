# Shared preprocessing utilities

Cross-scale preprocessing scripts used during the early data-acquisition and harmonization phases of the project. These are not part of the final per-scale analytical pipelines (which live in `scale_*/scripts/`), but are kept in the repository for full provenance and reproducibility of the dataset selection process.

## Scripts

| Script | Function |
|---|---|
| `01_download_datasets.py` | Programmatic download helpers for public clinical COVID-19 datasets |
| `02_clean_datasets.py` | Initial cleaning and column normalization |
| `03_harmonize_variables.py` | Cross-dataset variable name harmonization |
| `04_merge_datasets.py` | Merging logic for cross-dataset comparisons |
| `05_feature_engineering.py` | Feature engineering primitives (lab-value ratios, severity scores) |
| `06_train_ml_models.py` | Generic multi-model training scaffold (LR / RF / XGBoost / LightGBM with calibration) |
| `07_evaluate_models.py` | Performance evaluation, bootstrap CI, calibration diagnostics |

## Note

In the final manuscript pipeline, Scale C uses scale-specific versions of preprocessing and training (`scale_c_sirio_brazil/scripts/05_preprocess_sirio.py`, `06_preprocess_sirio_v2.py`, `07_train_sirio_models.py`), and Scale D uses `scale_d_kragujevac/scripts/08–12_*.py`. The scripts here are the **earlier, generic versions** that were later specialized per scale. They are kept here for transparency of the development history.
