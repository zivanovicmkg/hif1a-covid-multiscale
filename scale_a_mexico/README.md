# Scale A — Population analysis (Mexican DGE Open Data)

Population-level analysis of N=35,268 hospitalized PCR-confirmed COVID-19 patients selected from the Mexican Open Data on COVID-19 cases maintained by the Dirección General de Epidemiología (DGE), Secretaría de Salud, Gobierno de México.

## Cohort summary

| Parameter | Value |
|---|---|
| Source | Mexican DGE Open Data, Secretaría de Salud |
| Sampling period | (see manuscript Methods §2.2 for exact period) |
| Inclusion criteria | Hospitalized, PCR-confirmed SARS-CoV-2 infection |
| Final N | 35,268 |
| Primary outcomes | In-hospital mortality, ICU transfer |

## Scripts

| Script | Function |
|---|---|
| `scripts/13_scale_a_mexico.py` | Main Scale A pipeline: data filtering, demographic stratification, mortality logistic regression with comorbidity adjustment |
| `scripts/14_scale_a_extensions.py` | Extended sensitivity analyses, age-stratified models, no-pneumonia subset |

## Data access

The raw Mexican DGE Open Data CSV is **not redistributed here** because of its size (~900 MB current snapshot, daily-updated). See [`data/README.md`](data/README.md) for the download procedure and citation.

## Reproducing the analysis

```bash
# 1) Download the source CSV (see data/README.md)
# 2) Run the main pipeline
python scripts/13_scale_a_mexico.py

# 3) Run extensions
python scripts/14_scale_a_extensions.py
```

Outputs (small CSV/PKL) are written to `results/` and `../scale_a_mexico/` per the script's internal paths. Adjust input/output paths at the top of each script as needed.

## Results in this repository

- `results/scale_a_metrics.csv` — overall model performance
- `results/scale_a_coefficients.csv` — logistic regression coefficients with 95% CI
- `results/scale_a_age_stratified.csv` — age-stratified analyses
- `results/scale_a_composite_coefficients.csv` — composite-feature variant
- `results/scale_a_no_pneumonia_coefficients.csv` — sensitivity analysis excluding pneumonia
- `figures/A1_ROC_comparison.png`, `A2_feature_importance.png`, `A3_OR_forest_plot.png`, `A4_extensions.png` — see `figures/main/` in the repo root
