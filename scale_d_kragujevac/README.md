# Scale D — Kragujevac clinical-genetic cohort (Serbia)

Prospective single-center clinical-genetic cohort study of hospitalized COVID-19 patients at the University Clinical Centre Kragujevac, Serbia, integrating clinical, biochemical, and genotypic predictors of in-hospital mortality.

## Cohort

- **Source**: Infectious Diseases Clinic and Corona Center, University Clinical Centre Kragujevac (UCC KG), Serbia
- **Sampling period**: From May 2020
- **Inclusion criteria**: PCR-confirmed SARS-CoV-2 infection, hospitalized
- **Initial N**: 100 → **Final analytical N**: 93 (complete clinical + genotypic data)
- **Primary outcome**: In-hospital mortality
- **Ethics approval**: University Clinical Centre Kragujevac, decision **No. 01/20-498**

## Predictors

### Demographic & clinical (baseline at admission)
Age, sex, comorbidities (diabetes, hypertension, COPD, neurological diagnosis, malignant diagnosis, chronic renal failure), X-ray score, SpO₂, pO₂

### Biochemical (baseline at admission)
CRP, LDH, D-dimer, ferritin, IL-6, procalcitonin, urea, creatinine, AST, ALT, troponin, proBNP

### Genotypic
Two HIF1A polymorphisms in the oxygen-dependent degradation (ODD) domain:
- **rs11549465** (c.1744C>T, p.Pro582Ser) — TaqMan Assay C__25473074_10
- **rs41508050** (c.1762G>A, p.Ala588Thr) — TaqMan Assay C__86499352_10

## Genotyping platform

- **DNA extraction**: PureLink™ Genomic DNA Mini Kit (Invitrogen, Cat. K182002)
- **Master mix**: TaqMan™ Genotyping Master Mix (Applied Biosystems, Cat. 4371355)
- **Instrument**: Mic qPCR Cycler (BioMolecular Systems, Yatala, Australia)
- **Quality control**: All samples in technical duplicate; inter-run concordance 97.6% (81/83 concordant pairs)

This platform has been previously validated by our laboratory team in earlier COVID-19 host-genetic studies at the same clinical center, covering interferon-lambda (Matic et al. 2023a, 2023b *J Med Virol*) and ACE2/TMPRSS2 (Matic et al. 2024 *Front Med*) loci. The present study extends that institutional host-genetic line to the HIF1A pathway.

## Scripts

| Script | Function |
|---|---|
| `scripts/08_external_validation_kragujevac.py` | External validation of Scale C XGBoost model on Kragujevac cohort |
| `scripts/09_local_kragujevac_model.py` | Local XGBoost model trained on Kragujevac data only |
| `scripts/10_scale_d_conservative.py` | Conservative subset analysis (no-hypertension model, VIF check) |
| `scripts/11_scale_d_vif_sanity.py` | Multicollinearity diagnostics |
| `scripts/12_scale_d_final.py` | Final Scale D logistic regression with LOOCV |

## Data access

The Kragujevac patient-level data **is not redistributed** in this repository. See [`data/README.md`](data/README.md) for the on-request access procedure and the ethics framework.

## Models (`models/`)

Trained models contain only the model coefficients and feature ordering, not patient-level data:
- `scale_d_final_model.pkl` — final LR model
- `scaler.pkl`, `feature_names.pkl`
- `local_xgboost.pkl`, `local_feature_names.pkl`, `local_imputer.pkl`
- `local_shap_importance.csv`

## Results in this repository

- `results/external_validation_metrics.csv`
- `results/conservative/conservative_metrics.csv`, `coefficients_no_hta.csv`, `vif_analysis.csv`, `sensitivity_analysis.csv`, `stratified_performance.csv`
- `results/local_model/local_kragujevac_metrics.csv`, `local_shap_importance.csv`
- `results/scale_d_final/scale_d_final_predictions.csv`, `Table_D_final_coefficients.csv`, `Table_D_final_metrics.csv`, `stratified_HIF1A_performance.csv`
- Figures D3–D13 copied to `figures/main/`

## Key findings

- Diabetes is the strongest mortality predictor (OR 2.70, p=0.002)
- Neurological diagnosis (OR 2.43, p=0.012)
- HIF1A rs11549465 CT/TT carriers show numerically lower mortality (9.1% vs 35.9% in CC, OR≈0.20), consistent with a protective effect of the variant allele in this cohort
