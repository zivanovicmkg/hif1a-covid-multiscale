# HIF1A-COVID Multi-scale Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Source code and reproducibility materials for a multi-scale analysis of the **HIF1A (hypoxia-inducible factor 1-alpha)** axis in COVID-19 host biology, integrating evidence across four analytical scales:

- **Scale A** — Population-level analysis of N=35,268 hospitalized PCR-confirmed COVID-19 patients from the Mexican DGE Open Data (Secretaría de Salud, Dirección General de Epidemiología).
- **Scale B** — Bulk RNA-seq meta-analysis across five GEO cohorts (N=1,100 samples) using a 10-gene hypoxia-inflammation panel, plus single-cell RNA-seq of 25,773 peripheral neutrophils (GSE234904) stratified by 28-day extubation outcome.
- **Scale C** — Calibrated XGBoost + SHAP machine-learning model for ICU transfer prediction on the Sírio-Libanês cohort (Brazil, N=326).
- **Scale D** — Clinical-genetic logistic regression on a Kragujevac (Serbia) cohort (N=93 PCR-confirmed COVID-19 hospitalizations), genotyped for two HIF1A polymorphisms (rs11549465, rs41508050) with leave-one-out cross-validation.

## Repository layout

```
hif1a-covid-multiscale/
├── scale_a_mexico/          # Population (DGE Open Data)
├── scale_b_bulk/            # Bulk RNA-seq meta-analysis (5 GEO cohorts)
├── scale_b_scrna/           # Single-cell RNA-seq (GSE234904)
├── scale_c_sirio_brazil/    # Brazilian clinical ML cohort
├── scale_d_kragujevac/      # Serbian clinical-genetic cohort
├── shared/                  # Cross-scale preprocessing utilities
├── literature_mining/       # Bibliometric pipeline (Supplementary Section S5)
├── figures/                 # Final manuscript figures (main + supplementary)
└── docs/                    # Data access notes, reproducibility instructions
```

Each scale directory contains:
- `scripts/` — Python scripts used to produce the analysis
- `data/README.md` — instructions to obtain the corresponding dataset (raw data are not redistributed)
- `results/` — derived CSV/TSV tables that are small enough to version
- `models/` — serialized models where applicable

## Citation

If you use this code or the derived results, please cite:

> Živanović M, et al. *Multi-scale analysis of the HIF1A hypoxia axis in COVID-19 host biology: an integrated population, transcriptomic, and clinical-genetic study.* (Manuscript in preparation.)

and the archived release:

> Živanović M. hif1a-covid-multiscale (v1.0.0) [Computer software]. Zenodo. 2026. https://doi.org/10.5281/zenodo.XXXXXXXX

A machine-readable citation file is provided in [`CITATION.cff`](CITATION.cff).

## Data availability

Raw datasets are **not redistributed** in this repository because of their size and licensing. Each scale's `data/README.md` documents how to retrieve the corresponding source. Briefly:

| Scale | Source | Access |
|---|---|---|
| A | Mexican DGE Open Data (Secretaría de Salud) | Public download — https://www.gob.mx/salud/documentos/datos-abiertos-152127 |
| B (bulk) | NCBI GEO — GSE152075, GSE157103, GSE171110, GSE212861, GSE300696 | Public download via GEO accession |
| B (scRNA) | NCBI GEO — GSE234904 | Public download via GEO accession |
| C | Kaggle — Sírio-Libanês ICU prediction dataset | Public download (Kaggle account required) |
| D | Kragujevac clinical cohort (N=93) | Available on reasonable request from the corresponding author. Approved under ethics decision **01/20-498** of the University Clinical Centre Kragujevac, Serbia |

See [`docs/data_access.md`](docs/data_access.md) for full details.

## Reproducibility

A minimal Python 3.10+ environment is required. Install dependencies:

```bash
pip install -r requirements.txt
# OR with conda:
# conda env create -f environment.yml
```

End-to-end reproduction instructions for each scale are provided in [`docs/reproducibility.md`](docs/reproducibility.md).

## License

- **Code** is released under the [MIT License](LICENSE).
- **Data files and derived numerical results** in this repository are released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).

## Funding and acknowledgments

This work was supported by the **EngVIPO** EU-funded project. The clinical-genetic component (Scale D) was performed at the **BioIRC Center of Excellence for Bioengineering, University of Kragujevac, Serbia**, in collaboration with the University Clinical Centre Kragujevac under ethics decision 01/20-498.

## Author

**Marko Živanović**
BioIRC Center of Excellence for Bioengineering
University of Kragujevac, Serbia
ORCID: [`<ORCID_ID_PLACEHOLDER>`](https://orcid.org/<ORCID_ID_PLACEHOLDER>)
