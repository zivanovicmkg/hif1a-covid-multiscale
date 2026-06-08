# Data access overview

This document summarizes the upstream data sources used in each scale. **No raw patient-level data is redistributed in this repository.** For per-scale download procedures, see the corresponding `data/README.md`.

## Summary

| Scale | Dataset | Source | Access type | License |
|---|---|---|---|---|
| A | Mexican DGE Open Data — Influenza, COVID-19 y otros virus respiratorios | https://www.gob.mx/salud/documentos/datos-abiertos-152127 | Public download | Términos de Libre Uso DGE |
| B (bulk) | GSE152075, GSE157103, GSE171110, GSE212861, GSE300696 | NCBI GEO | Public (no account required) | Per GEO accession |
| B (scRNA) | GSE234904 | NCBI GEO | Public (no account required) | Per GEO accession |
| C | Sírio-Libanês COVID-19 ICU Prediction | Kaggle | Free Kaggle account required | CC0 1.0 (per Kaggle) |
| D | Kragujevac clinical-genetic cohort (N=93) | University Clinical Centre Kragujevac, Serbia | **On reasonable request**, ethics 01/20-498 | Restricted (see scale_d_kragujevac/data/README.md) |

## Per-scale download instructions

See:
- [`scale_a_mexico/data/README.md`](../scale_a_mexico/data/README.md)
- [`scale_b_bulk/data/README.md`](../scale_b_bulk/data/README.md)
- [`scale_b_scrna/data/README.md`](../scale_b_scrna/data/README.md)
- [`scale_c_sirio_brazil/data/README.md`](../scale_c_sirio_brazil/data/README.md)
- [`scale_d_kragujevac/data/README.md`](../scale_d_kragujevac/data/README.md)

## Citation requirements

When reusing these datasets, please cite **both**:
1. The upstream data provider (see per-scale `data/README.md` for exact citations)
2. The present multi-scale study (see top-level `CITATION.cff`)
