# Scale B — Bulk RNA-seq meta-analysis (5 GEO cohorts)

Meta-analysis of bulk RNA-seq host transcriptomic data across five independent NCBI GEO cohorts, focused on a curated 10-gene hypoxia–inflammation panel including HIF1A.

## Cohorts

| Accession | First author / year | Tissue | Samples (used) |
|---|---|---|---|
| GSE152075 | Lieberman 2020 (*PLOS Biol*) | Upper-airway swabs | ~450 |
| GSE157103 | Overmyer 2021 (*Cell Syst*) | Whole blood | 126 |
| GSE171110 | Lévy 2021 (*iScience*) | Whole blood | 44 |
| GSE212861 | Rombauts 2023 (*Biomedicines*) | PBMC | ~250 |
| GSE300696 | Ryu / An 2025 (*Commun Biol*) | Whole blood | ~250 |

Total ≈ 1,100 samples across five independent cohorts.

## 10-gene hypoxia–inflammation panel

- **Hypoxia / glycolysis**: HIF1A, LDHA, VEGFA, SLC2A1 (GLUT1)
- **Inflammation / cytokines**: IL6, TNF, CXCL8 (IL8), STAT3
- **Tissue remodeling**: MMP9
- **Innate immunity**: TLR4

## Scripts

### Per-cohort processing — `scripts/per_cohort/GSEnnnnnn/`

Each cohort has its own processing scripts following the same template:
- `parse_metadata_GSEnnnnnn.py` — parse the series matrix metadata
- `make_all_figures_GSEnnnnnn.py` or `make_panel_GSEnnnnnn.py` — score panel + figures
- (cohort-specific cleaning if needed, e.g. `clean_counts_GSE212861.py`)

### Meta-analysis — `scripts/meta_analysis/`

| Script | Function |
|---|---|
| `covid_master_analysis.py` | Cross-cohort hypoxia–inflammation score harmonization |
| `merge_covid_datasets.py` | Build master feature matrix across cohorts |
| `severity_correlation.py`, `severity_score_test.py`, `severity_stats.py`, `severity_trend.py` | Severity-aware effect-size analyses |
| `feature_importance.py` | Random forest / mutual information ranking |
| `gene_heatmap.py` | Per-gene direction heatmap across cohorts |
| `roc_analysis.py` | ROC AUC per cohort |
| `covid_vs_control_plot.py`, `covid_vs_control_violin.py` | Distribution plots |

### Rigorous pipeline — `scripts/16_scale_b_rigorous.py`

Final consolidated Scale B bulk pipeline used in the manuscript. Computes per-cohort Cohen's *d* with bootstrap CI, meta-analysis forest plot, gene-direction consistency matrix (5/5 vs 4/5 agreement), and per-cohort AUC table.

## Data access

GEO is public; no special access required. See [`data/README.md`](data/README.md) for accession-specific download instructions and citations.

## Results in this repository

- `results/meta_analysis_effect_sizes.csv` — per-gene Cohen's d across cohorts
- `results/per_cohort_auc_table.csv` — AUC by cohort
- `results/per_cohort_gene_stats.csv` — per-gene/per-cohort statistics
- Figure outputs from `Scale_B_rigorous/` are copied to `figures/main/` and `figures/supplementary/`

## Key finding

LDHA and HIF1A are up-regulated in 5/5 cohorts (the most consistent signal in the panel); STAT3 and MMP9 in 4/5; TNF down-regulates in 4/5 (consistent with late-phase immunosuppression).
