# Reproducibility guide

End-to-end instructions for reproducing the multi-scale analysis. Each scale is independent — you can run scales in any order, or run a subset.

## 1. Environment

```bash
# Clone the repository
git clone https://github.com/zivanovicmkg/hif1a-covid-multiscale.git
cd hif1a-covid-multiscale

# Create a Python 3.10+ environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

Tested under Python 3.10 and 3.11 on Linux (Ubuntu 22.04+).

## 2. Per-scale reproduction

### Scale A — Mexican DGE Open Data

```bash
cd scale_a_mexico/
# 1) Download Mexican DGE CSV per data/README.md
# 2) Run main pipeline
python scripts/13_scale_a_mexico.py
# 3) Run sensitivity / extensions
python scripts/14_scale_a_extensions.py
```

### Scale B — Bulk RNA-seq (5 GEO cohorts)

```bash
cd scale_b_bulk/
# 1) Download GEO series matrices and supplementary files per data/README.md
# 2) Per-cohort processing
for g in GSE152075 GSE157103 GSE171110 GSE212861 GSE300696; do
    python scripts/per_cohort/$g/parse_metadata_$g.py
    python scripts/per_cohort/$g/make_all_figures_$g.py  # or make_panel_$g.py
done
# 3) Cross-cohort meta-analysis
python scripts/meta_analysis/covid_master_analysis.py
python scripts/meta_analysis/merge_covid_datasets.py
python scripts/16_scale_b_rigorous.py    # final consolidated pipeline
```

### Scale B — scRNA-seq (GSE234904)

```bash
cd scale_b_scrna/
# 1) Download GSE234904_RAW.tar from GEO per data/README.md and unpack
# 2) Four-phase pipeline
python scripts/17_scale_b_scRNA_phase1.py    # load, demultiplex, QC
python scripts/18_scale_b_scRNA_phase2.py    # filter, normalize, HVG, PCA
python scripts/19_scale_b_scRNA_harmony.py   # Harmony integration, UMAP, clustering
python scripts/20_scale_b_scRNA_phase3.py    # HIF1A scoring, DEG, weaning stratification
```

NOTE: phase 1 produces a ~90 MB `merged_raw.h5ad`; phase 3 produces a ~1.8 GB `merged_harmony_scored.h5ad`. Ensure sufficient disk space.

### Scale C — Sírio-Libanês (Brazil)

```bash
cd scale_c_sirio_brazil/
# 1) Download Kaggle Sírio-Libanês dataset per data/README.md
# 2) Preprocessing
python scripts/02_explore_sirio_libanes.py
python scripts/05_preprocess_sirio.py
python scripts/06_preprocess_sirio_v2.py
# 3) Training (LR / RF / XGBoost / LightGBM with calibration)
python scripts/07_train_sirio_models.py
# 4) Polished final pipeline (bootstrap CI, DCA, SHAP)
python scripts/15_scale_c_polish.py
```

### Scale D — Kragujevac (Serbia)

```bash
cd scale_d_kragujevac/
# 1) Obtain Kragujevac data per data/README.md (on request, ethics 01/20-498)
# 2) External validation of Scale C model on Kragujevac
python scripts/08_external_validation_kragujevac.py
# 3) Local Kragujevac-only model
python scripts/09_local_kragujevac_model.py
# 4) Conservative + sanity checks
python scripts/10_scale_d_conservative.py
python scripts/11_scale_d_vif_sanity.py
# 5) Final logistic regression with LOOCV
python scripts/12_scale_d_final.py
```

### Literature mining (Supplementary S5)

The OCTE engine (https://doi.org/10.5281/zenodo.17093171) must be installed separately. Then:

```bash
cd literature_mining/
python scripts/modified_framework/covid_counts.py
python scripts/custom_helpers/covid_figures.py
python scripts/custom_helpers/covid_prisma.py
python scripts/custom_helpers/covid_topics.py
```

## 3. Path configuration

Each script declares its input/output paths near the top of the file (typically the first 10–20 lines). Adjust these to match your local layout. The scripts are independent — no shared global configuration file.

## 4. Computational requirements

| Scale | RAM | Disk | Runtime (rough, single thread) |
|---|---|---|---|
| A | 8 GB | 1 GB | 10–20 min |
| B (bulk, all 5) | 16 GB | 5 GB | 1–2 h total |
| B (scRNA) | 32 GB | 10 GB | 1–3 h (Harmony is the bottleneck) |
| C | 8 GB | 1 GB | 20–40 min |
| D | 4 GB | < 100 MB | < 5 min (small N) |

The scRNA pipeline benefits substantially from a multi-core CPU. None of the pipelines require a GPU.

## 5. Expected outputs

Each scale produces:
- Small numerical results (CSV / TSV) → committed under `scale_*/results/`
- Figures (PNG) → copied to repo-level `figures/main/` and `figures/supplementary/`
- Serialized models (PKL, where applicable) → `scale_*/models/`
- (Scale B scRNA only) Large `.h5ad` files → NOT committed; regenerable by the pipeline

If a script fails, the most common cause is a missing or differently-named input file. Check the top of the script for expected paths.
