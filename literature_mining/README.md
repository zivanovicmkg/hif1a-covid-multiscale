# Literature mining (Supplementary Section S5)

Bibliometric analysis of COVID-19 research output 2019–2026 supporting Supplementary Section S5 of the manuscript, including: AI/ML adoption rates, HIF1A-related publication trends, PRISMA flow, and topic clustering.

## Relationship to OCTE framework

The core mining engine used here is the **OCTE AI-Mining Framework**, which is **separately archived on Zenodo**:

> Živanović M. *octe-ai-mining-framework* (v1.0.1) [Computer software]. Zenodo. 2025. https://doi.org/10.5281/zenodo.17093171

This directory contains **only the COVID-specific customizations** — search-term configurations, post-processing scripts, and figure-generation helpers — that wrap the upstream OCTE framework. To reproduce the bibliometric analysis end-to-end, install the OCTE framework from its Zenodo record (or from its public GitHub repo) and run the helpers in this directory.

## Scripts

### `scripts/modified_framework/`
COVID-specific configuration for the upstream OCTE framework:
- `covid_config.py` — PubMed query group definitions (G1–G5)
- `covid_counts.py` — temporal count extraction
- `covid_top_cited.py` — top-cited-paper extraction
- `validate_queries.py` — query validation logic

### `scripts/custom_helpers/`
Post-processing and figure-generation helpers:
- `covid_bibtex.py` — assemble BibTeX bibliography per group
- `covid_figures.py` — generate growth curves, AI-ratio plots
- `covid_prisma.py` — PRISMA flow diagram
- `covid_topics.py` — topic clustering / top-term extraction
- `build_s5_docx.py` — assemble Supplementary Section S5 DOCX
- `finalize_manuscript.py` — finalization helpers

## Query groups

| Group | Focus | Description |
|---|---|---|
| G1 | Master | All COVID-19 publications, 2019–2026 |
| G2 | AI / ML in COVID-19 | COVID-19 publications using AI or ML methodologies |
| G3 | HIF1A in COVID-19 | COVID-19 publications mentioning HIF1A or hypoxia-inducible pathway |
| G4 | Recent (2024–2026) | COVID-19 publications in the most recent three-year window |
| G5 | AI-assisted vaccine design | AI/ML methods for next-generation vaccine design |

## Key findings (manuscript §1, §4.1, Supplementary S5)

- 496,768 COVID-19 publications in PubMed across 2019–2026
- AI/ML used in only 2.7% of COVID-19 publications; the AI share roughly tripled from 1.5% (2020) to 4.5% (2026)
- Only 139 HIF1A-related COVID-19 publications across the entire period; 47 in 2024–2026
- None of these combined populational, transcriptomic, single-cell, clinical-ML, and host-genetic evidence within a unified framework

## Results in this repository

- `results/prisma/prisma_counts.csv` — PRISMA flow counts
- `results/topic_analysis/topic_clusters.csv`, `top_terms.csv`
- `results/raw_searches/covid_lit_matrix.csv`, `covid_ratio_by_year.csv`, query group CSV files

## Figures

- `figures/trends/FigS5_1_growth.pdf|.png` — annual growth curves
- `figures/heatmaps/FigS5_2_heatmap.pdf|.png` — topic-by-year heatmap
- `figures/ratios/FigS5_3_ai_ratio.pdf|.png` — AI share trend
- `figures/wordclouds/FigS5_4_wordcloud.pdf|.png`, `FigS5_4b_topterms_bar.png`
- `figures/prisma/FigS5_5_prisma.pdf|.png` — PRISMA flow diagram

## Note on PubMed abstracts

The raw PubMed abstracts retrieved during mining (300+ small `.txt` files in `data/abstracts/`) are **not committed to this repository** to keep the git history clean. They are packaged separately and uploaded to Zenodo alongside the code release for full reproducibility.
