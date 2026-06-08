"""
===============================================================================
Scale B scRNA-seq — Phase 2b: Harmony batch integration
===============================================================================
Problem identified in phase 2:
    Leiden clusters dominated by single samples (e.g., cluster 0 = 99.6% S3116).
    This is batch effect, not biology.

Solution: Harmony integration on PCA space, keyed by sample_id.
    Harmony iteratively corrects the PC embedding so that cells cluster
    by biology, not by sample origin.

Output:
    - merged_harmony.h5ad (integrated)
    - B6_UMAP_before_after.png (side-by-side comparison)
    - B6_UMAP_overview.png (updated with Harmony)
===============================================================================
"""
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, dpi_save=300, fontsize=10)

OUT = Path('/home/marko-b2/COVID_AI_Project/03_Transcriptomics_GEO/Scale_B_scRNA')
FIG = OUT / 'figures'
sc.settings.figdir = str(FIG)

print('='*75)
print('SCALE B scRNA-seq — PHASE 2b: HARMONY BATCH INTEGRATION')
print('='*75)

# ============================================================================
# Load already-processed data (post normalize + PCA + pre-harmony UMAP)
# ============================================================================
print('\n[STEP 1] Loading merged_processed.h5ad (has PCA, pre-Harmony UMAP)...')
adata = sc.read_h5ad(OUT / 'merged_processed.h5ad')
print(f'  Shape: {adata.shape}')
print(f'  Available obsm: {list(adata.obsm.keys())}')

# Save the pre-Harmony UMAP as a separate embedding
adata.obsm['X_umap_preHarmony'] = adata.obsm['X_umap'].copy()
adata.obs['leiden_preHarmony'] = adata.obs['leiden'].copy()
print('  Saved pre-Harmony UMAP as X_umap_preHarmony')

# ============================================================================
# Harmony integration
# ============================================================================
print('\n[STEP 2] Running Harmony integration (key = sample_id)...')
print('  This corrects the PCA embedding to remove sample-driven clustering')

from scanpy.external.pp import harmony_integrate
harmony_integrate(adata, key='sample_id', basis='X_pca', adjusted_basis='X_pca_harmony')
print(f'  Harmony output shape: {adata.obsm["X_pca_harmony"].shape}')

# ============================================================================
# Rerun neighbors + UMAP + clustering on Harmony-corrected PCs
# ============================================================================
print('\n[STEP 3] Recomputing kNN graph on Harmony embedding...')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep='X_pca_harmony',
                key_added='harmony')

print('\n[STEP 4] Recomputing UMAP on Harmony embedding...')
sc.tl.umap(adata, random_state=42, neighbors_key='harmony')

print('\n[STEP 5] Re-clustering on Harmony embedding...')
sc.tl.leiden(adata, resolution=0.5, random_state=42,
             neighbors_key='harmony', key_added='leiden_harmony')
n_clusters_h = adata.obs['leiden_harmony'].nunique()
print(f'  Post-Harmony: {n_clusters_h} Leiden clusters')
print(f'  Cluster sizes:')
print(adata.obs['leiden_harmony'].value_counts().sort_index().to_string())

# Save Harmony version
out_file = OUT / 'merged_harmony.h5ad'
adata.write_h5ad(out_file)
print(f'\nSaved: {out_file}')
print(f'Size: {out_file.stat().st_size / (1024**2):.1f} MB')

# ============================================================================
# Cluster × sample — after Harmony (hope: less dominated)
# ============================================================================
print('\n=== Cluster × Sample AFTER Harmony (should be more mixed) ===')
ct_h = pd.crosstab(adata.obs['leiden_harmony'], adata.obs['sample_id'])
# Normalize per cluster to see fractions
ct_h_norm = ct_h.div(ct_h.sum(axis=1), axis=0).round(2)
print('Absolute counts:')
print(ct_h.to_string())
print('\nFractions (row-normalized):')
print(ct_h_norm.to_string())

# Max single-sample dominance per cluster
max_dom = ct_h_norm.max(axis=1)
print(f'\nMax single-sample dominance per cluster:')
for c, v in max_dom.items():
    flag = '  BATCH-DOMINATED' if v > 0.8 else ''
    print(f'  Cluster {c}: {v*100:.0f}%{flag}')

# ============================================================================
# Cluster × Condition AFTER Harmony
# ============================================================================
print('\n=== Cluster × Condition AFTER Harmony ===')
ct_cond = pd.crosstab(adata.obs['leiden_harmony'], adata.obs['condition'])
ct_cond['Total'] = ct_cond.sum(axis=1)
ct_cond['%COVID'] = (ct_cond['COVID'] / ct_cond['Total'] * 100).round(1)
print(ct_cond.to_string())

# ============================================================================
# Cluster × Weaning AFTER Harmony
# ============================================================================
print('\n=== Cluster × Weaning (COVID only) AFTER Harmony ===')
covid_only = adata.obs[adata.obs['condition'] == 'COVID']
ct_w = pd.crosstab(covid_only['leiden_harmony'], covid_only['weaning_group'])
ct_w['Total'] = ct_w.sum(axis=1)
ct_w['%NonWeaned'] = (ct_w['NonWeaned'] / ct_w['Total'] * 100).round(1)
print(ct_w.to_string())

# ============================================================================
# Figure B6: Side-by-side before/after Harmony
# ============================================================================
print('\n[STEP 6] Figure B6 — before/after Harmony comparison...')

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# ROW 1: Before Harmony
adata_pre = adata.copy()
adata_pre.obsm['X_umap'] = adata_pre.obsm['X_umap_preHarmony']
sc.pl.umap(adata_pre, color='sample_id', ax=axes[0, 0], show=False,
           title='BEFORE Harmony: colored by sample (batch)',
           frameon=False)
sc.pl.umap(adata_pre, color='condition', ax=axes[0, 1], show=False,
           title='BEFORE Harmony: colored by condition',
           palette={'COVID': '#E24A4A', 'Healthy': '#4A90E2'},
           frameon=False)
sc.pl.umap(adata_pre, color='leiden_preHarmony', ax=axes[0, 2], show=False,
           title='BEFORE Harmony: Leiden clusters',
           legend_loc='on data', legend_fontsize=9,
           frameon=False)

# ROW 2: After Harmony
sc.pl.umap(adata, color='sample_id', ax=axes[1, 0], show=False,
           title='AFTER Harmony: colored by sample',
           frameon=False)
sc.pl.umap(adata, color='condition', ax=axes[1, 1], show=False,
           title='AFTER Harmony: colored by condition',
           palette={'COVID': '#E24A4A', 'Healthy': '#4A90E2'},
           frameon=False)
sc.pl.umap(adata, color='leiden_harmony', ax=axes[1, 2], show=False,
           title=f'AFTER Harmony: Leiden clusters (n={n_clusters_h})',
           legend_loc='on data', legend_fontsize=9,
           frameon=False)

plt.suptitle(
    'Figure B6 — Harmony batch integration (25,773 neutrophils × 6 patients)\n'
    'Top: pre-Harmony (clusters driven by sample); Bottom: post-Harmony (biology-driven)',
    fontsize=13, y=1.00
)
plt.tight_layout()
plt.savefig(FIG / 'B6_UMAP_before_after_Harmony.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: B6_UMAP_before_after_Harmony.png')

# ============================================================================
# Also save a clean "after Harmony only" 4-panel for docx
# ============================================================================
print('\n[STEP 7] Figure B6 clean — 4-panel (after Harmony only)...')
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

sc.pl.umap(adata, color='leiden_harmony', ax=axes[0, 0], show=False,
           legend_loc='on data', legend_fontsize=10,
           title=f'Leiden clusters (post-Harmony, n={n_clusters_h})',
           frameon=False)

sc.pl.umap(adata, color='condition', ax=axes[0, 1], show=False,
           title='Condition',
           palette={'COVID': '#E24A4A', 'Healthy': '#4A90E2'},
           frameon=False)

sc.pl.umap(adata, color='sample_id', ax=axes[1, 0], show=False,
           title='Sample ID (post-Harmony — should be mixed)',
           frameon=False)

sc.pl.umap(adata, color='weaning_group', ax=axes[1, 1], show=False,
           title='Weaning group',
           palette={'Weaned': '#2CA02C', 'NonWeaned': '#FF7F0E', 'NA': '#AAAAAA'},
           frameon=False)

plt.suptitle(
    'Figure B6 — UMAP overview post-Harmony (25,773 neutrophils × 6 patients)',
    fontsize=14, y=1.00
)
plt.tight_layout()
plt.savefig(FIG / 'B6_UMAP_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: B6_UMAP_overview.png (replaced with Harmony version)')

print('\n' + '='*75)
print('HARMONY INTEGRATION COMPLETE')
print('='*75)
print(f'Next: Phase 3 (HIF1A pathway scoring + DEG + final figures B7-B9)')
