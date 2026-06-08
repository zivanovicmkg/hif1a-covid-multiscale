"""
===============================================================================
Scale B scRNA-seq — Phase 2: Normalization, HVG, PCA, UMAP, Clustering
===============================================================================
Input: merged_raw.h5ad (25,773 cells × 8,050 genes, 6 samples)
Output: merged_processed.h5ad + QC/UMAP figures
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
FIG.mkdir(exist_ok=True)
sc.settings.figdir = str(FIG)

print('='*75)
print('SCALE B scRNA-seq — PHASE 2: NORMALIZATION + UMAP + CLUSTERING')
print('='*75)

# ============================================================================
# Load merged data
# ============================================================================
print('\n[STEP 1] Loading merged AnnData...')
adata = sc.read_h5ad(OUT / 'merged_raw.h5ad')
print(f'  Loaded: {adata.shape}')
print(f'  Conditions: {adata.obs["condition"].value_counts().to_dict()}')

# ============================================================================
# QC plot BEFORE normalization
# ============================================================================
print('\n[STEP 2] QC violin plots pre-normalization...')
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
              jitter=0.4, groupby='sample_id', rotation=45,
              save='_B6_QC_by_sample.png', show=False)
print('  Saved: figures/violin_B6_QC_by_sample.png')

# ============================================================================
# Store raw, then normalize
# ============================================================================
print('\n[STEP 3] Normalization...')
# Keep raw counts
adata.layers['counts'] = adata.X.copy()

# Normalize: target_sum=10000 (standard), then log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Store normalized version in 'raw' for later use (scanpy convention)
adata.raw = adata

print(f'  Normalized and log1p transformed')
print(f'  X is now log(norm+1), counts stored in layer "counts"')

# ============================================================================
# Highly variable genes
# ============================================================================
print('\n[STEP 4] Highly variable gene selection...')
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000,
                              batch_key='sample_id')
n_hvg = adata.var['highly_variable'].sum()
print(f'  Selected {n_hvg} highly variable genes (across samples)')

# Plot HVG
sc.pl.highly_variable_genes(adata, save='_B7_HVG.png', show=False)
print('  Saved: figures/filter_genes_dispersion_B7_HVG.png')

# ============================================================================
# PCA
# ============================================================================
print('\n[STEP 5] PCA...')
# Scale (clip to 10 to limit outliers)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, use_highly_variable=True, random_state=42)

# Plot variance explained
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True,
                           save='_B8_PCA_variance.png', show=False)
print('  Saved: figures/pca_variance_ratio_B8_PCA_variance.png')

# ============================================================================
# Neighbors + UMAP
# ============================================================================
print('\n[STEP 6] kNN graph + UMAP...')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=42)
sc.tl.umap(adata, random_state=42)
print('  UMAP computed')

# ============================================================================
# Leiden clustering
# ============================================================================
print('\n[STEP 7] Leiden clustering...')
sc.tl.leiden(adata, resolution=0.5, random_state=42)
n_clusters = adata.obs['leiden'].nunique()
print(f'  {n_clusters} Leiden clusters identified (resolution=0.5)')
print(f'  Cluster sizes:')
print(adata.obs['leiden'].value_counts().sort_index().to_string())

# ============================================================================
# UMAP visualizations
# ============================================================================
print('\n[STEP 8] UMAP plots — cell annotations...')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: by cluster
sc.pl.umap(adata, color='leiden', ax=axes[0, 0], show=False,
             legend_loc='on data', legend_fontsize=10,
             title=f'Leiden clusters (n={n_clusters})')

# Panel 2: by condition
sc.pl.umap(adata, color='condition', ax=axes[0, 1], show=False,
             title='Condition')

# Panel 3: by sample
sc.pl.umap(adata, color='sample_id', ax=axes[1, 0], show=False,
             title='Sample ID (batch check)')

# Panel 4: by weaning
sc.pl.umap(adata, color='weaning_group', ax=axes[1, 1], show=False,
             title='Weaning group')

plt.suptitle('Figure B6 — UMAP overview (25,773 neutrophils × 6 patients)',
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(FIG / 'B6_UMAP_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: B6_UMAP_overview.png')

# ============================================================================
# UMAP plot of QC metrics
# ============================================================================
print('\n[STEP 9] UMAP colored by QC metrics...')
sc.pl.umap(adata, color=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             save='_B7_QC_overlay.png', show=False, ncols=3)
print('  Saved: figures/umap_B7_QC_overlay.png')

# ============================================================================
# Save
# ============================================================================
out_file = OUT / 'merged_processed.h5ad'
adata.write_h5ad(out_file)
print(f'\nSaved: {out_file}')
print(f'Size: {out_file.stat().st_size / (1024**2):.1f} MB')

# Summary: composition per cluster
print('\n=== Cluster composition by condition ===')
ct = pd.crosstab(adata.obs['leiden'], adata.obs['condition'])
print(ct.to_string())
print('\n=== Cluster composition by sample (batch effect indicator) ===')
ct2 = pd.crosstab(adata.obs['leiden'], adata.obs['sample_id'])
print(ct2.to_string())

print('\n' + '='*75)
print('PHASE 2 COMPLETE')
print('='*75)
print(f'Output: {out_file}')
print(f'Figures: {FIG}/B6_UMAP_overview.png + aux QC figures')
print(f'Next: run phase 3 (HIF1A pathway scoring + DEG + final figures)')
