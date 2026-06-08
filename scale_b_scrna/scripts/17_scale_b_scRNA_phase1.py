"""
===============================================================================
Scale B scRNA-seq — Phase 1: Load, hashtag demultiplex, merge all samples
===============================================================================
Input: 6 raw feature .h5 files from GSE234904 unpacked tar
Metadata: 1 Healthy + 2 COVID weaned + 3 COVID non-weaned

Pipeline:
  1. Load each h5 file
  2. Separate Gene Expression from Antibody Capture (hashtags)
  3. Basic filter: min_genes=200, min_cells=3, max_mt%=20
  4. Annotate each cell with: sample_id, condition, weaning_group
  5. Merge all into single AnnData object
  6. Save intermediate result
===============================================================================
"""
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, dpi_save=300)

# ============================================================================
# CONFIG
# ============================================================================
DATA = Path('/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/GEO_scRNA/GSE234904/unpacked')
OUT = Path('/home/marko-b2/COVID_AI_Project/03_Transcriptomics_GEO/Scale_B_scRNA')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)

# Sample metadata mapping (from series_matrix.txt)
sample_metadata = {
    # GSM ID: (sample_prefix, condition, weaning_group)
    'GSM7476346_2646-3-1-1-1_raw_feature.h5': {
        'sample_id': 'S2646', 'condition': 'Healthy', 'weaning_group': 'NA', 'patient_num': 1},
    'GSM7476348_2706-1-1-1-1_raw_feature.h5': {
        'sample_id': 'S2706', 'condition': 'COVID', 'weaning_group': 'Weaned', 'patient_num': 2},
    'GSM7476349_2708-1-1-1_raw_feature.h5': {
        'sample_id': 'S2708', 'condition': 'COVID', 'weaning_group': 'Weaned', 'patient_num': 3},
    'GSM7476350_2753-1-1-1-1_raw_feature.h5': {
        'sample_id': 'S2753', 'condition': 'COVID', 'weaning_group': 'NonWeaned', 'patient_num': 4},
    'GSM7476351_2765-1-1-1_raw_feature.h5': {
        'sample_id': 'S2765', 'condition': 'COVID', 'weaning_group': 'NonWeaned', 'patient_num': 5},
    'GSM7476352_3116-1-2-2_raw_feature.h5': {
        'sample_id': 'S3116', 'condition': 'COVID', 'weaning_group': 'NonWeaned', 'patient_num': 6},
}

print('='*75)
print('SCALE B scRNA-seq — PHASE 1: LOAD + MERGE')
print('='*75)

# ============================================================================
# Load each sample
# ============================================================================
adatas = []
summary_rows = []

for fname, meta in sample_metadata.items():
    fpath = DATA / fname
    if not fpath.exists():
        print(f'\nWARNING: {fname} not found — skipping')
        continue
    
    print(f'\n[LOADING] {meta["sample_id"]}: {meta["condition"]} / {meta["weaning_group"]}')
    
    # Load with all feature types (GEX + Antibody Capture)
    adata = sc.read_10x_h5(str(fpath), gex_only=False)
    adata.var_names_make_unique()
    print(f'  Raw shape: {adata.shape}')
    print(f'  Feature types: {adata.var["feature_types"].value_counts().to_dict()}')
    
    # Keep only Gene Expression (drop hashtags — not analyzing multiplex since each file = 1 patient)
    gene_mask = adata.var['feature_types'] == 'Gene Expression'
    adata_gex = adata[:, gene_mask].copy()
    print(f'  After GEX filter: {adata_gex.shape}')
    
    # --- QC filter ---
    # Cell filter: min 200 genes
    sc.pp.filter_cells(adata_gex, min_genes=200)
    # Gene filter: genes in at least 3 cells
    sc.pp.filter_genes(adata_gex, min_cells=3)
    
    # Mitochondrial gene %
    adata_gex.var['mt'] = adata_gex.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_gex, qc_vars=['mt'], percent_top=None,
                                log1p=False, inplace=True)
    
    # Filter high MT% cells (>20%)
    n_before = adata_gex.n_obs
    adata_gex = adata_gex[adata_gex.obs['pct_counts_mt'] < 20, :].copy()
    n_after = adata_gex.n_obs
    print(f'  After QC (MT<20%, min_genes=200): {n_after} cells ({n_before - n_after} removed)')
    
    # Annotate
    adata_gex.obs['sample_id'] = meta['sample_id']
    adata_gex.obs['condition'] = meta['condition']
    adata_gex.obs['weaning_group'] = meta['weaning_group']
    adata_gex.obs['patient_num'] = meta['patient_num']
    
    adatas.append(adata_gex)
    summary_rows.append({
        'sample_id': meta['sample_id'],
        'condition': meta['condition'],
        'weaning_group': meta['weaning_group'],
        'n_cells_after_QC': adata_gex.n_obs,
        'n_genes_median': int(adata_gex.obs['n_genes_by_counts'].median()),
        'mt_pct_median': round(adata_gex.obs['pct_counts_mt'].median(), 2),
        'n_counts_median': int(adata_gex.obs['total_counts'].median())
    })

# ============================================================================
# Merge all samples
# ============================================================================
print('\n' + '='*60)
print('MERGING all samples')
print('='*60)
adata_merged = ad.concat(adatas, merge='same', label='batch',
                          keys=[meta['sample_id'] for fname, meta in sample_metadata.items() if (DATA / fname).exists()])
adata_merged.obs_names_make_unique()
adata_merged.var_names_make_unique()

print(f'\nMerged shape: {adata_merged.shape}')
print(f'\nCell composition by condition:')
print(adata_merged.obs.groupby(['condition', 'weaning_group']).size())

print(f'\nCell composition by sample:')
print(adata_merged.obs['sample_id'].value_counts())

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / 'sample_summary.csv', index=False)
print('\nSummary table:')
print(summary_df.to_string(index=False))

# Save merged adata
out_file = OUT / 'merged_raw.h5ad'
adata_merged.write_h5ad(out_file)
print(f'\nSaved: {out_file}')
print(f'Size: {out_file.stat().st_size / (1024**2):.1f} MB')

# Confirm HIF1A pathway genes are present
target_genes = ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PDK1', 'IL6', 'TNF', 'CXCL8', 'STAT3', 'MMP9']
print(f'\nHIF1A pathway genes presence:')
for g in target_genes:
    present = g in adata_merged.var_names
    if present:
        expr_cells = int((adata_merged[:, g].X > 0).sum())
        expr_pct = expr_cells / adata_merged.n_obs * 100
        print(f'  {g}: ✓ detected in {expr_cells} cells ({expr_pct:.1f}%)')
    else:
        print(f'  {g}: ✗ NOT FOUND')

print('\n' + '='*75)
print('PHASE 1 COMPLETE')
print('='*75)
print(f'Output: {out_file}')
print(f'Next: run phase 2 (normalization + UMAP + clustering)')
