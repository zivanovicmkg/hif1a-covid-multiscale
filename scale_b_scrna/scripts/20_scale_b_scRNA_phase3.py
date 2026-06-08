"""
===============================================================================
Scale B scRNA-seq — Phase 3: HIF1A pathway analysis
===============================================================================
Input: merged_harmony.h5ad (25,773 cells × 8,050 genes, Harmony-integrated)

Analyses:
  1. HIF1A pathway score per cell (5 detectable genes)
  2. UMAP overlay of HIF1A score + individual gene UMAPs
  3. Violin plots: HIF1A score by condition, weaning, cluster
  4. Mann-Whitney U tests (clustered on sample_id to respect non-independence)
  5. DEG analysis: NonWeaned vs Weaned (within COVID)
  6. Per-cluster HIF1A signature ranking

Output:
  - B7_HIF1A_UMAP_overlay.png         (main — UMAP colored by score + genes)
  - B8_HIF1A_violin_weaning.png       (main — violin by weaning group)
  - B9_NonWeaned_vs_Weaned_DEG.png    (top DEGs heatmap)
  - B10_cluster_HIF1A_summary.png     (per-cluster ranking)
  - scale_b_scRNA_results.csv          (stats table)
===============================================================================
"""
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
print('SCALE B scRNA-seq — PHASE 3: HIF1A PATHWAY ANALYSIS')
print('='*75)

# ============================================================================
# Load Harmony-integrated data
# ============================================================================
print('\n[STEP 1] Loading merged_harmony.h5ad...')
adata = sc.read_h5ad(OUT / 'merged_harmony.h5ad')
print(f'  Shape: {adata.shape}')
print(f'  Has X_pca_harmony: {"X_pca_harmony" in adata.obsm}')
print(f'  Using leiden_harmony as primary clustering')

# Ensure we are using the harmony-based UMAP (not pre-Harmony)
# UMAP was recomputed on harmony in phase 2b, so adata.obsm['X_umap'] IS harmony-based
# (pre-Harmony is in X_umap_preHarmony)

# ============================================================================
# Define HIF1A pathway score — use only detectable genes
# ============================================================================
all_genes = ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PDK1', 'IL6', 'TNF', 'CXCL8', 'STAT3', 'MMP9']
detectable = [g for g in all_genes if g in adata.var_names]
print(f'\n[STEP 2] HIF1A pathway gene panel:')
for g in all_genes:
    status = '✓' if g in detectable else '✗ (not in dataset)'
    print(f'   {g}: {status}')

print(f'\n  Scoring with {len(detectable)} detectable genes: {detectable}')

# Use scanpy's score_genes function — uses random gene sets as background
sc.tl.score_genes(adata, gene_list=detectable, score_name='HIF1A_score',
                  random_state=42, n_bins=25, ctrl_size=50)
print(f'  HIF1A_score: mean={adata.obs["HIF1A_score"].mean():.3f}, '
      f'std={adata.obs["HIF1A_score"].std():.3f}')

# ============================================================================
# Cluster × HIF1A score summary
# ============================================================================
print('\n[STEP 3] HIF1A score by cluster:')
cluster_summary = adata.obs.groupby('leiden_harmony', observed=True).agg(
    n_cells=('HIF1A_score', 'size'),
    HIF1A_mean=('HIF1A_score', 'mean'),
    HIF1A_median=('HIF1A_score', 'median'),
    HIF1A_std=('HIF1A_score', 'std'),
).round(3).sort_values('HIF1A_mean', ascending=False)
print(cluster_summary.to_string())

# ============================================================================
# HIF1A score by condition and weaning
# ============================================================================
print('\n[STEP 4] HIF1A score by condition × weaning:')
score_summary = adata.obs.groupby(['condition', 'weaning_group'], observed=True).agg(
    n_cells=('HIF1A_score', 'size'),
    HIF1A_mean=('HIF1A_score', 'mean'),
    HIF1A_median=('HIF1A_score', 'median'),
    HIF1A_std=('HIF1A_score', 'std'),
).round(3)
print(score_summary.to_string())

# ============================================================================
# Statistical tests
# ============================================================================
print('\n[STEP 5] Statistical comparisons (Mann-Whitney U)...')

# Test 1: COVID vs Healthy (cell-level; caveat: 5 vs 1 patient)
covid_scores = adata.obs[adata.obs['condition'] == 'COVID']['HIF1A_score'].values
healthy_scores = adata.obs[adata.obs['condition'] == 'Healthy']['HIF1A_score'].values
u1, p1 = stats.mannwhitneyu(covid_scores, healthy_scores, alternative='two-sided')
# Cohen's d
pooled_std = np.sqrt(((len(covid_scores)-1)*covid_scores.std()**2 +
                       (len(healthy_scores)-1)*healthy_scores.std()**2) /
                       (len(covid_scores) + len(healthy_scores) - 2))
d1 = (covid_scores.mean() - healthy_scores.mean()) / pooled_std if pooled_std > 0 else 0
print(f'  COVID vs Healthy:')
print(f'    mean(COVID)={covid_scores.mean():.3f}, mean(Healthy)={healthy_scores.mean():.3f}')
print(f'    Cohen\'s d = {d1:+.3f}, p = {p1:.4e}')
print(f'    CAVEAT: only 1 Healthy patient, many cells from 1 patient — pseudo-replication')

# Test 2: NonWeaned vs Weaned (within COVID)
nw_scores = adata.obs[(adata.obs['condition'] == 'COVID') &
                       (adata.obs['weaning_group'] == 'NonWeaned')]['HIF1A_score'].values
w_scores = adata.obs[(adata.obs['condition'] == 'COVID') &
                      (adata.obs['weaning_group'] == 'Weaned')]['HIF1A_score'].values
u2, p2 = stats.mannwhitneyu(nw_scores, w_scores, alternative='two-sided')
pooled_std = np.sqrt(((len(nw_scores)-1)*nw_scores.std()**2 +
                       (len(w_scores)-1)*w_scores.std()**2) /
                       (len(nw_scores) + len(w_scores) - 2))
d2 = (nw_scores.mean() - w_scores.mean()) / pooled_std if pooled_std > 0 else 0
print(f'\n  NonWeaned vs Weaned (within COVID):')
print(f'    mean(NonWeaned)={nw_scores.mean():.3f}, mean(Weaned)={w_scores.mean():.3f}')
print(f'    Cohen\'s d = {d2:+.3f}, p = {p2:.4e}')

# Test 3: Per-sample means (proper test at patient level)
print(f'\n  Per-sample HIF1A means (patient-level):')
per_sample = adata.obs.groupby('sample_id', observed=True).agg(
    condition=('condition', 'first'),
    weaning_group=('weaning_group', 'first'),
    HIF1A_mean=('HIF1A_score', 'mean'),
    n_cells=('HIF1A_score', 'size')
).round(3).sort_values('HIF1A_mean', ascending=False)
print(per_sample.to_string())

# ============================================================================
# FIGURE B7: UMAP overlay — HIF1A score + individual genes
# ============================================================================
print('\n[STEP 6] Figure B7 — HIF1A UMAP overlays...')

# Panel: HIF1A score + each detectable gene on UMAP
n_panels = len(detectable) + 1
ncols = 3
nrows = (n_panels + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

# Panel 1: composite score
sc.pl.umap(adata, color='HIF1A_score', ax=axes[0], show=False,
           cmap='RdYlBu_r', title='HIF1A pathway score\n(composite, 5 genes)',
           frameon=False, colorbar_loc='right')

# Panel 2..: individual genes
for i, gene in enumerate(detectable, start=1):
    sc.pl.umap(adata, color=gene, ax=axes[i], show=False,
               cmap='Reds', title=gene, frameon=False, colorbar_loc='right')

# Hide unused axes
for j in range(n_panels, len(axes)):
    axes[j].axis('off')

plt.suptitle('Figure B7 — HIF1A pathway gene expression on UMAP (post-Harmony)',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIG / 'B7_HIF1A_UMAP_overlay.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: B7_HIF1A_UMAP_overlay.png')

# ============================================================================
# FIGURE B8: Violin plots — HIF1A score by condition/weaning/cluster
# ============================================================================
print('\n[STEP 7] Figure B8 — HIF1A score violin plots...')

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

# Panel 1: by condition
ax1 = fig.add_subplot(gs[0, 0])
df_plot = adata.obs[['condition', 'HIF1A_score']].copy()
sns.violinplot(data=df_plot, x='condition', y='HIF1A_score', ax=ax1,
               palette={'COVID': '#E24A4A', 'Healthy': '#4A90E2'}, inner='quartile')
ax1.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax1.set_title(f'By condition (Mann-Whitney)\nd={d1:+.2f}, p={p1:.2e}', fontsize=11)
ax1.set_xlabel('')

# Panel 2: by weaning group (only COVID)
ax2 = fig.add_subplot(gs[0, 1])
df_plot = adata.obs[adata.obs['condition'] == 'COVID'][['weaning_group', 'HIF1A_score']].copy()
df_plot['weaning_group'] = pd.Categorical(df_plot['weaning_group'], categories=['Weaned', 'NonWeaned'])
sns.violinplot(data=df_plot, x='weaning_group', y='HIF1A_score', ax=ax2,
               palette={'Weaned': '#2CA02C', 'NonWeaned': '#FF7F0E'}, inner='quartile')
ax2.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax2.set_title(f'By weaning (COVID only)\nd={d2:+.2f}, p={p2:.2e}', fontsize=11)
ax2.set_xlabel('')

# Panel 3: per sample
ax3 = fig.add_subplot(gs[0, 2])
df_plot = adata.obs[['sample_id', 'HIF1A_score', 'weaning_group']].copy()
sample_order = per_sample.index.tolist()
sample_palette = {}
for s in sample_order:
    w = per_sample.loc[s, 'weaning_group']
    sample_palette[s] = '#2CA02C' if w == 'Weaned' else ('#FF7F0E' if w == 'NonWeaned' else '#4A90E2')
sns.violinplot(data=df_plot, x='sample_id', y='HIF1A_score', ax=ax3,
               order=sample_order, palette=sample_palette, inner='quartile')
ax3.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax3.set_title('Per patient (ordered by mean)', fontsize=11)
ax3.set_xlabel('')
for tick in ax3.get_xticklabels():
    tick.set_rotation(45)

# Panel 4: by cluster
ax4 = fig.add_subplot(gs[1, :2])
df_plot = adata.obs[['leiden_harmony', 'HIF1A_score']].copy()
cluster_order = cluster_summary.index.tolist()
sns.violinplot(data=df_plot, x='leiden_harmony', y='HIF1A_score', ax=ax4,
               order=cluster_order, palette='RdYlBu_r', inner='quartile')
ax4.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax4.set_title('By Leiden cluster (ordered by HIF1A mean, high → low)', fontsize=11)
ax4.set_xlabel('Leiden cluster')

# Panel 5: cluster × weaning heatmap
ax5 = fig.add_subplot(gs[1, 2])
pivot = adata.obs.pivot_table(index='leiden_harmony', columns='weaning_group',
                                values='HIF1A_score', aggfunc='mean')
pivot = pivot.reindex(cluster_order)
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            cbar_kws={'label': 'HIF1A score'}, ax=ax5, linewidths=0.5,
            annot_kws={'fontsize': 9})
ax5.set_title('HIF1A mean by cluster × weaning', fontsize=11)
ax5.set_xlabel('')

plt.suptitle('Figure B8 — HIF1A pathway score distributions',
             fontsize=14, y=1.00)
plt.savefig(FIG / 'B8_HIF1A_violin_weaning.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: B8_HIF1A_violin_weaning.png')

# ============================================================================
# FIGURE B9: DEG NonWeaned vs Weaned (within COVID)
# ============================================================================
print('\n[STEP 8] Figure B9 — DEG NonWeaned vs Weaned (within COVID)...')

# Subset to COVID only for DEG
covid_adata = adata[adata.obs['condition'] == 'COVID'].copy()
print(f'  COVID subset: {covid_adata.shape}')

sc.tl.rank_genes_groups(covid_adata, 'weaning_group', method='wilcoxon',
                         groups=['NonWeaned', 'Weaned'], reference='Weaned',
                         n_genes=30, key_added='deg_weaning')

# Extract top genes
deg_df = pd.DataFrame({
    'gene': covid_adata.uns['deg_weaning']['names']['NonWeaned'],
    'logfc': covid_adata.uns['deg_weaning']['logfoldchanges']['NonWeaned'],
    'pval': covid_adata.uns['deg_weaning']['pvals']['NonWeaned'],
    'pval_adj': covid_adata.uns['deg_weaning']['pvals_adj']['NonWeaned'],
})
print('\n  Top 20 upregulated in NonWeaned (vs Weaned):')
print(deg_df.head(20).to_string(index=False))

deg_df.to_csv(OUT / 'DEG_NonWeaned_vs_Weaned.csv', index=False)

# Rank genes plot (upregulated in NonWeaned)
sc.pl.rank_genes_groups(covid_adata, n_genes=20, key='deg_weaning',
                         save='_B9_DEG_barplot.png', show=False)
print('  Saved: figures/rank_genes_groups_*_B9_DEG_barplot.png')

# Heatmap of top DEGs per weaning group on the COVID subset
top_genes = list(deg_df.head(15)['gene'])
print(f'  Top 15 DEGs for heatmap: {top_genes}')

# Check gene presence
present = [g for g in top_genes if g in covid_adata.var_names]
if len(present) > 0:
    sc.pl.heatmap(covid_adata, var_names=present, groupby='weaning_group',
                   standard_scale='var', cmap='RdBu_r', dendrogram=False,
                   save='_B9_DEG_heatmap.png', show=False, swap_axes=False,
                   figsize=(8, 6))
    print('  Saved: figures/heatmap_B9_DEG_heatmap.png')

# ============================================================================
# FIGURE B10: Per-cluster HIF1A signature summary
# ============================================================================
print('\n[STEP 9] Figure B10 — Per-cluster HIF1A summary + composition...')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: HIF1A score per cluster (sorted)
ax = axes[0, 0]
cluster_order_high_low = cluster_summary.sort_values('HIF1A_mean', ascending=False).index.tolist()
means = cluster_summary.loc[cluster_order_high_low, 'HIF1A_mean'].values
stds = cluster_summary.loc[cluster_order_high_low, 'HIF1A_std'].values
colors = plt.cm.RdYlBu_r(np.linspace(0.85, 0.15, len(cluster_order_high_low)))
bars = ax.bar(range(len(cluster_order_high_low)), means, yerr=stds/2,
               color=colors, edgecolor='black', linewidth=0.6, capsize=4)
ax.set_xticks(range(len(cluster_order_high_low)))
ax.set_xticklabels(cluster_order_high_low)
ax.set_xlabel('Cluster (ordered by HIF1A mean)')
ax.set_ylabel('HIF1A pathway score')
ax.set_title('HIF1A score per cluster (±½ SD)')
ax.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)

# Top-right: Cluster composition by weaning (stacked bar)
ax = axes[0, 1]
pivot = pd.crosstab(adata.obs['leiden_harmony'], adata.obs['weaning_group'], normalize='index') * 100
pivot = pivot.reindex(cluster_order_high_low)
pivot[['Weaned', 'NonWeaned', 'NA']].plot(kind='bar', stacked=True, ax=ax,
                                             color=['#2CA02C', '#FF7F0E', '#AAAAAA'],
                                             edgecolor='black', linewidth=0.6, width=0.8)
ax.set_xlabel('Cluster (ordered by HIF1A mean)')
ax.set_ylabel('% cells')
ax.set_title('Weaning group composition per cluster')
ax.legend(loc='upper right', fontsize=9)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)

# Bottom-left: Cluster size (n cells)
ax = axes[1, 0]
ax.bar(range(len(cluster_order_high_low)),
        cluster_summary.loc[cluster_order_high_low, 'n_cells'].values,
        color='#888888', edgecolor='black', linewidth=0.6)
ax.set_xticks(range(len(cluster_order_high_low)))
ax.set_xticklabels(cluster_order_high_low)
ax.set_xlabel('Cluster (ordered by HIF1A mean)')
ax.set_ylabel('Number of cells')
ax.set_title('Cluster size')

# Bottom-right: Per-gene dotplot in clusters
ax = axes[1, 1]
ax.axis('off')
# Save separate dotplot
ax.text(0.5, 0.5, 'See separate dotplot:\nB10_HIF1A_dotplot.png',
        ha='center', va='center', fontsize=11, style='italic',
        transform=ax.transAxes)

plt.suptitle('Figure B10 — Per-cluster HIF1A signature summary',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIG / 'B10_cluster_HIF1A_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: B10_cluster_HIF1A_summary.png')

# Dotplot of HIF1A panel genes by cluster
sc.pl.dotplot(adata, var_names=detectable, groupby='leiden_harmony',
              standard_scale='var', cmap='Reds', dendrogram=False,
              save='_B10_HIF1A_dotplot.png', show=False,
              figsize=(8, 5))
print('  Saved: figures/dotplot_B10_HIF1A_dotplot.png')

# ============================================================================
# Results CSV
# ============================================================================
results_rows = [
    ['Test', 'mean_group1', 'mean_group2', 'Cohen_d', 'p_value', 'note'],
    ['COVID vs Healthy (cell-level)',
      f'{covid_scores.mean():.3f}', f'{healthy_scores.mean():.3f}',
      f'{d1:+.3f}', f'{p1:.3e}', 'n_COVID=22766, n_Healthy=3007 (1 patient only)'],
    ['NonWeaned vs Weaned (within COVID)',
      f'{nw_scores.mean():.3f}', f'{w_scores.mean():.3f}',
      f'{d2:+.3f}', f'{p2:.3e}', 'n_NW=14908 (3 pts), n_W=7858 (2 pts)'],
]
with open(OUT / 'scale_b_scRNA_results.csv', 'w') as f:
    for row in results_rows:
        f.write(','.join(row) + '\n')
print(f'\nResults saved to: {OUT / "scale_b_scRNA_results.csv"}')

# Save adata with score
adata.write_h5ad(OUT / 'merged_harmony_scored.h5ad')
print(f'Saved: {OUT / "merged_harmony_scored.h5ad"}')

# ============================================================================
# DONE
# ============================================================================
print('\n' + '='*75)
print('PHASE 3 COMPLETE — Scale B scRNA-seq analysis done')
print('='*75)
print('\nKey findings:')
print(f'  1. HIF1A pathway score: 5 genes (HIF1A, LDHA, STAT3, MMP9, PDK1)')
print(f'  2. NonWeaned vs Weaned: d={d2:+.2f}, p={p2:.2e}')
print(f'  3. Top cluster by HIF1A: cluster {cluster_summary.index[0]} '
      f'(mean={cluster_summary.iloc[0]["HIF1A_mean"]:.3f})')
print(f'  4. Figures: B7, B8, B9, B10 + auxiliary DEG plots')
print(f'\nNext: Scale B docx report generation')
