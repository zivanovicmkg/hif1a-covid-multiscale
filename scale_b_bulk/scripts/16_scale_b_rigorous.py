"""
===============================================================================
Scale B — Rigorous within-cohort analyses + meta-analysis
===============================================================================
Problem identified (23.04.2026):
    The previous "COVID vs Control" analysis pooled 5 GEO cohorts with:
    (a) catastrophic raw-expression scale differences (1000x between cohorts)
    (b) perfect confounding — "Control" group existed only in GSE152075
    (c) AUC 0.87 was likely predicting cohort identity, not disease state

Solution: Within-cohort analyses using z-score normalized expression
    (per-cohort standardization already applied in master matrix).

Five within-cohort comparisons:
    1. GSE152075: COVID vs Control (N=484)
    2. GSE157103: ICU vs NonICU (N=123)
    3. GSE171110: Severe vs Healthy (N=54)
    4. GSE212861: Covid19_SDRA vs Covid19 (N=93)
    5. GSE300696: Hospitalized vs Convalescent (N=346)

Meta-analysis: per-gene effect sizes (Cohen's d) across cohorts, forest plot.

Output (in /home/marko-b2/COVID_AI_Project/03_Transcriptomics_GEO/Scale_B_rigorous/):
    - per_cohort_auc_table.csv
    - per_cohort_gene_stats.csv
    - meta_analysis_effect_sizes.csv
    - figures/B1_per_cohort_AUCs.png
    - figures/B2_meta_analysis_forest.png
    - figures/B3_gene_direction_heatmap.png
    - figures/B4_panel_score_by_cohort.png
    - figures/B5_consistency_matrix.png
    - scale_b_rigorous_log.txt
===============================================================================
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

# ============================================================================
DATA = Path('/home/marko-b2/COVID_AI_Project/03_Transcriptomics_GEO/Multi_cohort_analysis/processed/master_feature_matrix.tsv')
OUT = Path('/home/marko-b2/COVID_AI_Project/03_Transcriptomics_GEO/Scale_B_rigorous')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE B — RIGOROUS WITHIN-COHORT ANALYSES + META-ANALYSIS')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load data
# ============================================================================
df = pd.read_csv(DATA, sep='\t')
log(f'\n[STEP 1] Loaded N={len(df)} samples, {df["dataset"].nunique()} cohorts')

# Gene panel (z-score normalized, per-cohort)
genes = ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PDK1',
         'IL6', 'TNF', 'CXCL8', 'STAT3', 'MMP9']
genes_z = [f'{g}_z' for g in genes]

# ============================================================================
# STEP 2: Define 5 within-cohort comparisons
# ============================================================================
comparisons = [
    {'cohort': 'GSE152075', 'name': 'COVID vs Control',
     'pos': 'COVID', 'neg': 'Control',
     'note': 'SARS-CoV-2+ hospital patients vs healthcare workers'},
    {'cohort': 'GSE157103', 'name': 'ICU vs NonICU',
     'pos': 'ICU', 'neg': 'NonICU',
     'note': 'Hospitalized COVID — severity comparison'},
    {'cohort': 'GSE171110', 'name': 'Severe vs Healthy',
     'pos': 'Severe', 'neg': 'Healthy',
     'note': 'Severe COVID-19 vs healthy donors (small N)'},
    {'cohort': 'GSE212861', 'name': 'Covid19_SDRA vs Covid19',
     'pos': 'Covid19_SDRA', 'neg': 'Covid19',
     'note': 'ARDS-complicated COVID vs standard COVID'},
    {'cohort': 'GSE300696', 'name': 'Hospitalized vs Convalescent',
     'pos': 'Hospitalized', 'neg': 'Convalescent',
     'note': 'Acute hospitalized vs post-COVID recovered'},
]

# ============================================================================
# STEP 3: Per-cohort analysis loop
# ============================================================================
log('\n[STEP 2] Running 5 within-cohort analyses...\n')

all_auc_rows = []
all_gene_stats_rows = []

for comp in comparisons:
    log(f'='*60)
    log(f'COHORT: {comp["cohort"]}  |  {comp["name"]}')
    log(f'  Context: {comp["note"]}')
    log('='*60)
    
    sub = df[df['dataset'] == comp['cohort']].copy()
    sub = sub[sub['severity'].isin([comp['pos'], comp['neg']])].copy()
    sub['label'] = (sub['severity'] == comp['pos']).astype(int)
    
    n_pos = (sub['label'] == 1).sum()
    n_neg = (sub['label'] == 0).sum()
    log(f'  N positive ({comp["pos"]}): {n_pos}')
    log(f'  N negative ({comp["neg"]}): {n_neg}')
    
    X = sub[genes_z].values
    y = sub['label'].values
    
    # ---- Per-gene univariate statistics ----
    log(f'\n  Per-gene Mann-Whitney U + Cohen\'s d:')
    for g in genes:
        vals_pos = sub[sub['label'] == 1][f'{g}_z'].values
        vals_neg = sub[sub['label'] == 0][f'{g}_z'].values
        try:
            # Mann-Whitney U
            u_stat, p_val = stats.mannwhitneyu(vals_pos, vals_neg, alternative='two-sided')
            # Cohen's d (standardized mean difference)
            pooled_std = np.sqrt(((len(vals_pos)-1)*vals_pos.std()**2 +
                                   (len(vals_neg)-1)*vals_neg.std()**2) /
                                   (len(vals_pos) + len(vals_neg) - 2))
            if pooled_std > 0:
                cohens_d = (vals_pos.mean() - vals_neg.mean()) / pooled_std
            else:
                cohens_d = 0.0
            # AUC per gene
            try:
                single_auc = roc_auc_score(y, sub[f'{g}_z'].values)
            except:
                single_auc = 0.5
        except Exception as e:
            p_val, cohens_d, single_auc = np.nan, np.nan, np.nan
        
        all_gene_stats_rows.append({
            'cohort': comp['cohort'], 'comparison': comp['name'],
            'gene': g,
            'mean_pos': round(vals_pos.mean(), 3),
            'mean_neg': round(vals_neg.mean(), 3),
            'cohens_d': round(cohens_d, 3),
            'p_value': round(p_val, 4) if not np.isnan(p_val) else 'NA',
            'univariate_AUC': round(single_auc, 3)
        })
        
        direction = '↑' if cohens_d > 0 else '↓'
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        log(f'    {direction} {g:8s} d={cohens_d:+.2f}  p={p_val:.4f} {sig}  AUC={single_auc:.3f}')
    
    # ---- Multivariate LR (10 genes combined) with 5-fold CV ----
    if n_pos >= 5 and n_neg >= 5:
        skf = StratifiedKFold(n_splits=min(5, min(n_pos, n_neg)), shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
        try:
            cv_aucs = cross_val_score(lr, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            log(f'\n  Multivariate LR 10-gene panel CV AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}')
            panel_auc = cv_aucs.mean()
            panel_std = cv_aucs.std()
        except Exception as e:
            log(f'\n  Multivariate LR failed: {e}')
            panel_auc = np.nan
            panel_std = np.nan
    else:
        panel_auc = np.nan
        panel_std = np.nan
    
    # ---- Panel score (pre-computed) AUC ----
    try:
        panel_score_auc = roc_auc_score(y, sub['panel_score_z'].values)
        log(f'  Pre-computed panel_score_z AUC: {panel_score_auc:.3f}')
    except:
        panel_score_auc = np.nan
    
    all_auc_rows.append({
        'cohort': comp['cohort'], 'comparison': comp['name'],
        'positive_class': comp['pos'], 'negative_class': comp['neg'],
        'N_positive': n_pos, 'N_negative': n_neg,
        'LR_CV_AUC': round(panel_auc, 3) if not np.isnan(panel_auc) else 'NA',
        'LR_CV_std': round(panel_std, 3) if not np.isnan(panel_std) else 'NA',
        'panel_score_AUC': round(panel_score_auc, 3) if not np.isnan(panel_score_auc) else 'NA',
        'note': comp['note']
    })
    log('')

# Save tables
auc_df = pd.DataFrame(all_auc_rows)
gene_stats_df = pd.DataFrame(all_gene_stats_rows)
auc_df.to_csv(OUT / 'per_cohort_auc_table.csv', index=False)
gene_stats_df.to_csv(OUT / 'per_cohort_gene_stats.csv', index=False)

log('\n=== PER-COHORT AUC SUMMARY ===')
log(auc_df[['cohort', 'comparison', 'N_positive', 'N_negative', 'LR_CV_AUC', 'panel_score_AUC']].to_string(index=False))

# ============================================================================
# STEP 4: Meta-analysis — effect sizes across cohorts
# ============================================================================
log('\n[STEP 3] Meta-analysis — effect sizes per gene across 5 cohorts...')

# Pivot to: gene × cohort matrix of Cohen's d
meta = gene_stats_df.pivot(index='gene', columns='cohort', values='cohens_d')
meta = meta.reindex(genes)  # preserve gene order
log('\nCohen\'s d per gene × cohort:')
log(meta.round(2).to_string())

# Directional consistency: count cohorts where d > 0 or < 0
consistency = pd.DataFrame({
    'gene': genes,
    'n_cohorts_up': [(meta.loc[g] > 0).sum() for g in genes],
    'n_cohorts_down': [(meta.loc[g] < 0).sum() for g in genes],
    'mean_d': [meta.loc[g].mean() for g in genes],
    'median_d': [meta.loc[g].median() for g in genes],
})
consistency['dominant_direction'] = consistency.apply(
    lambda r: '↑' if r['n_cohorts_up'] > r['n_cohorts_down']
              else ('↓' if r['n_cohorts_down'] > r['n_cohorts_up'] else '='),
    axis=1
)
consistency.to_csv(OUT / 'meta_analysis_effect_sizes.csv', index=False)
log('\nDirectional consistency:')
log(consistency.to_string(index=False))

# ============================================================================
# STEP 5: Figure B1 — Per-cohort AUC bar chart
# ============================================================================
log('\n[STEP 4] Figure B1 — per-cohort AUCs...')
fig, ax = plt.subplots(figsize=(12, 6))

plot_df = auc_df.copy()
# Convert NA strings to NaN for plotting
plot_df['LR_CV_AUC_plot'] = pd.to_numeric(plot_df['LR_CV_AUC'], errors='coerce')
plot_df['panel_score_AUC_plot'] = pd.to_numeric(plot_df['panel_score_AUC'], errors='coerce')

x = np.arange(len(plot_df))
w = 0.35

bars1 = ax.bar(x - w/2, plot_df['LR_CV_AUC_plot'], w,
               color='#1f77b4', label='Multivariate LR (10-gene CV)', edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + w/2, plot_df['panel_score_AUC_plot'], w,
               color='#E24A4A', label='Panel score (z-averaged)', edgecolor='black', linewidth=0.6)

ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6, label='Chance')
labels = [f"{r['cohort']}\n{r['comparison']}\n(N={r['N_positive']}+{r['N_negative']})"
           for _, r in plot_df.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('Figure B1 — Within-cohort AUC for HIF1A gene panel', fontsize=12)
ax.set_ylim(0.40, 1.0)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Annotate values
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f'{h:.2f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / 'figures' / 'B1_per_cohort_AUCs.png', dpi=300)
plt.close()
log('  Saved: B1_per_cohort_AUCs.png')

# ============================================================================
# STEP 6: Figure B2 — Meta-analysis forest plot
# ============================================================================
log('\n[STEP 5] Figure B2 — Meta-analysis forest plot...')

# For each gene, plot d across 5 cohorts + mean line
fig, ax = plt.subplots(figsize=(14, 10))

cohort_colors = {
    'GSE152075': '#1f77b4',
    'GSE157103': '#2ca02c',
    'GSE171110': '#E24A4A',
    'GSE212861': '#ff7f0e',
    'GSE300696': '#9467bd',
}
cohort_order = list(cohort_colors.keys())

for i, gene in enumerate(genes):
    d_values = []
    for j, cohort in enumerate(cohort_order):
        d = meta.loc[gene, cohort] if cohort in meta.columns else np.nan
        if not np.isnan(d):
            d_values.append(d)
            y_pos = i + (j - 2) * 0.12
            ax.scatter(d, y_pos, s=100, color=cohort_colors[cohort],
                         edgecolor='black', zorder=3)
    # Mean across cohorts
    if len(d_values) > 0:
        mean_d = np.mean(d_values)
        ax.plot([mean_d]*2, [i - 0.35, i + 0.35], 'k-', lw=2.5, zorder=4)

# Legend for cohorts
legend_patches = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=c, markersize=10,
                               markeredgecolor='black', label=n)
                    for n, c in cohort_colors.items()]
legend_patches.append(plt.Line2D([0], [0], color='k', lw=2.5, label='Mean across cohorts'))
ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

ax.axvline(0, color='gray', linestyle='--', lw=1)
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("Cohen's d (positive class vs negative class, z-score scale)", fontsize=11)
ax.set_title("Figure B2 — Meta-analysis: per-gene effect sizes across 5 cohorts",
             fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Shade areas
x_min, x_max = ax.get_xlim()
ax.axvspan(0, x_max, alpha=0.05, color='red', label='↑ in positive class')
ax.axvspan(x_min, 0, alpha=0.05, color='blue', label='↓ in positive class')

plt.tight_layout()
plt.savefig(OUT / 'figures' / 'B2_meta_analysis_forest.png', dpi=300)
plt.close()
log('  Saved: B2_meta_analysis_forest.png')

# ============================================================================
# STEP 7: Figure B3 — Heatmap of effect sizes (direction visualization)
# ============================================================================
log('\n[STEP 6] Figure B3 — Direction heatmap...')
fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(meta, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            cbar_kws={'label': "Cohen's d"}, ax=ax, linewidths=0.5,
            vmin=-1.5, vmax=1.5, annot_kws={'fontsize': 10})
ax.set_title("Figure B3 — Gene effect sizes (Cohen's d) by cohort\n" +
             "Red = higher in positive class, Blue = higher in negative class",
             fontsize=11)
ax.set_xlabel('')
ax.set_ylabel('')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig(OUT / 'figures' / 'B3_gene_direction_heatmap.png', dpi=300)
plt.close()
log('  Saved: B3_gene_direction_heatmap.png')

# ============================================================================
# STEP 8: Figure B4 — Panel score distribution per cohort (boxplot)
# ============================================================================
log('\n[STEP 7] Figure B4 — panel_score_z by severity (all cohorts)...')
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

for ax, comp in zip(axes, comparisons):
    sub = df[df['dataset'] == comp['cohort']].copy()
    sub = sub[sub['severity'].isin([comp['pos'], comp['neg']])].copy()
    # Preserve order: negative then positive
    sub['severity'] = pd.Categorical(sub['severity'],
                                       categories=[comp['neg'], comp['pos']],
                                       ordered=True)
    sns.boxplot(data=sub, x='severity', y='panel_score_z', ax=ax,
                palette={comp['neg']: '#4A90E2', comp['pos']: '#E24A4A'},
                showfliers=True)
    sns.stripplot(data=sub, x='severity', y='panel_score_z', ax=ax,
                   color='black', size=2, alpha=0.4, jitter=0.2)
    
    # Compute p-value
    pos_vals = sub[sub['severity'] == comp['pos']]['panel_score_z'].values
    neg_vals = sub[sub['severity'] == comp['neg']]['panel_score_z'].values
    try:
        _, p_val = stats.mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        p_str = f'p={p_val:.3g}' if p_val >= 0.001 else 'p<0.001'
    except:
        p_str = 'p=NA'
    
    ax.set_title(f'{comp["cohort"]}\n{comp["name"]}\n{p_str}', fontsize=10)
    ax.set_xlabel('')
    ax.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)

axes[0].set_ylabel('panel_score_z', fontsize=11)
plt.suptitle('Figure B4 — HIF1A panel score by cohort (z-normalized)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'figures' / 'B4_panel_score_by_cohort.png', dpi=300)
plt.close()
log('  Saved: B4_panel_score_by_cohort.png')

# ============================================================================
# STEP 9: Figure B5 — Consistency matrix
# ============================================================================
log('\n[STEP 8] Figure B5 — Directional consistency matrix...')
fig, ax = plt.subplots(figsize=(10, 7))

# Colored matrix: +1 (up), -1 (down), 0 (near 0)
direction_mat = np.sign(meta.values)
# Threshold small d to 0
direction_mat = np.where(np.abs(meta.values) < 0.1, 0, direction_mat)

sns.heatmap(direction_mat, annot=meta.round(2).values, fmt='',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Direction (sign of Cohen\'s d)'},
            ax=ax, linewidths=1, xticklabels=meta.columns, yticklabels=meta.index,
            annot_kws={'fontsize': 11})
ax.set_title('Figure B5 — Direction of gene effect by cohort\n' +
             'Red (+1) = gene ↑ in case; Blue (−1) = gene ↓ in case; White (0) = |d|<0.1',
             fontsize=11)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig(OUT / 'figures' / 'B5_consistency_matrix.png', dpi=300)
plt.close()
log('  Saved: B5_consistency_matrix.png')

# ============================================================================
# DONE
# ============================================================================
log('\n' + '='*75)
log('SCALE B RIGOROUS — COMPLETE')
log(f'Finished: {datetime.now().isoformat()}')
log('='*75)

log('\n=== KEY FINDINGS ===')
log('\n1. Per-cohort AUC performance (10-gene LR panel):')
for _, r in auc_df.iterrows():
    log(f'   {r["cohort"]} ({r["comparison"]}, N={r["N_positive"]}+{r["N_negative"]}): AUC = {r["LR_CV_AUC"]}')

log('\n2. Genes with CONSISTENT direction across cohorts:')
consistent = consistency[(consistency['n_cohorts_up'] >= 4) | (consistency['n_cohorts_down'] >= 4)]
if len(consistent) > 0:
    for _, r in consistent.iterrows():
        log(f'   {r["dominant_direction"]} {r["gene"]:8s}: up in {r["n_cohorts_up"]}/{5}, down in {r["n_cohorts_down"]}/{5} cohorts, mean d={r["mean_d"]:+.2f}')
else:
    log('   No gene shows consistent direction across ≥4 of 5 cohorts')

log('\n3. Genes with HETEROGENEOUS direction (possible heterogeneity):')
heterog = consistency[(consistency['n_cohorts_up'] >= 2) & (consistency['n_cohorts_down'] >= 2)]
for _, r in heterog.iterrows():
    log(f'   ∿ {r["gene"]:8s}: up in {r["n_cohorts_up"]}, down in {r["n_cohorts_down"]}, mean d={r["mean_d"]:+.2f}')

with open(OUT / 'scale_b_rigorous_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
