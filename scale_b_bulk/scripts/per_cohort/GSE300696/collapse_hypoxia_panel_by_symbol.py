import pandas as pd

expr_file = "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/GSE300696_expression_matrix_genes.results_TPM.tsv"
sample_map_file = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/sample_map.tsv"

target_symbols = [
    "HIF1A",
    "VEGFA",
    "SLC2A1",
    "LDHA",
    "PDK1",
    "IL6",
    "TNF",
    "CXCL8",
    "STAT3",
    "MMP9",
]

expr = pd.read_csv(expr_file, sep="\t")
meta = pd.read_csv(sample_map_file, sep="\t")

expr2 = expr.copy()
expr2["gene_symbol"] = expr2["ID"].astype(str).str.rsplit("_", n=1).str[-1]
expr2 = expr2[expr2["gene_symbol"].isin(target_symbols)].copy()

sample_cols = [c for c in expr2.columns if c not in ["ID", "gene_symbol"]]

collapsed = expr2.groupby("gene_symbol")[sample_cols].sum()

print("Collapsed genes:")
print(collapsed.index.tolist())

panel_t = collapsed.T
panel_t.index.name = "expr_sample"
panel_t.reset_index(inplace=True)

merged = pd.merge(meta, panel_t, on="expr_sample")

out = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_panel_collapsed.tsv"
merged.to_csv(out, sep="\t", index=False)

print("\nSaved:", out)
print("Shape:", merged.shape)
print(merged.head())
