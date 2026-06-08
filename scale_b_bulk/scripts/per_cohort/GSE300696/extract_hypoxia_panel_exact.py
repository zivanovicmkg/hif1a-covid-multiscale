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

# simbol gena = deo posle poslednje "_"
expr["gene_symbol"] = expr["ID"].astype(str).str.rsplit("_", n=1).str[-1]

panel = expr[expr["gene_symbol"].isin(target_symbols)].copy()

print("Selected genes:")
print(panel[["ID", "gene_symbol"]].to_string(index=False))

panel = panel.drop(columns=["gene_symbol"])
panel_t = panel.set_index("ID").T
panel_t.index.name = "expr_sample"
panel_t.reset_index(inplace=True)

merged = pd.merge(meta, panel_t, on="expr_sample")

out = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_panel_exact.tsv"
merged.to_csv(out, sep="\t", index=False)

print("\nSaved:", out)
print("Rows, cols:", merged.shape)
print(merged.head())
