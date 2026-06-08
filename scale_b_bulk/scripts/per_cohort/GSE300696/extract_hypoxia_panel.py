import pandas as pd

expr_file = "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/GSE300696_expression_matrix_genes.results_TPM.tsv"
sample_map_file = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/sample_map.tsv"

genes_of_interest = [
    "HIF1A",
    "VEGFA",
    "SLC2A1",
    "LDHA",
    "PDK1",
    "IL6",
    "TNF",
    "CXCL8",
    "STAT3",
    "MMP9"
]

expr = pd.read_csv(expr_file, sep="\t")
meta = pd.read_csv(sample_map_file, sep="\t")

matches = expr[expr["ID"].str.contains("|".join(genes_of_interest), na=False)].copy()

print("Matched rows:")
print(matches["ID"].tolist())

panel_t = matches.set_index("ID").T
panel_t.index.name = "expr_sample"
panel_t.reset_index(inplace=True)

merged = pd.merge(meta, panel_t, on="expr_sample")

out = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_panel.tsv"
merged.to_csv(out, sep="\t", index=False)

print("Saved:", out)
print(merged.head())
