import pandas as pd

expr_file = "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/GSE300696_expression_matrix_genes.results_TPM.tsv"
sample_map_file = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/sample_map.tsv"

target_genes = [
    "_HIF1A",
    "_VEGFA",
    "_SLC2A1",
    "_LDHA",
    "_PDK1",
    "_IL6\t",
    "_TNF\t",
    "_CXCL8",
    "_STAT3",
    "_MMP9"
]

expr = pd.read_csv(expr_file, sep="\t")
meta = pd.read_csv(sample_map_file, sep="\t")

rows = []
for gene in target_genes:
    r = expr[expr["ID"].str.contains(gene, regex=False)]
    rows.append(r)

panel = pd.concat(rows)

print("Selected genes:")
print(panel["ID"].tolist())

panel_t = panel.set_index("ID").T
panel_t.index.name = "expr_sample"
panel_t.reset_index(inplace=True)

merged = pd.merge(meta, panel_t, on="expr_sample")

out = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_panel_clean.tsv"
merged.to_csv(out, sep="\t", index=False)

print("Saved:", out)
print(merged.head())
