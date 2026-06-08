import pandas as pd

expr_file = "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/GSE300696_expression_matrix_genes.results_TPM.tsv"
sample_map_file = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/sample_map.tsv"

# učitaj expression matricu
expr = pd.read_csv(expr_file, sep="\t")

# pronađi HIF1A red
hif = expr[expr["ID"].str.contains("HIF1A", na=False)]

print("HIF1A rows:")
print(hif.head())

# transponuj
hif_t = hif.set_index("ID").T
hif_t.index.name = "expr_sample"
hif_t.reset_index(inplace=True)

# učitaj sample map
meta = pd.read_csv(sample_map_file, sep="\t")

# spoji
merged = pd.merge(meta, hif_t, on="expr_sample")

out = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/HIF1A_expression.tsv"
merged.to_csv(out, sep="\t", index=False)

print("Saved:", out)
print(merged.head())
