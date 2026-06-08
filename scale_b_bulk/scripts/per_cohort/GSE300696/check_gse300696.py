import pandas as pd

meta = pd.read_csv(
    "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/metadata_parsed.tsv",
    sep="\t"
)

expr = pd.read_csv(
    "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/GSE300696_expression_matrix_genes.results_TPM.tsv",
    sep="\t"
)

expr_samples = expr.columns[1:].tolist()
meta_samples = meta["GSM"].tolist()

print("Expression columns:", len(expr_samples))
print("Metadata rows:", len(meta_samples))
print("First 5 expression samples:", expr_samples[:5])
print("First 5 metadata GSM:", meta_samples[:5])

common = set(expr_samples).intersection(set(meta_samples))
print("Common sample names:", len(common))
