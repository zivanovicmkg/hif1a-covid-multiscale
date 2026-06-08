import pandas as pd

# učitaj metadata
meta = pd.read_csv(
    "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/metadata_parsed.tsv",
    sep="\t"
)

# učitaj samo header iz expression matrice
expr = pd.read_csv(
    "/home/marko-b2/COVID_Transcriptomics_AI/01_Raw_Data/Genetics/GSE300696_expression_matrix_genes.results_TPM.tsv",
    sep="\t",
    nrows=0
)

expr_samples = expr.columns[1:].tolist()

print("Expression samples:", len(expr_samples))
print("Metadata rows:", len(meta))

if len(expr_samples) != len(meta):
    raise ValueError("Broj expression sample-ova i metadata redova nije isti.")

# dodaj expression sample imena po redosledu
meta = meta.copy()
meta["expr_sample"] = expr_samples

# sačuvaj mapu
out = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/sample_map.tsv"
meta.to_csv(out, sep="\t", index=False)

print("Saved:", out)
print(meta.head(10).to_string(index=False))
