import pandas as pd

matrix_file = "GSE152075_series_matrix.txt"
out_file = "metadata_GSE152075.tsv"

titles = []
gsm_ids = []

with open(matrix_file) as f:
    for line in f:
        if line.startswith("!Sample_title"):
            titles = [x.replace('"', '') for x in line.strip().split("\t")[1:]]
        if line.startswith("!Sample_geo_accession"):
            gsm_ids = [x.replace('"', '') for x in line.strip().split("\t")[1:]]

rows = []
for gsm, title in zip(gsm_ids, titles):
    expr_sample = title.strip()
    disease = "COVID" if expr_sample.startswith("POS_") else "Control"
    severity = disease
    rows.append([gsm, disease, severity, expr_sample])

df = pd.DataFrame(rows, columns=["GSM", "disease", "severity", "expr_sample"])
df.to_csv(out_file, sep="\t", index=False)

print(df.head())
print("\nRows:", len(df))
print("\nGroup counts:")
print(df["severity"].value_counts())
