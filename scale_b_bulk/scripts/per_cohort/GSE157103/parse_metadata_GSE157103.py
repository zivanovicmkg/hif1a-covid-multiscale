import pandas as pd
import re

matrix_file = "GSE157103_series_matrix.txt"
out_file = "metadata_GSE157103.tsv"

titles = []
gsm_ids = []

with open(matrix_file) as f:
    for line in f:
        if line.startswith("!Sample_title"):
            titles = line.strip().split("\t")[1:]
            titles = [t.replace('"','') for t in titles]

        if line.startswith("!Sample_geo_accession"):
            gsm_ids = line.strip().split("\t")[1:]
            gsm_ids = [g.replace('"','') for g in gsm_ids]

rows = []

for gsm, title in zip(gsm_ids, titles):
    # Example: COVID_01_39y_male_NonICU
    parts = title.split("_")

    disease = parts[0]
    age = parts[2].replace("y","")
    sex = parts[3]
    severity = parts[4]

    rows.append([gsm, disease, severity, age, sex, title])

df = pd.DataFrame(rows, columns=["GSM","disease","severity","age","sex","title"])
df.to_csv(out_file, sep="\t", index=False)

print(df.head())
print("Rows:", len(df))
