import pandas as pd
import re

matrix_file = "GSE212861_series_matrix.txt"
out_file = "metadata_GSE212861.tsv"

samples = []
disease = []
time = []
batch = []

with open(matrix_file) as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            samples = line.strip().split("\t")[1:]
        if "disease state" in line:
            disease = [re.sub('"', '', x.split(": ")[1]) for x in line.strip().split("\t")[1:]]
        if "time:" in line:
            time = [re.sub('"', '', x.split(": ")[1]) for x in line.strip().split("\t")[1:]]
        if "batch:" in line:
            batch = [re.sub('"', '', x.split(": ")[1]) for x in line.strip().split("\t")[1:]]

df = pd.DataFrame({
    "sample": samples,
    "disease": disease,
    "time": time,
    "batch": batch
})

df.to_csv(out_file, sep="\t", index=False)
print("Saved metadata:", out_file)
