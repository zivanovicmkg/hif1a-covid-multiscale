import pandas as pd

infile = "GSE212861_Counts_1stPaper.txt"
outfile = "GSE212861_counts_clean.tsv"

df = pd.read_csv(infile, sep="\t")

# ukloni navodnike iz naziva kolona
df.columns = [c.replace('"', '') for c in df.columns]

# prvi stub je Geneid
df.rename(columns={"Geneid": "Gene"}, inplace=True)

df.to_csv(outfile, sep="\t", index=False)

print("Saved:", outfile)
print("Shape:", df.shape)
