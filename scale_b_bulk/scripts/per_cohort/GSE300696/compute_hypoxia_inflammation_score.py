import pandas as pd
import numpy as np

infile = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_panel_collapsed.tsv"
outfile = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_score.tsv"

genes = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

df = pd.read_csv(infile, sep="\t")

# z-score po genu
z = df[genes].apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)

df["HypoxiaInflammationScore"] = z.mean(axis=1)

df.to_csv(outfile, sep="\t", index=False)

print("Saved:", outfile)
print(df[["GSM","group","subject","visit","HypoxiaInflammationScore"]].head())

print("\nGroup summary:")
print(df.groupby("group")["HypoxiaInflammationScore"].agg(["count","mean","median","std"]))

print("\nVisit summary:")
print(df.groupby("visit")["HypoxiaInflammationScore"].agg(["count","mean","median","std"]).sort_index())
