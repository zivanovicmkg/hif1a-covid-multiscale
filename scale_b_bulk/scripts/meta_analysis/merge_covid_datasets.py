import pandas as pd
import numpy as np

genes = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

files = {
    "GSE300696": "hypoxia_inflammation_panel_collapsed_GSE300696.tsv",
    "GSE212861": "hypoxia_inflammation_panel_collapsed_GSE212861.tsv",
    "GSE157103": "hypoxia_inflammation_panel_collapsed_GSE157103.tsv",
    "GSE171110": "hypoxia_inflammation_panel_collapsed_GSE171110.tsv",
    "GSE152075": "hypoxia_inflammation_panel_collapsed_GSE152075.tsv",
}

dfs = []

for dataset, file in files.items():
    df = pd.read_csv(file, sep="\t").copy()
    df["dataset"] = dataset

    # standardizuj identitet uzorka
    if "GSM" not in df.columns:
        df["GSM"] = [f"{dataset}_{i+1}" for i in range(len(df))]

    # standardizuj severity
    if "severity" not in df.columns:
        if "group" in df.columns:
            df["severity"] = df["group"]
        elif "disease" in df.columns:
            df["severity"] = df["disease"]
        else:
            df["severity"] = "Unknown"

    # standardizuj timepoint
    if "timepoint" not in df.columns:
        if "visit" in df.columns:
            df["timepoint"] = df["visit"]
        elif "time" in df.columns:
            df["timepoint"] = df["time"]
        else:
            df["timepoint"] = np.nan

    # standardizuj disease
    if "disease" not in df.columns:
        df["disease"] = df["severity"]

    # osiguraj da su geni numerički
    for g in genes:
        df[g] = pd.to_numeric(df[g], errors="coerce")

    # within-dataset z-score po genu
    z = df[genes].apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)
    z.columns = [f"{c}_z" for c in z.columns]
    df = pd.concat([df, z], axis=1)

    # score iz z-vrednosti, ne iz raw
    df["panel_score_z"] = z.mean(axis=1)

    keep_cols = [
        "dataset", "GSM", "disease", "severity", "timepoint"
    ] + genes + [f"{g}_z" for g in genes] + ["panel_score_z"]

    dfs.append(df[keep_cols])

master = pd.concat(dfs, ignore_index=True)
master.to_csv("master_feature_matrix.tsv", sep="\t", index=False)

print(master.shape)
print("\nPer dataset:")
print(master["dataset"].value_counts())
print("\nPer severity:")
print(master["severity"].value_counts(dropna=False).head(20))
print("\nSaved: master_feature_matrix.tsv")
