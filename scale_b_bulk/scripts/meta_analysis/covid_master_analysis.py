import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

genes_z = [f"{g}_z" for g in ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]]

X = df[genes_z].values

pca = PCA(n_components=2)
pc = pca.fit_transform(X)

df["PC1"] = pc[:, 0]
df["PC2"] = pc[:, 1]

# PCA po datasetu
plt.figure(figsize=(6,5))
for ds in df["dataset"].unique():
    sub = df[df["dataset"] == ds]
    plt.scatter(sub["PC1"], sub["PC2"], label=ds, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA by dataset")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("PCA_by_dataset.png", dpi=300)
plt.close()

# PCA po severity
plt.figure(figsize=(6,5))
for sev in df["severity"].dropna().unique():
    sub = df[df["severity"] == sev]
    plt.scatter(sub["PC1"], sub["PC2"], label=sev, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA by severity")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("PCA_by_severity.png", dpi=300)
plt.close()

# score distribucija po datasetu
plt.figure(figsize=(7,5))
for ds in df["dataset"].unique():
    sub = df[df["dataset"] == ds]
    plt.hist(sub["panel_score_z"], bins=20, alpha=0.4, label=ds)
plt.xlabel("panel_score_z")
plt.ylabel("Count")
plt.title("Panel score distribution by dataset")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("panel_score_distribution_by_dataset.png", dpi=300)
plt.close()

print("Saved:")
print("  PCA_by_dataset.png")
print("  PCA_by_severity.png")
print("  panel_score_distribution_by_dataset.png")
