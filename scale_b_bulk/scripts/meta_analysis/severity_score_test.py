import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

plt.figure(figsize=(8,5))

groups = [g for g in df["severity"].dropna().unique()]
data = [df[df["severity"] == g]["panel_score_z"].dropna() for g in groups]

plt.boxplot(data, tick_labels=groups)
plt.xticks(rotation=45, ha="right")
plt.ylabel("panel_score_z")
plt.title("Hypoxia–Inflammation score by severity")
plt.tight_layout()
plt.savefig("panel_score_by_severity_boxplot.png", dpi=300)

print("Saved: panel_score_by_severity_boxplot.png")
