import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

covid = df[df["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"])]["panel_score_z"]
control = df[df["severity"].isin(["Healthy","Control","Convalescent"])]["panel_score_z"]

plt.figure(figsize=(6,5))
plt.violinplot([covid, control], showmeans=True)

plt.xticks([1,2], ["COVID", "Control"])
plt.ylabel("panel_score_z")
plt.title("Hypoxia–Inflammation score distribution")

plt.tight_layout()
plt.savefig("covid_vs_control_violin.png", dpi=300)

print("Saved: covid_vs_control_violin.png")
