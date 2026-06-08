import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

covid = df[df["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"])]
control = df[df["severity"].isin(["Healthy","Control","Convalescent"])]

data = [
    covid["panel_score_z"].dropna(),
    control["panel_score_z"].dropna()
]

plt.figure(figsize=(6,5))
plt.boxplot(data, tick_labels=["COVID", "Control"])

plt.ylabel("panel_score_z")
plt.title("Hypoxia–Inflammation score: COVID vs Control")

plt.tight_layout()
plt.savefig("covid_vs_control_boxplot.png", dpi=300)

print("Saved: covid_vs_control_boxplot.png")
