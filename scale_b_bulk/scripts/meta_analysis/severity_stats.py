import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

covid = df[df["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"])]["panel_score_z"]
control = df[df["severity"].isin(["Healthy","Control","Convalescent"])]["panel_score_z"]

stat, p = mannwhitneyu(covid, control)

print("COVID vs Control")
print("N covid:", len(covid))
print("N control:", len(control))
print("U statistic:", stat)
print("p-value:", p)
print("Mean covid:", covid.mean())
print("Mean control:", control.mean())
