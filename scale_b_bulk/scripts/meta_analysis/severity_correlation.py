import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

severity_rank = {
    "Healthy": 0,
    "Control": 0,
    "Convalescent": 1,
    "COVID": 2,
    "Covid19": 2,
    "Hospitalized": 3,
    "NonICU": 4,
    "ICU": 5,
    "Severe": 6,
    "Covid19_SDRA": 7,
}

df["severity_rank"] = df["severity"].map(severity_rank)

df2 = df.dropna(subset=["severity_rank", "panel_score_z"]).copy()

rho, p = spearmanr(df2["panel_score_z"], df2["severity_rank"])

print("N:", len(df2))
print("Spearman rho:", rho)
print("p-value:", p)
