import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

order = ["Healthy","Control","Convalescent","Covid19","Hospitalized","NonICU","ICU","Severe","Covid19_SDRA"]

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="severity", y="panel_score_z", order=order)
plt.xticks(rotation=45)
plt.title("Hypoxia–Inflammation score vs severity")
plt.tight_layout()
plt.savefig("severity_trend.png", dpi=300)
