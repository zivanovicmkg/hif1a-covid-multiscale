import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

genes = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

covid = df[df["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"])]
control = df[df["severity"].isin(["Healthy","Control","Convalescent"])]

data = pd.concat([covid, control])

heat = data[genes]
heat = (heat - heat.mean()) / heat.std()

plt.figure(figsize=(8,6))
sns.heatmap(heat.T, cmap="coolwarm", center=0)
plt.title("Hypoxia–Inflammation gene expression heatmap")
plt.tight_layout()
plt.savefig("gene_heatmap.png", dpi=300)
