import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

covid = df[df["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"])]
control = df[df["severity"].isin(["Healthy","Control","Convalescent"])]

data = pd.concat([covid, control])
data["label"] = data["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"]).astype(int)

features = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

X = data[features]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values()

importance.plot(kind="barh")
plt.title("Gene importance for COVID classification")
plt.tight_layout()
plt.savefig("gene_feature_importance.png", dpi=300)

print(importance)
