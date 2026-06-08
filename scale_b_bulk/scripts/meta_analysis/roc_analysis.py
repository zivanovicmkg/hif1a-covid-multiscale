import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("master_feature_matrix.tsv", sep="\t")

covid = df[df["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"])]
control = df[df["severity"].isin(["Healthy","Control","Convalescent"])]

data = pd.concat([covid, control]).copy()
data["label"] = data["severity"].isin(["COVID","Covid19","Severe","ICU","Hospitalized"]).astype(int)

features = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

X = data[features]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve – COVID vs Control")
plt.legend()
plt.tight_layout()
plt.savefig("roc_covid_vs_control.png", dpi=300)

print("AUC:", roc_auc)
