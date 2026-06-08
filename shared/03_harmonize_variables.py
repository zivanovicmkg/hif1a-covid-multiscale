import pandas as pd
from pathlib import Path

BASE = Path.home() / "COVID_AI_Project" / "01_Data_Clinical_10k" / "catalogs"
MASTER_VARS = BASE / "clinical_variables_master_list.tsv"
MAPPING_TEMPLATE = BASE / "dataset_mapping_template.tsv"

master = pd.read_csv(MASTER_VARS, sep="\t")
mapping = pd.read_csv(MAPPING_TEMPLATE, sep="\t")

print("MASTER VARIABLES:", master.shape[0])
print(master[["canonical_name", "category", "required"]].head(15).to_string(index=False))

print("\nMAPPING TEMPLATE:")
print(mapping.to_string(index=False))

print("\nRequired canonical variables:")
req = master[master["required"] == "yes"]["canonical_name"].tolist()
for x in req:
    print("-", x)
