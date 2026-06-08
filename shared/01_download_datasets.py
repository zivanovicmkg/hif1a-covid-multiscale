import pandas as pd
from pathlib import Path

base = Path.home() / "COVID_AI_Project" / "01_Data_Clinical_10k" / "catalogs"

master = pd.read_csv(base / "clinical_datasets_master_catalog.tsv", sep="\t")
queue = pd.read_csv(base / "source_priority_queue.tsv", sep="\t")
audit = pd.read_csv(base / "access_audit.tsv", sep="\t")

# unify key column
if "Dataset" in master.columns and "source_name" not in master.columns:
    master = master.rename(columns={"Dataset": "source_name"})
if "Source" in master.columns and "source_type" not in master.columns:
    master = master.rename(columns={"Source": "source_type"})

df = master.merge(queue, on="source_name", how="outer", suffixes=("_master", "_queue"))
df = df.merge(audit, on="source_name", how="outer", suffixes=("", "_audit"))

# keep one priority column if duplicates exist
priority_cols = [c for c in df.columns if c.startswith("priority")]
if "priority" not in df.columns and priority_cols:
    df["priority"] = df[priority_cols[0]]
elif len(priority_cols) > 1:
    df["priority"] = df[priority_cols].bfill(axis=1).iloc[:, 0]

cols = [
    "source_name",
    "priority",
    "Patients",
    "Type",
    "Level",
    "Access",
    "role",
    "access_mode",
    "current_status",
    "next_action",
    "next_step",
    "Status",
]
existing = [c for c in cols if c in df.columns]
df = df[existing].drop_duplicates().sort_values(
    by=["priority", "source_name"], na_position="last"
)

print("\n=== CLINICAL DATASET ACQUISITION REGISTRY ===\n")
print(df.to_string(index=False))

print("\n=== PUBLIC / DOWNLOAD CANDIDATES ===")
download_now = df[
    df.apply(
        lambda r: str(r.get("Access", "")).lower() in {"public", "download"}
        or "public" in str(r.get("access_mode", "")).lower(),
        axis=1,
    )
]
if len(download_now) == 0:
    print("No fully verified public dataset ready for automatic download yet.")
else:
    print(download_now.to_string(index=False))

print("\n=== REQUEST / APPLICATION SOURCES ===")
request_sources = df[
    df.apply(
        lambda r: any(
            x in str(r.get("Access", "")).lower() or x in str(r.get("access_mode", "")).lower()
            for x in ["request", "application", "secure_platform"]
        ),
        axis=1,
    )
]
if len(request_sources) == 0:
    print("None")
else:
    print(request_sources.to_string(index=False))

outdir = Path.home() / "COVID_AI_Project" / "01_Data_Clinical_10k" / "merged"
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / "clinical_dataset_acquisition_registry.tsv"
df.to_csv(outfile, sep="\t", index=False)

print(f"\nSaved: {outfile}")
