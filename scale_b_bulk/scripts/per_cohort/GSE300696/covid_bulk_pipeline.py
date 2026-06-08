import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

GENE_PANEL = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

def compute_score(df):
    z = df[GENE_PANEL].apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)
    df["HypoxiaInflammationScore"] = z.mean(axis=1)
    return df

def get_group_col(df):
    if "group" in df.columns:
        return "group"
    if "disease" in df.columns:
        return "disease"
    return None

def get_visit_col(df):
    if "visit" in df.columns:
        return "visit"
    if "time" in df.columns:
        return "time"
    return None

def save_boxplot(df, out_png, out_tiff):
    group_col = get_group_col(df)
    if group_col is None:
        return

    groups = df[group_col].dropna().unique()
    data = [df.loc[df[group_col] == g, "HypoxiaInflammationScore"].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=(5,4))
    ax.boxplot(data, tick_labels=groups)
    ax.set_ylabel("Hypoxia–Inflammation Score")
    ax.set_title("Group comparison")
    plt.tight_layout()

    fig.savefig(out_png, dpi=600)
    plt.close(fig)

    img = Image.open(out_png)
    img.save(out_tiff, dpi=(600,600), compression="tiff_lzw")

def save_visit_plot(df, out_png, out_tiff):
    visit_col = get_visit_col(df)
    if visit_col is None:
        return

    order = sorted(df[visit_col].dropna().unique())
    data = [df.loc[df[visit_col] == v, "HypoxiaInflammationScore"].dropna().values for v in order]

    fig, ax = plt.subplots(figsize=(5,4))
    ax.boxplot(data, tick_labels=order)
    ax.set_ylabel("Hypoxia–Inflammation Score")
    ax.set_title("Longitudinal visits")
    plt.tight_layout()

    fig.savefig(out_png, dpi=600)
    plt.close(fig)

    img = Image.open(out_png)
    img.save(out_tiff, dpi=(600,600), compression="tiff_lzw")

def dataset_summary(df, out_file):
    group_col = get_group_col(df)
    subject_col = "subject" if "subject" in df.columns else None

    summary = {
        "samples_total": len(df),
        "groups": df[group_col].nunique() if group_col else np.nan,
        "subjects": df[subject_col].nunique() if subject_col else np.nan,
        "score_mean": df["HypoxiaInflammationScore"].mean(),
        "score_std": df["HypoxiaInflammationScore"].std()
    }
    pd.DataFrame([summary]).to_csv(out_file, sep="\t", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir + "/tables", exist_ok=True)
    os.makedirs(args.outdir + "/plots", exist_ok=True)
    os.makedirs(args.outdir + "/notes", exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.input, sep="\t")

    print("Computing score...")
    df = compute_score(df)

    print("Saving tables...")
    df.to_csv(args.outdir + "/tables/hypoxia_inflammation_score.tsv", sep="\t", index=False)

    print("Creating plots...")
    save_boxplot(
        df,
        args.outdir + "/plots/score_boxplot.png",
        args.outdir + "/plots/score_boxplot.tiff"
    )

    save_visit_plot(
        df,
        args.outdir + "/plots/score_by_visit.png",
        args.outdir + "/plots/score_by_visit.tiff"
    )

    print("Saving summary...")
    dataset_summary(df, args.outdir + "/tables/dataset_summary.tsv")

    print("Writing report...")
    with open(args.outdir + "/notes/report.txt", "w") as f:
        f.write("Dataset processed\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Mean score: {df['HypoxiaInflammationScore'].mean()}\n")

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
