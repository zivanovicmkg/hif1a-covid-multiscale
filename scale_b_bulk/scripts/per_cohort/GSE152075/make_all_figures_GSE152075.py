import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

GENES = ["HIF1A", "VEGFA", "SLC2A1", "LDHA", "PDK1", "IL6", "TNF", "CXCL8", "STAT3", "MMP9"]

META_FILE = "metadata_GSE152075.tsv"
COUNTS_FILE = "GSE152075_raw_counts_GEO.txt"
OUTDIR = "figures_GSE152075"

os.makedirs(OUTDIR, exist_ok=True)


def load_panel_expression(counts_file: str, genes: list[str]) -> pd.DataFrame:
    with open(counts_file) as f:
        header_samples = f.readline().strip().split()

    gene_rows = {}
    for g in genes:
        gene_rows[g] = None

    with open(counts_file) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            gene = parts[0]
            if gene in gene_rows:
                values = parts[1:]
                if len(values) != len(header_samples):
                    raise ValueError(
                        f"Gene {gene} has {len(values)} values, expected {len(header_samples)}"
                    )
                gene_rows[gene] = [float(x) for x in values]

    missing = [g for g, v in gene_rows.items() if v is None]
    if missing:
        raise ValueError(f"Missing genes in matrix: {missing}")

    df = pd.DataFrame({"expr_sample": header_samples})
    for g in genes:
        df[g] = gene_rows[g]
    return df


def zscore_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mean = out[c].mean()
        std = out[c].std(ddof=0)
        if std == 0:
            out[c + "_z"] = 0.0
        else:
            out[c + "_z"] = (out[c] - mean) / std
    return out


def save_boxplot(df: pd.DataFrame, path_png: str, path_tiff: str) -> None:
    covid = df[df["severity"] == "COVID"]["panel_score_z"].dropna()
    control = df[df["severity"] == "Control"]["panel_score_z"].dropna()

    plt.figure(figsize=(6, 5))
    plt.boxplot([covid, control], tick_labels=["COVID", "Control"])
    plt.ylabel("Hypoxia–Inflammation score (z)")
    plt.title("GSE152075: COVID vs Control")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_tiff, dpi=600)
    plt.close()


def save_histogram(df: pd.DataFrame, path_png: str, path_tiff: str) -> None:
    covid = df[df["severity"] == "COVID"]["panel_score_z"].dropna()
    control = df[df["severity"] == "Control"]["panel_score_z"].dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(control, bins=25, alpha=0.5, label="Control")
    plt.hist(covid, bins=25, alpha=0.5, label="COVID")
    plt.xlabel("Hypoxia–Inflammation score (z)")
    plt.ylabel("Count")
    plt.title("GSE152075: score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_tiff, dpi=600)
    plt.close()


def save_pca(df: pd.DataFrame, zcols: list[str], path_png: str, path_tiff: str) -> None:
    X = df[zcols].values
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    plot_df = df.copy()
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]

    plt.figure(figsize=(6, 5))
    for grp in ["Control", "COVID"]:
        sub = plot_df[plot_df["severity"] == grp]
        plt.scatter(sub["PC1"], sub["PC2"], label=grp, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("GSE152075: PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_tiff, dpi=600)
    plt.close()


def save_heatmap(df: pd.DataFrame, zcols: list[str], path_png: str, path_tiff: str) -> None:
    ordered = df.sort_values(["severity", "expr_sample"]).copy()
    mat = ordered[zcols].T.values
    row_labels = [c.replace("_z", "") for c in zcols]

    plt.figure(figsize=(12, 4))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xticks([])
    plt.title("GSE152075: hypoxia–inflammation heatmap")
    cbar = plt.colorbar(im)
    cbar.set_label("z-score")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_tiff, dpi=600)
    plt.close()


def save_roc_and_importance(
    df: pd.DataFrame, features: list[str], roc_png: str, roc_tiff: str, imp_png: str, imp_tiff: str
) -> pd.Series:
    model_df = df.copy()
    model_df["label"] = (model_df["severity"] == "COVID").astype(int)

    X = model_df[features]
    y = model_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GSE152075: ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_png, dpi=300)
    plt.savefig(roc_tiff, dpi=600)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=features).sort_values()

    plt.figure(figsize=(6, 5))
    importance.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("GSE152075: feature importance")
    plt.tight_layout()
    plt.savefig(imp_png, dpi=300)
    plt.savefig(imp_tiff, dpi=600)
    plt.close()

    print(f"AUC: {roc_auc:.3f}")
    return importance


def main() -> None:
    meta = pd.read_csv(META_FILE, sep="\t")
    panel_expr = load_panel_expression(COUNTS_FILE, GENES)

    df = pd.merge(meta, panel_expr, on="expr_sample", how="inner")
    df = zscore_df(df, GENES)
    zcols = [g + "_z" for g in GENES]
    df["panel_score_z"] = df[zcols].mean(axis=1)

    df.to_csv(os.path.join(OUTDIR, "GSE152075_panel_with_score.tsv"), sep="\t", index=False)

    save_boxplot(
        df,
        os.path.join(OUTDIR, "Fig1_score_boxplot.png"),
        os.path.join(OUTDIR, "Fig1_score_boxplot.tiff"),
    )
    save_histogram(
        df,
        os.path.join(OUTDIR, "Fig2_score_histogram.png"),
        os.path.join(OUTDIR, "Fig2_score_histogram.tiff"),
    )
    save_pca(
        df,
        zcols,
        os.path.join(OUTDIR, "Fig3_pca.png"),
        os.path.join(OUTDIR, "Fig3_pca.tiff"),
    )
    save_heatmap(
        df,
        zcols,
        os.path.join(OUTDIR, "Fig4_heatmap.png"),
        os.path.join(OUTDIR, "Fig4_heatmap.tiff"),
    )
    importance = save_roc_and_importance(
        df,
        GENES,
        os.path.join(OUTDIR, "Fig5_roc.png"),
        os.path.join(OUTDIR, "Fig5_roc.tiff"),
        os.path.join(OUTDIR, "Fig6_feature_importance.png"),
        os.path.join(OUTDIR, "Fig6_feature_importance.tiff"),
    )

    print("\nFeature importance:")
    print(importance)
    print(f"\nSaved all files in: {OUTDIR}")


if __name__ == "__main__":
    main()
