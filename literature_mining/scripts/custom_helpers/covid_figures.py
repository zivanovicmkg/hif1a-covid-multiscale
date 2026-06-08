"""
covid_figures.py — Trend / ratio / heatmap figures from the counts matrix.

Inputs : data/raw_searches/covid_lit_matrix.csv, covid_ratio_by_year.csv
Outputs: figures/trends/, figures/ratios/, figures/heatmaps/  (300 dpi PNG + PDF)

Note: 2026 is a partial year (PubMed window ends 2026/06/01 and recent months lag
in indexing); it is annotated as such on every figure.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modified_framework"))
import covid_config as cfg

sns.set_theme(style="whitegrid", context="paper")
PARTIAL_YEAR = 2026
LABELS = cfg.GROUP_LABELS


def _save(fig, *paths):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load():
    m = pd.read_csv(cfg.RAW_DIR / "covid_lit_matrix.csv")
    r = pd.read_csv(cfg.RAW_DIR / "covid_ratio_by_year.csv")
    return m, r


def fig_growth(m):
    """S5.1 — annual COVID total (G1) + AI/HIF/vaccine groups on a log panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Panel a: COVID total
    bars = ax1.bar(m["Year"], m["G1"], color="#3b6ea5")
    if PARTIAL_YEAR in m["Year"].values:
        idx = m.index[m["Year"] == PARTIAL_YEAR][0]
        bars[idx].set_hatch("//")
        bars[idx].set_alpha(0.55)
    ax1.set_title("(a) COVID-19 publications per year (G1, all)")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Publications")
    ax1.ticklabel_format(axis="y", style="plain")

    # Panel b: smaller groups, log scale
    for g, c in [("G2", "#d1495b"), ("G5", "#edae49"), ("G3", "#66a182")]:
        ax2.plot(m["Year"], m[g].clip(lower=0.5), marker="o", label=LABELS[g], color=c)
    ax2.set_yscale("log")
    ax2.set_title("(b) AI, AI-vaccine, and HIF1A subsets (log scale)")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Publications (log)")
    ax2.legend(fontsize=8)
    fig.suptitle("Figure S5.1 — COVID-19 literature growth, 2019–2026 "
                 "(2026 partial, hatched)", fontsize=11)
    _save(fig, cfg.FIGURES_DIR / "trends" / "FigS5_1_growth.png",
          cfg.FIGURES_DIR / "trends" / "FigS5_1_growth.pdf")


def fig_ratio(r):
    """S5.3 — KEY figure for goal (a): AI subset as % of COVID total, by year."""
    rr = r[r["G1_total"] > 0].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rr["Year"], rr["ai_ratio_pct"], marker="o", color="#d1495b", lw=2)
    for _, row in rr.iterrows():
        ax.annotate(f"{row['ai_ratio_pct']:.1f}%",
                    (row["Year"], row["ai_ratio_pct"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.axvline(2022, ls="--", color="grey", alpha=0.7)
    ax.text(2022.05, ax.get_ylim()[1]*0.15, "LLM era\n(2022+)", fontsize=8, color="grey")
    ax.set_title("Figure S5.3 — AI/ML as a share of COVID-19 publications")
    ax.set_xlabel("Year"); ax.set_ylabel("AI/ML papers as % of COVID total")
    overall = 100 * rr["G2_ai"].sum() / rr["G1_total"].sum()
    ax.axhline(overall, ls=":", color="#3b6ea5", alpha=0.8)
    ax.text(rr["Year"].min(), overall, f" overall {overall:.2f}%",
            va="bottom", fontsize=8, color="#3b6ea5")
    _save(fig, cfg.FIGURES_DIR / "ratios" / "FigS5_3_ai_ratio.png",
          cfg.FIGURES_DIR / "ratios" / "FigS5_3_ai_ratio.pdf")
    return overall


def fig_heatmap(m):
    """S5.2 — group x year heatmap, row-normalized so each group's trend shows."""
    groups = ["G1", "G2", "G3", "G5"]
    mat = m.set_index("Year")[groups].T
    norm = mat.div(mat.max(axis=1), axis=0)  # 0..1 per group
    fig, ax = plt.subplots(figsize=(9, 3.2))
    sns.heatmap(norm, annot=mat.astype(int), fmt="d", cmap="rocket_r",
                cbar_kws={"label": "intensity (row-normalized)"},
                yticklabels=[LABELS[g] for g in groups], ax=ax, annot_kws={"fontsize": 7})
    ax.set_title("Figure S5.2 — Publication intensity by group and year "
                 "(cells = raw counts; 2026 partial)")
    ax.set_xlabel("Year"); ax.set_ylabel("")
    _save(fig, cfg.FIGURES_DIR / "heatmaps" / "FigS5_2_heatmap.png",
          cfg.FIGURES_DIR / "heatmaps" / "FigS5_2_heatmap.pdf")


def main():
    m, r = load()
    fig_growth(m)
    overall = fig_ratio(r)
    fig_heatmap(m)
    print(f"Figures written. Overall AI ratio = {overall:.2f}%")
    print("  figures/trends/FigS5_1_growth.png")
    print("  figures/ratios/FigS5_3_ai_ratio.png")
    print("  figures/heatmaps/FigS5_2_heatmap.png")


if __name__ == "__main__":
    main()
