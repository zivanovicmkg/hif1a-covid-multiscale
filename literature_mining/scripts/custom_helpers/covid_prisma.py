"""
covid_prisma.py — PRISMA-like flow diagram for the literature mining (Fig S5.5).

Pulls group totals from data/raw_searches/query_validation_counts.csv and the
fetched/abstract counts from data/cited_metrics/*_top.csv. Renders a flow diagram
with matplotlib boxes (no external graphviz dependency).

Output: figures/prisma/FigS5_5_prisma.png/.pdf  + data/prisma/prisma_counts.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modified_framework"))
import covid_config as cfg


def load_counts():
    v = pd.read_csv(cfg.RAW_DIR / "query_validation_counts.csv").set_index("group")
    counts = {g: int(v.loc[g, "n"]) for g in v.index}
    fetched, abstracts = {}, {}
    for g in ["G2", "G3", "G4", "G5"]:
        top = cfg.CITED_DIR / f"{g}_top.csv"
        corpus = cfg.CITED_DIR / f"{g}_corpus.csv"
        abstracts[g] = len(pd.read_csv(top)) if top.exists() else 0
        fetched[g] = len(pd.read_csv(corpus)) if corpus.exists() else 0
    return counts, fetched, abstracts


def box(ax, x, y, w, h, text, fc="#eaf2fb", ec="#3b6ea5"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))


def main():
    counts, fetched, abstracts = load_counts()
    total_abstracts = sum(abstracts.values())

    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    # Identification
    ids = (f"Records identified from PubMed (2019/01/01–2026/06/01)\n"
           f"G1 COVID-19 (all): {counts['G1']:,}\n"
           f"G2 COVID-19 + AI/ML: {counts['G2']:,}\n"
           f"G3 COVID-19 + HIF1A: {counts['G3']:,}\n"
           f"G4 COVID-19 recent 2024–2026: {counts['G4']:,}\n"
           f"G5 COVID-19 + AI vaccines: {counts['G5']:,}")
    box(ax, 1.0, 9.8, 8.0, 1.8, ids, fc="#eaf2fb")

    box(ax, 1.0, 8.0, 8.0, 1.1,
        "Trend & ratio analysis on full counts\n"
        "(per-year counts; AI ratio = G2/G1)", fc="#f3eaf9")

    fetch_txt = ("Records retrieved for citation ranking & abstracts\n"
                 f"G2: {fetched['G2']:,}   G3: {fetched['G3']:,}   "
                 f"G4: {fetched['G4']:,} (recent corpus)   G5: {fetched['G5']:,}")
    box(ax, 1.0, 6.3, 8.0, 1.2, fetch_txt, fc="#eaf6ef")

    rank_txt = ("Top-cited papers selected (OpenAlex cited_by_count)\n"
                f"G2 top {abstracts['G2']} · G3 top {abstracts['G3']} · "
                f"G5 top {abstracts['G5']} · G4 recent-themes corpus")
    box(ax, 1.0, 4.6, 8.0, 1.2, rank_txt, fc="#fdf3e7")

    box(ax, 1.0, 2.9, 8.0, 1.1,
        f"Full abstracts retrieved for review\nN = {total_abstracts}", fc="#fdeaea")

    box(ax, 1.0, 1.2, 8.0, 1.1,
        "Included in Supplementary S5\n"
        "(trend figures, ratio plot, word cloud, topic clusters,\n"
        "top-cited tables, curated abstracts, BibTeX)", fc="#eaf2fb")

    for y0, y1 in [(9.8, 9.1), (8.0, 7.5), (6.3, 5.8), (4.6, 4.0), (2.9, 2.3)]:
        arrow(ax, 5.0, y0, 5.0, y1)

    ax.set_title("Figure S5.5 — Literature mining flow (PRISMA-like)", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIGURES_DIR / "prisma" / f"FigS5_5_prisma.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # persist the numbers
    rows = [{"stage": "identified", "group": g, "n": counts[g]} for g in counts]
    rows += [{"stage": "fetched", "group": g, "n": fetched[g]} for g in fetched]
    rows += [{"stage": "abstracts", "group": g, "n": abstracts[g]} for g in abstracts]
    pd.DataFrame(rows).to_csv(cfg.PRISMA_DIR / "prisma_counts.csv", index=False)
    print(f"PRISMA saved. Total curated abstracts = {total_abstracts}")


if __name__ == "__main__":
    main()
