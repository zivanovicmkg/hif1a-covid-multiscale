"""
covid_counts.py — Counts-by-year per group + year x group matrix.

Adapted from octe-ai-mining-framework/literature_octe_ai.py (counts-only esearch,
retmax=0, so it is safe regardless of group size — no paging needed).

Outputs (data/raw_searches/):
  G1.csv, G2.csv, G3.csv, G5.csv   (Year, Count)
  covid_lit_matrix.csv             (Year, G1, G2, G3, G5)  -> trend/ratio/heatmap
  covid_ratio_by_year.csv          (Year, G1, G2, ai_ratio_pct)

G4 is intentionally excluded here: its query is identical to G1 and it only
differs by date window, so a separate yearly column would duplicate G1. G4's role
(recent-themes abstracts) is handled by covid_top_cited.py.
"""

import csv
import time

import covid_config as cfg

Entrez = cfg.configure_entrez()

# Groups that get a yearly trend column (skip G4 = redundant with G1).
TREND_GROUPS = ["G1", "G2", "G3", "G5"]


def count_year(term: str, year: int) -> int:
    q = f'({term}) AND ("{year}/01/01"[PDAT] : "{year}/12/31"[PDAT])'
    h = Entrez.esearch(db="pubmed", term=q, retmax=0)
    rec = Entrez.read(h)
    h.close()
    return int(rec.get("Count", "0"))


def main():
    qs = cfg.queries(decorated=True)
    all_counts = {g: {} for g in TREND_GROUPS}

    for g in TREND_GROUPS:
        term = qs[g]
        print(f"\n=== {g}: {cfg.GROUP_LABELS[g]} ===")
        for y in cfg.YEARS:
            n = count_year(term, y)
            all_counts[g][y] = n
            print(f"  {y}: {n:,}")
            time.sleep(cfg.SLEEP_SEC)
        # per-group CSV
        fn = cfg.RAW_DIR / f"{g}.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Year", "Count"])
            for y in cfg.YEARS:
                w.writerow([y, all_counts[g][y]])
        print(f"  saved {fn}")

    # matrix
    mfn = cfg.RAW_DIR / "covid_lit_matrix.csv"
    with open(mfn, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Year"] + TREND_GROUPS)
        for y in cfg.YEARS:
            w.writerow([y] + [all_counts[g][y] for g in TREND_GROUPS])
    print(f"\nSaved matrix: {mfn}")

    # AI ratio by year (key figure for goal a)
    rfn = cfg.RAW_DIR / "covid_ratio_by_year.csv"
    with open(rfn, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Year", "G1_total", "G2_ai", "ai_ratio_pct"])
        for y in cfg.YEARS:
            g1 = all_counts["G1"][y]
            g2 = all_counts["G2"][y]
            ratio = round(100.0 * g2 / g1, 3) if g1 else 0.0
            w.writerow([y, g1, g2, ratio])
    print(f"Saved ratio: {rfn}")

    tot1 = sum(all_counts["G1"].values())
    tot2 = sum(all_counts["G2"].values())
    print(f"\nTotals -> G1={tot1:,} G2={tot2:,} overall AI ratio={100*tot2/tot1:.2f}%")


if __name__ == "__main__":
    main()
