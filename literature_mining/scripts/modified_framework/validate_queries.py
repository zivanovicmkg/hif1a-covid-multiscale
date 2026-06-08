"""
validate_queries.py — Cheap pre-flight check before the long Phase 2 run.

Runs one count-only (retmax=0) esearch per group with its date bounds, prints
actual N vs the plan's estimate, and writes a small CSV. ~6 API calls total.
"""

import csv
import time

import covid_config as cfg

Entrez = cfg.configure_entrez()

PLAN_EST = {
    "G1": "~400,000",
    "G2": "~25,000-40,000",
    "G3": "~200-500",
    "G4": "~80,000-100,000",
    "G5": "~500-1,500",
}


def count(term: str, d0: str, d1: str) -> int:
    q = f'({term}) AND ("{d0}"[PDAT] : "{d1}"[PDAT])'
    h = Entrez.esearch(db="pubmed", term=q, retmax=0)
    rec = Entrez.read(h)
    h.close()
    return int(rec.get("Count", "0"))


def main():
    qs = cfg.queries(decorated=True)
    rows = []
    print(f"API key active: {bool(cfg.NCBI_API_KEY)} | sleep={cfg.SLEEP_SEC}s\n")
    print(f"{'Grp':<4}{'N (actual)':>14}  {'plan estimate':<18}{'label'}")
    print("-" * 70)
    for gid, term in qs.items():
        d0, d1 = cfg.date_bounds(gid)
        n = count(term, d0, d1)
        rows.append({"group": gid, "n": n, "plan_estimate": PLAN_EST[gid],
                     "date_start": d0, "date_end": d1,
                     "label": cfg.GROUP_LABELS[gid]})
        print(f"{gid:<4}{n:>14,}  {PLAN_EST[gid]:<18}{cfg.GROUP_LABELS[gid]}")
        time.sleep(cfg.SLEEP_SEC)

    out = cfg.RAW_DIR / "query_validation_counts.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n", "plan_estimate",
                                          "date_start", "date_end", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
