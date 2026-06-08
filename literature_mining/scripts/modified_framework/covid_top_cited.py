"""
covid_top_cited.py — PMIDs -> metadata+abstract -> OpenAlex citations -> top-N.

Adapted from octe-ai-mining-framework/top_cited.py with three changes:
  1. COVID query groups + API key (via covid_config).
  2. History-server paging (usehistory) so groups > 10,000 records are not
     silently truncated at PubMed's retmax cap (the original used retmax=100000,
     which PubMed caps at 10,000).
  3. Abstract text retrieved in the same efetch pass and saved alongside.

Outputs:
  data/cited_metrics/<G>_all.csv        full fetched set (pmid,doi,year,journal,
                                        title,citations,abstract_present)
  data/cited_metrics/<G>_top.csv        top-N ranked by OpenAlex cited_by_count
  data/abstracts/<G>/<rank>_<pmid>.txt  abstract text for each top-N paper

Run for selected groups, e.g.:
  python3 covid_top_cited.py --groups G3 G5 G2
  python3 covid_top_cited.py --groups G4 --max-fetch 3000   # recent-themes corpus
"""

import argparse
import csv
import time
import warnings
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

import covid_config as cfg

Entrez = cfg.configure_entrez()
warnings.filterwarnings("ignore", category=UserWarning, module="ssl")
OPENALEX_BASE = "https://api.openalex.org/works"


# ----------------------------- HTTP session -----------------------------
def session_with_retry() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=6, backoff_factor=0.6,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": f"CovidLitMining/1.0 (mailto:{cfg.OPENALEX_MAILTO})"})
    return s


# ----------------------------- PubMed -----------------------------
EFETCH_HISTORY_CAP = 9800  # PubMed efetch via history server fails past ~10,000

def esearch_history(term: str, d0: str, d1: str):
    """Return (count, webenv, query_key) using the history server."""
    q = f'({term}) AND ("{d0}"[PDAT] : "{d1}"[PDAT])'
    h = Entrez.esearch(db="pubmed", term=q, retmax=0, usehistory="y")
    rec = Entrez.read(h)
    h.close()
    return int(rec["Count"]), rec["WebEnv"], rec["QueryKey"]


def years_in_range(d0: str, d1: str):
    return list(range(int(d0[:4]), int(d1[:4]) + 1))


def _abstract_text(art) -> str:
    try:
        abst = art["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
    except (KeyError, TypeError):
        return ""
    parts = []
    for seg in abst:
        label = ""
        try:
            label = seg.attributes.get("Label", "")
        except AttributeError:
            pass
        text = str(seg)
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts).strip()


def fetch_records(webenv, query_key, total, max_fetch, sleep):
    """Page efetch over one history set (must be <= EFETCH_HISTORY_CAP)."""
    out = []
    n = min(total, max_fetch) if max_fetch else total
    n = min(n, EFETCH_HISTORY_CAP)  # never page past the history-server cap
    BATCH = 200
    for start in range(0, n, BATCH):
        h = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml",
                          retstart=start, retmax=BATCH,
                          webenv=webenv, query_key=query_key)
        rec = Entrez.read(h)
        h.close()
        for art in rec.get("PubmedArticle", []):
            d = {"pmid": None, "title": "", "year": None, "journal": "",
                 "doi": None, "abstract": ""}
            d["pmid"] = str(art["MedlineCitation"]["PMID"])
            article = art["MedlineCitation"]["Article"]
            d["title"] = str(article.get("ArticleTitle", ""))
            try:
                pd_ = article["Journal"]["JournalIssue"]["PubDate"]
                d["year"] = pd_.get("Year") or pd_.get("MedlineDate", "")[:4]
            except Exception:
                pass
            try:
                d["journal"] = str(article["Journal"]["Title"])
            except Exception:
                pass
            try:
                for aid in art["PubmedData"]["ArticleIdList"]:
                    if aid.attributes.get("IdType") == "doi":
                        d["doi"] = str(aid)
                        break
            except Exception:
                pass
            d["abstract"] = _abstract_text(art)
            out.append(d)
        print(f"    fetched {min(start+BATCH, n)}/{n}")
        time.sleep(sleep)
    return out


# ----------------------------- OpenAlex -----------------------------
def openalex_citations(dois, sess, sleep):
    # OpenAlex OR-filter caps at 50 values; comma means AND, so OR with '|'.
    out = {}
    BATCH = 50
    dois = [d for d in dois if d]
    for i in range(0, len(dois), BATCH):
        chunk = [d.lower() for d in dois[i:i+BATCH]]
        filt = "filter=doi:" + "|".join(quote(d) for d in chunk)
        url = f"{OPENALEX_BASE}?{filt}&per-page=200&mailto={quote(cfg.OPENALEX_MAILTO)}"
        r = sess.get(url, timeout=30)
        if r.status_code == 200:
            for item in r.json().get("results", []):
                doi = (item.get("doi") or "").lower().replace("https://doi.org/", "")
                out[doi] = (item.get("cited_by_count", 0), item.get("id"))
        else:
            print(f"    openalex WARN status {r.status_code} on batch {i//BATCH}")
        print(f"    openalex {min(i+BATCH, len(dois))}/{len(dois)}")
        time.sleep(sleep)
    return out


# ----------------------------- pipeline -----------------------------
def process_group(gid, max_fetch, top_n, sess):
    term = cfg.queries(decorated=True)[gid]
    d0, d1 = cfg.date_bounds(gid)
    print(f"\n=== {gid}: {cfg.GROUP_LABELS[gid]} ===")
    grand_total, _, _ = esearch_history(term, d0, d1)
    print(f"  records: {grand_total:,}  (fetching up to {max_fetch or grand_total:,})")

    # Partition by year so no single efetch history set exceeds the 10k cap.
    yrs = years_in_range(d0, d1)
    per_year_cap = None
    if max_fetch:
        # distribute the cap across years (e.g. G4 recent-themes corpus)
        per_year_cap = max(1, max_fetch // len(yrs))

    recs, seen = [], set()
    for y in yrs:
        yd0 = f"{y}/01/01" if y != int(d0[:4]) else d0
        yd1 = f"{y}/12/31" if y != int(d1[:4]) else d1
        yc, we, qk = esearch_history(term, yd0, yd1)
        if yc == 0:
            continue
        cap = per_year_cap
        got = fetch_records(we, qk, yc, cap, cfg.SLEEP_SEC)
        for r in got:                       # dedupe by PMID (epub/print straddle)
            if r["pmid"] not in seen:
                seen.add(r["pmid"]); recs.append(r)
        print(f"  year {y}: {yc:,} records, fetched {len(got)} (cumulative {len(recs)})")
        if max_fetch and len(recs) >= max_fetch:
            break
    print(f"  parsed {len(recs)} unique records")

    dois = sorted({(r["doi"] or "").strip().lower() for r in recs if r["doi"]})
    print(f"  DOIs: {len(dois)} -> OpenAlex")
    doi2cit = openalex_citations(dois, sess, cfg.SLEEP_SEC)

    for r in recs:
        d = (r["doi"] or "").strip().lower()
        cit, oa = doi2cit.get(d, (0, None))
        r["citations"] = int(cit)
        r["openalex_id"] = oa or ""

    df = pd.DataFrame(recs)
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0).astype(int)
    df["abstract_present"] = df["abstract"].str.len().gt(0)
    df = df.sort_values(["citations", "year"], ascending=[False, True]).reset_index(drop=True)

    # full set (without bulky abstract column to keep CSV lean)
    all_csv = cfg.CITED_DIR / f"{gid}_all.csv"
    df.drop(columns=["abstract"]).to_csv(all_csv, index=False, encoding="utf-8")

    # full corpus WITH abstracts (feeds topic analysis + abstracts library)
    corpus_csv = cfg.CITED_DIR / f"{gid}_corpus.csv"
    df[["pmid", "doi", "year", "journal", "citations", "title", "abstract"]].to_csv(
        corpus_csv, index=False, encoding="utf-8")

    top = df.head(top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top_csv = cfg.CITED_DIR / f"{gid}_top.csv"
    top.drop(columns=["abstract"]).to_csv(top_csv, index=False, encoding="utf-8")

    # abstract text files for the top-N
    adir = cfg.ABSTRACTS_DIR / gid
    adir.mkdir(parents=True, exist_ok=True)
    for _, row in top.iterrows():
        fn = adir / f"{int(row['rank']):03d}_{row['pmid']}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"PMID: {row['pmid']}\nDOI: {row['doi']}\n")
            f.write(f"Year: {row['year']}\nJournal: {row['journal']}\n")
            f.write(f"Citations (OpenAlex): {row['citations']}\n")
            f.write(f"Title: {row['title']}\n\nAbstract:\n{row['abstract']}\n")
    print(f"  saved: {all_csv.name}, {top_csv.name}, {len(top)} abstracts -> {adir}")
    return {"group": gid, "records": len(recs), "dois": len(dois),
            "with_citations": int((df["citations"] > 0).sum()),
            "top_n": len(top)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["G3", "G5", "G2"],
                    help="group ids to process (G1 excluded by default — counts only)")
    ap.add_argument("--max-fetch", type=int, default=None,
                    help="cap records fetched per group (use for G4 ~104k)")
    ap.add_argument("--top-n", type=int, default=None,
                    help="override top-N (else 20, or 50 for G3/G5)")
    args = ap.parse_args()

    sess = session_with_retry()
    summary = []
    for gid in args.groups:
        top_n = args.top_n or (cfg.TOP_N_DEEP if gid in cfg.DEEP_GROUPS else cfg.TOP_N_DEFAULT)
        # sensible default cap for the huge recent group
        max_fetch = args.max_fetch
        if max_fetch is None and gid == "G4":
            max_fetch = 3000
        summary.append(process_group(gid, max_fetch, top_n, sess))

    print("\n==== SUMMARY ====")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
