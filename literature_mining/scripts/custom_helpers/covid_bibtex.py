"""
covid_bibtex.py — BibTeX + Vancouver references from top-cited DOIs.

For each group's top-cited table, fetch a BibTeX record via CrossRef content
negotiation (https://api.crossref.org/works/{doi}/transform/application/x-bibtex)
and assemble a Vancouver-style numbered reference list.

Outputs:
  bibliography/bibtex/covid_lit.bib          (all groups, deduped by DOI)
  bibliography/by_group/<G>.bib              (per group)
  reports/references_vancouver.txt           (numbered Vancouver list)
  bibliography/by_group/<G>_refs.csv         (rank,key,citation)

Also emits the three fixed AI-vaccine method references named in the plan
(AlphaFold / LinearDesign / ProteinMPNN) for the Conclusions paragraph.
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modified_framework"))
import covid_config as cfg

GROUPS = ["G2", "G3", "G5"]   # main-text / S5 citable groups (not the 13k/104k bulk)
CROSSREF_BIBTEX = "https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
CROSSREF_JSON = "https://api.crossref.org/works/{doi}"

# Fixed references from the plan's Conclusions paragraph (goal c).
FIXED_REFS = {
    "jumper2021alphafold": "10.1038/s41586-021-03819-2",   # AlphaFold
    "zhang2023lineardesign": "10.1038/s41586-023-06127-z",  # LinearDesign (optimized mRNA design)
    "dauparas2022proteinmpnn": "10.1126/science.add2187",   # ProteinMPNN
}


def session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=0.7, status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": f"CovidLitMining/1.0 (mailto:{cfg.OPENALEX_MAILTO})"})
    return s


def fetch_bibtex(sess, doi):
    try:
        r = sess.get(CROSSREF_BIBTEX.format(doi=doi), timeout=30)
        if r.status_code == 200 and "@" in r.text:
            return r.text.strip()
    except requests.RequestException:
        pass
    return None


def fetch_meta(sess, doi):
    try:
        r = sess.get(CROSSREF_JSON.format(doi=doi), timeout=30)
        if r.status_code == 200:
            return r.json().get("message", {})
    except requests.RequestException:
        pass
    return {}


def vancouver(meta, doi):
    """Build a Vancouver-style citation string from CrossRef metadata."""
    authors = meta.get("author", []) or []
    names = []
    for a in authors[:6]:
        fam = a.get("family", "")
        given = a.get("given", "")
        initials = "".join(p[0] for p in re.split(r"[\s\-]+", given) if p)
        if fam:
            names.append(f"{fam} {initials}".strip())
    auth = ", ".join(names)
    if len(authors) > 6:
        auth += ", et al"
    title = (meta.get("title") or [""])[0].rstrip(".")
    journal = (meta.get("short-container-title") or meta.get("container-title") or [""])[0]
    year = ""
    for k in ("published-print", "published-online", "published", "issued"):
        dp = meta.get(k, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            year = dp[0][0]
            break
    vol = meta.get("volume", "")
    issue = meta.get("issue", "")
    pages = meta.get("page", "")
    cite = f"{auth}. {title}. {journal}. {year}"
    if vol:
        cite += f";{vol}"
        if issue:
            cite += f"({issue})"
        if pages:
            cite += f":{pages}"
    cite += f". doi:{doi}"
    return re.sub(r"\s+", " ", cite).strip()


def rekey(bibtex, key):
    """Replace the cite key inside an @article{key, ...} entry."""
    return re.sub(r"(@\w+\{)[^,]+,", rf"\1{key},", bibtex, count=1)


def main():
    sess = session()
    all_entries, vanc_lines = {}, []
    ref_no = 0

    for g in GROUPS:
        top = cfg.CITED_DIR / f"{g}_top.csv"
        if not top.exists():
            print(f"  skip {g} (no {top.name})")
            continue
        df = pd.read_csv(top)
        group_entries, group_rows = [], []
        for _, row in df.iterrows():
            doi = str(row.get("doi", "")).strip()
            if not doi or doi.lower() == "nan":
                continue
            key = f"{g}_{int(row['rank']):02d}"
            bib = fetch_bibtex(sess, doi)
            meta = fetch_meta(sess, doi)
            time.sleep(0.15)
            if bib:
                bib = rekey(bib, key)
                group_entries.append(bib)
                all_entries[doi.lower()] = bib
            van = vancouver(meta, doi) if meta else f"[{doi}]"
            ref_no += 1
            vanc_lines.append(f"{ref_no}. {van}")
            group_rows.append({"rank": int(row["rank"]), "key": key,
                               "citations": row.get("citations", ""), "vancouver": van})
        # per-group outputs
        (cfg.BIB_DIR / "by_group").mkdir(parents=True, exist_ok=True)
        with open(cfg.BIB_DIR / "by_group" / f"{g}.bib", "w", encoding="utf-8") as f:
            f.write("\n\n".join(group_entries))
        pd.DataFrame(group_rows).to_csv(cfg.BIB_DIR / "by_group" / f"{g}_refs.csv", index=False)
        print(f"  {g}: {len(group_entries)} bibtex / {len(group_rows)} refs")

    # fixed AI-vaccine refs
    fixed_entries = []
    for key, doi in FIXED_REFS.items():
        bib = fetch_bibtex(sess, doi)
        time.sleep(0.15)
        if bib:
            fixed_entries.append(rekey(bib, key))
            all_entries[doi.lower()] = all_entries.get(doi.lower(), rekey(bib, key))
    with open(cfg.BIB_DIR / "by_group" / "fixed_aivaccine.bib", "w", encoding="utf-8") as f:
        f.write("\n\n".join(fixed_entries))

    # combined dedup bib
    with open(cfg.BIB_DIR / "bibtex" / "covid_lit.bib", "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_entries.values()))
    with open(cfg.REPORTS_DIR / "references_vancouver.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vanc_lines))
    print(f"\nCombined: {len(all_entries)} unique bibtex entries; "
          f"{len(vanc_lines)} Vancouver refs.")


if __name__ == "__main__":
    main()
