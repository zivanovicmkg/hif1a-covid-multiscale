"""
covid_config.py — Shared configuration for the HIF1A-COVID literature mining.

Adapted (minimally) from:
  - octe-ai-mining-framework  (Zivanovic, 2025a; Zenodo 10.5281/zenodo.17093171)
  - bioprinting-ai-mining-framework (Zivanovic, 2025b)

Only query strings, paths, and NCBI/OpenAlex credentials are changed from the
original frameworks; the retrieval methodology is preserved.

Query group definitions follow Literature_Mining_Plan_REVISED.docx, Table 1 (G1-G5).
"""

import os
from pathlib import Path

# ===================== CREDENTIALS =====================
# NCBI requires an email; the API key (set in ~/.bashrc) lifts the rate limit
# from 3 -> 10 requests/sec.
ENTREZ_EMAIL = "marko.zivanovic@uni.kg.ac.rs"
OPENALEX_MAILTO = "marko.zivanovic@uni.kg.ac.rs"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")  # None if unset -> falls back to 3 req/s

# ===================== SCOPE =====================
# Full analysis window (plan: 2019/01/01 - 2026/06/01).
YEAR_START = 2019
YEAR_END = 2026
YEARS = list(range(YEAR_START, YEAR_END + 1))

# Exact date bounds for PubMed [PDAT] range filters.
DATE_START = "2019/01/01"
DATE_END = "2026/06/01"

# Recent-themes sub-window (plan: goal b, group G4).
RECENT_START = "2024/01/01"
RECENT_END = "2026/06/01"

# Polite delay between API calls. With an API key NCBI allows 10 req/s, so 0.12s
# is safe; without a key keep >=0.34s.
SLEEP_SEC = 0.12 if NCBI_API_KEY else 0.34

# ---------------------------------------------------------------------------
# METHODOLOGY SWITCHES (flagged to Marko — these affect manuscript numbers).
#
# The plan's Table 1 query strings are bare (no publication-type or language
# filter). Set these True only after Marko confirms; default False = matches the
# approved plan verbatim, so the reported [TOTAL]/[X]/[HIF_N] trace directly to it.
APPLY_PUBTYPE_FILTER = False   # adds: AND (review[pt] OR journal article[pt])
APPLY_ENGLISH_FILTER = False   # adds: AND english[la]
# ---------------------------------------------------------------------------

PUBTYPE_FILTER = '(review[Publication Type] OR "journal article"[Publication Type])'
ENGLISH_FILTER = "english[Language]"


def _decorate(term: str) -> str:
    """Apply optional pubtype/language filters to a base query."""
    parts = [f"({term})"]
    if APPLY_PUBTYPE_FILTER:
        parts.append(PUBTYPE_FILTER)
    if APPLY_ENGLISH_FILTER:
        parts.append(ENGLISH_FILTER)
    return " AND ".join(parts)


# ===================== QUERY GROUPS (plan Table 1) =====================
# COVID base — SHARED by all groups so the AI ratio (G2/G1) and trend curves use
# an identical denominator/numerator base. Marko confirmed "MeSH OR tiab" (captures
# recent 2024-2026 papers MeSH hasn't indexed yet). Validated N(G1)=496,768.
COVID_CORE = '("COVID-19"[MeSH] OR "SARS-CoV-2"[MeSH] OR "COVID-19"[tiab] OR "SARS-CoV-2"[tiab])'

AI_TERMS = '("artificial intelligence" OR "machine learning" OR "deep learning")'
HIF_TERMS = ('("HIF1A" OR "HIF-1alpha" OR "HIF-1 alpha" OR '
             '"hypoxia inducible factor 1 alpha" OR "hypoxia-inducible factor 1-alpha")')
AIVAX_TERMS = ('("artificial intelligence" OR "machine learning" OR "deep learning" OR '
               '"AlphaFold" OR "computational design" OR "generative")')

# Base (undecorated) query strings, keyed by group id.
_BASE_QUERIES = {
    # G1: COVID total (baseline) — validated N=496,768
    "G1": COVID_CORE,
    # G2: COVID + AI/ML (key claim a) — validated N=13,545; ratio vs G1 = 2.73%
    "G2": f"{COVID_CORE} AND {AI_TERMS}",
    # G3: COVID + HIF1A (niche) — validated N~140; full screening, not just top-cited
    "G3": f"{COVID_CORE} AND {HIF_TERMS}",
    # G4: Recent COVID 2024-2026 (themes b) — validated N=105,889
    #     Date range handled separately at search time (RECENT_START/RECENT_END).
    "G4": COVID_CORE,
    # G5: COVID + AI vaccines (future direction c) — validated N~1,106
    "G5": f"{COVID_CORE} AND \"vaccine\" AND {AIVAX_TERMS}",
}

# Human-readable labels for figures/tables.
GROUP_LABELS = {
    "G1": "COVID-19 (all)",
    "G2": "COVID-19 + AI/ML",
    "G3": "COVID-19 + HIF1A",
    "G4": "COVID-19 recent (2024-2026)",
    "G5": "COVID-19 + AI vaccines",
}

# Groups that get full screening / larger top-N (small niche groups).
DEEP_GROUPS = {"G3", "G5"}
TOP_N_DEFAULT = 20
TOP_N_DEEP = 50


def queries(decorated: bool = True) -> dict:
    """Return {group_id: pubmed_query_string}. decorated applies optional filters."""
    if not decorated:
        return dict(_BASE_QUERIES)
    return {gid: _decorate(q) for gid, q in _BASE_QUERIES.items()}


def date_bounds(group_id: str):
    """Return (start, end) PDAT bounds for a group. G4 uses the recent window."""
    if group_id == "G4":
        return RECENT_START, RECENT_END
    return DATE_START, DATE_END


# ===================== OUTPUT PATHS =====================
PROJECT_ROOT = Path("/home/marko-b2/COVID_Literature_Mining")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_searches"
CITED_DIR = DATA_DIR / "cited_metrics"
ABSTRACTS_DIR = DATA_DIR / "abstracts"
TOPIC_DIR = DATA_DIR / "topic_analysis"
PRISMA_DIR = DATA_DIR / "prisma"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
BIB_DIR = PROJECT_ROOT / "bibliography"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (RAW_DIR, CITED_DIR, ABSTRACTS_DIR, TOPIC_DIR, PRISMA_DIR,
           FIGURES_DIR, REPORTS_DIR, BIB_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def configure_entrez():
    """Apply NCBI Entrez credentials. Call once at the top of any script."""
    from Bio import Entrez
    Entrez.email = ENTREZ_EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY
    return Entrez
