"""
finalize_manuscript.py — Merge the 5 literature-mining inserts into a COPY of the
v2 manuscript (Variant A for Conclusions). Never modifies the original.

Output: reports/HIF1A_COVID_Manuscript_v3_merged_DRAFT.docx
        reports/main_text_inserts/new_references_vancouver.txt
"""

import shutil
import sys
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modified_framework"))
import covid_config as cfg

SRC = Path("/home/marko-b2/COVID_Literature_Mining/manuscripts_ref/HIF1A_COVID_Manuscript_v2.docx")
OUT = cfg.REPORTS_DIR / "HIF1A_COVID_Manuscript_v3_merged_DRAFT.docx"

INTRO_A = (
    "Despite the pandemic's massive scientific output — with 496,768 COVID-19 publications "
    "indexed in PubMed between January 2019 and June 2026 — the application of artificial "
    "intelligence and machine learning to COVID-19 research has remained relatively limited, "
    "comprising only 2.7% of all COVID-19 publications across this period. This reflects in "
    "part the temporal sequence of the pandemic: modern AI methodologies, particularly large "
    "language models and advanced multi-modal architectures, achieved transformative "
    "capabilities primarily after the acute pandemic phase (2022 onwards), and indeed the AI "
    "share of COVID-19 publications roughly tripled from 1.5% in 2020 to 4.5% by 2026. "
    "Furthermore, host genetic factors in the hypoxia-inducible pathway — the focus of the "
    "present work — have received comparatively limited attention: only 139 HIF1A-related "
    "COVID-19 publications appear in PubMed across the same period, with virtually none "
    "integrating populational, transcriptomic, clinical, and host-genetic evidence within a "
    "unified multi-scale framework. Detailed literature mining results supporting these "
    "observations are provided in Supplementary Section S5."
)

INTRO_B = (
    "COVID-19 research itself remains highly active: 104,621 publications were indexed in "
    "PubMed during 2024–2026 alone, with output growing year-on-year across this window. "
    "Recent work has shifted from the acute phase toward long-term consequences — long COVID "
    "and post-acute sequelae, host immunity, and the mental-health and healthcare-system "
    "burden of the pandemic — alongside next-generation vaccine and respiratory-virus "
    "surveillance strategies (Supplementary Section S5)."
)

MM_HEAD = "2.7 Literature mining methodology"
MM_BODY = (
    "To contextualize the present multi-scale HIF1A analysis within the broader COVID-19 "
    "literature, we performed automated bibliometric mining using our previously published "
    "frameworks: the octe-ai-mining-framework (Živanović, 2025a) for PubMed (NCBI Entrez) "
    "literature retrieval and OpenAlex citation analysis, and the bioprinting-ai-mining-"
    "framework (Živanović, 2025b) for visualization utilities. The frameworks were minimally "
    "modified to accommodate COVID-19-specific query strings and date ranges; the original "
    "retrieval and citation-ranking methodology was otherwise preserved. Five parallel PubMed "
    "queries were executed covering: (G1) all COVID-19 publications; (G2) COVID-19 publications "
    "applying artificial intelligence or machine learning; (G3) COVID-19 publications addressing "
    "HIF1A or the hypoxia-inducible factor 1-alpha pathway; (G4) recent (2024–2026) COVID-19 "
    "publications for topic-distribution analysis; and (G5) COVID-19 publications addressing "
    "AI-assisted vaccine design. COVID-19 was defined by the MeSH terms \"COVID-19\" or "
    "\"SARS-CoV-2\" or their occurrence in title/abstract, and the date range was 2019/01/01 "
    "through 2026/06/01. Top-cited papers per group were identified using OpenAlex citation "
    "counts (Priem et al., 2022), and full abstracts of the most-cited papers per group (top 20; "
    "top 50 for the smaller HIF1A and AI-vaccine groups) were retrieved for qualitative review. "
    "Recent-theme structure (group G4) was characterized by TF-IDF term weighting and "
    "unsupervised K-means clustering of abstracts. Detailed search strategy, group-specific "
    "results, and curated abstract summaries are provided in Supplementary Section S5."
)

RES_HEAD = "3.7 Literature context of the multi-scale HIF1A analysis"
RES_BODY = (
    "Bibliometric mining of PubMed between 2019 and 2026 (Figure S5.1, Table S5.1) confirms the "
    "sustained relevance of COVID-19 research, with 104,621 publications indexed during 2024–2026 "
    "alone — predominantly addressing long COVID and post-acute sequelae, host immunity, the "
    "mental-health and healthcare-system burden of the pandemic, and next-generation vaccine and "
    "respiratory-virus surveillance strategies (Figure S5.4, Table S5.3). Within the full "
    "2019–2026 corpus, only 2.7% of COVID-19 publications incorporated artificial intelligence or "
    "machine-learning methodologies (Figure S5.3), with the AI subset showing accelerated growth "
    "after 2022 — rising from 1.5% of annual output in 2020 to 4.5% by 2026. HIF1A-focused "
    "COVID-19 publications constituted only 139 papers across the entire period (47 in 2024–2026), "
    "of which none combined populational, transcriptomic, single-cell, and host-genetic evidence "
    "within a unified predictive framework. This positions the present multi-scale HIF1A study as "
    "both a methodological novelty and a substantive contribution to an under-explored axis of "
    "COVID-19 host biology. The complete literature analysis — full search strategy, growth "
    "curves, AI-ratio trend, recent-topic distribution, and selected abstracts — is provided in "
    "Supplementary Section S5."
)

CONC_BODY = (  # Variant A
    "Looking forward, recent advances illustrate the emerging role of artificial intelligence in "
    "COVID-19 vaccine development. General-purpose AI methods for molecular design — AlphaFold-"
    "driven protein structure prediction (Jumper et al., 2021) and generative protein engineering "
    "with ProteinMPNN (Dauparas et al., 2022) — have already been adapted to SARS-CoV-2: "
    "machine-learning reverse vaccinology has been used to prioritize antigen candidates beyond "
    "the spike protein (Ong et al., 2020), and deep-learning frameworks have been applied to "
    "multi-epitope vaccine design directly from the viral proteome (Yang et al., 2021). In "
    "parallel, algorithmic optimization of mRNA sequence and structure (Zhang et al., 2023) is "
    "improving the stability and immunogenicity of RNA vaccines. These approaches, which became "
    "viable largely after the acute pandemic phase, exemplify the broader methodological shift "
    "that AI now enables in infectious-disease research. Building on this trend and on the "
    "multi-scale HIF1A framework presented here, our future work will focus specifically on "
    "AI-assisted design and stratification strategies for next-generation coronavirus vaccines, "
    "with particular emphasis on integrating host genetic variants (including HIF1A-pathway "
    "variants) with antigen-design optimization."
)

NEW_REFS = [
    "Živanović M. octe-ai-mining-framework (v1.0.1) [Computer software]. Zenodo; 2025. "
    "doi:10.5281/zenodo.17093171",
    "Živanović M. bioprinting-ai-mining-framework [Computer software]. Zenodo; 2025. "
    "https://github.com/zivanovicmkg/bioprinting-ai-mining-framework",
    "Priem J, Piwowar H, Orr R. OpenAlex: A fully-open index of scholarly works, authors, "
    "venues, institutions, and concepts. arXiv:2205.01833; 2022.",
    "Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with "
    "AlphaFold. Nature. 2021;596(7873):583–589. doi:10.1038/s41586-021-03819-2",
    "Dauparas J, Anishchenko I, Bennett N, et al. Robust deep learning–based protein sequence "
    "design using ProteinMPNN. Science. 2022;378(6615):49–56. doi:10.1126/science.add2187",
    "Ong E, Wong MU, Huffman A, He Y. COVID-19 coronavirus vaccine design using reverse "
    "vaccinology and machine learning. Front Immunol. 2020;11:1581. doi:10.3389/fimmu.2020.01581",
    "Yang Z, Bogdan P, Nazarian S. An in silico deep learning approach to multi-epitope vaccine "
    "design: a SARS-CoV-2 case study. Sci Rep. 2021;11:3238. doi:10.1038/s41598-021-81749-9",
    "Zhang H, Zhang L, Lin A, et al. Algorithm for optimized mRNA design improves stability and "
    "immunogenicity. Nature. 2023;621(7978):396–403. doi:10.1038/s41586-023-06127-z",
]


def find_anchor(doc, predicate):
    for p in doc.paragraphs:
        if predicate(p):
            return p
    raise RuntimeError("anchor not found")


def main():
    shutil.copyfile(SRC, OUT)
    doc = docx.Document(str(OUT))

    # 1) Introduction — before "In this paper, we present a multi-scale framework"
    intro_anchor = find_anchor(doc, lambda p: p.text.strip().startswith("In this paper, we present a multi-scale"))
    intro_anchor.insert_paragraph_before(INTRO_A, style="Normal")
    intro_anchor.insert_paragraph_before(INTRO_B, style="Normal")

    # 2) M&M 2.7 — before "3. Results"
    res_h = find_anchor(doc, lambda p: p.text.strip() == "3. Results" and p.style.name.startswith("Heading"))
    res_h.insert_paragraph_before(MM_HEAD, style="Heading 2")
    res_h.insert_paragraph_before(MM_BODY, style="Normal")

    # 3) Results 3.7 — before "4. Discussion"
    disc_h = find_anchor(doc, lambda p: p.text.strip() == "4. Discussion" and p.style.name.startswith("Heading"))
    disc_h.insert_paragraph_before(RES_HEAD, style="Heading 2")
    disc_h.insert_paragraph_before(RES_BODY, style="Normal")

    # 4) Conclusions — before "7. Acknowledgements and Funding"
    ack_h = find_anchor(doc, lambda p: p.text.strip().startswith("7. Acknowledgements") and p.style.name.startswith("Heading"))
    ack_h.insert_paragraph_before(CONC_BODY, style="Normal")

    doc.save(str(OUT))

    refs_out = cfg.REPORTS_DIR / "main_text_inserts" / "new_references_vancouver.txt"
    refs_out.write_text(
        "New references introduced by the literature-mining inserts (Vancouver style).\n"
        "To be merged into the manuscript's numbered reference list during final assembly.\n\n"
        + "\n".join(f"- {r}" for r in NEW_REFS), encoding="utf-8")

    print(f"Saved merged manuscript: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"Saved new-refs list: {refs_out}")


if __name__ == "__main__":
    main()
