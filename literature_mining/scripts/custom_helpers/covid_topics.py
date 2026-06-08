"""
covid_topics.py — Recent-themes (2024-2026) analysis from the G4 abstract corpus.

Input : data/cited_metrics/G4_corpus.csv  (pmid,year,...,title,abstract)
Outputs:
  figures/wordclouds/FigS5_4_wordcloud.png/.pdf
  data/topic_analysis/top_terms.csv          (Table S5.2 — top 30 terms)
  data/topic_analysis/topic_clusters.csv     (Table S5.3 — KMeans topic clusters)
  figures/wordclouds/FigS5_4b_topterms_bar.png

Uses title+abstract text. Domain-generic words (covid, sars, patient, ...) are
stopworded so the *distinctive* recent themes surface.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modified_framework"))
import covid_config as cfg

# Domain-generic terms to suppress (they dominate but carry no theme signal).
DOMAIN_STOP = {
    "covid", "covid-19", "covid19", "sars", "cov", "sars-cov-2", "sarscov2",
    "sars-cov", "coronavirus", "coronaviruses", "pandemic", "ncov", "2019-ncov",
    "disease", "infection", "infections", "infected", "patient", "patients",
    "study", "studies", "result", "results", "method", "methods", "conclusion",
    "conclusions", "background", "objective", "objectives", "analysis", "data",
    "using", "used", "use", "associated", "based", "compared", "including",
    "included", "showed", "found", "may", "also", "however", "among", "group",
    "groups", "level", "levels", "high", "higher", "low", "case", "cases",
    "clinical", "health", "care", "respiratory", "acute", "syndrome", "severe",
    "risk", "factor", "factors", "outcome", "outcomes", "effect", "effects",
    "treatment", "increased", "decreased", "significant", "significantly", "non",
    "two", "one", "three", "new", "p", "ci", "n", "vs", "95",
    # tokenization / HTML artifacts and generic methods verbs
    "sup", "sub", "fig", "figure", "table", "findings", "conducted", "reported",
    "identified", "potential", "time", "times", "years", "year", "age", "aged",
    "research", "participants", "total", "period", "rate", "rates", "number",
    "present", "observed", "performed", "assessed", "evaluated", "examined",
    "associated", "related", "various", "different", "increase", "decrease",
    "due", "within", "across", "overall", "respectively", "compared",
}
# Generic academic vocabulary (not themes) — kept separate for clarity.
ACADEMIC_STOP = {
    "impact", "public", "need", "needs", "support", "strategies", "strategy",
    "aimed", "aim", "aims", "future", "analyzed", "analyse", "analyze", "analysed",
    "particularly", "population", "populations", "individuals", "individual",
    "approach", "approaches", "collected", "introduction", "revealed", "reveal",
    "assess", "assessment", "following", "change", "changes", "role", "roles",
    "response", "responses", "review", "reviews", "model", "models", "evidence",
    "primary", "secondary", "key", "multiple", "demonstrated", "demonstrate",
    "suggest", "suggests", "suggested", "status", "provide", "provides",
    "provided", "common", "essential", "important", "finding", "prevalence",
    "association", "associations", "cross-sectional", "sectional", "cohort",
    "retrospective", "prospective", "survey", "questionnaire", "online", "tool",
    "tools", "framework", "information", "system", "systems", "measure",
    "measures", "test", "tests", "work", "global", "medical", "lower", "effective",
    "context", "insights", "characteristics", "conditions", "control", "limited",
    "considered", "regarding", "addition", "additional", "report", "reports",
    "regression", "logistic", "statistical", "score", "scores", "scale",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
STOP = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP | ACADEMIC_STOP

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")
# strip HTML sup/sub tag remnants and entities before tokenizing
HTML_RE = re.compile(r"</?\s*(sup|sub|i|b|em)\s*>|&[a-z]+;", re.I)
N_CLUSTERS = 6
TOP_TERMS = 30


def load_corpus():
    p = cfg.CITED_DIR / "G4_corpus.csv"
    df = pd.read_csv(p).fillna("")
    raw = (df["title"].astype(str) + ". " + df["abstract"].astype(str)).str.lower()
    df["text"] = raw.map(lambda t: HTML_RE.sub(" ", t))
    df = df[df["abstract"].str.len() > 0].reset_index(drop=True)
    return df


def tokenize(text):
    out = []
    for w in WORD_RE.findall(text):
        w = w.strip("-")               # 'covid-' -> 'covid', 'sars-cov-' -> 'sars-cov'
        if len(w) > 2 and w not in STOP and not w.isdigit():
            out.append(w)
    return out


def build_tfidf(df):
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                          max_df=0.5, min_df=5, ngram_range=(1, 2))
    X = vec.fit_transform(df["text"])
    return vec, X


def term_weights(vec, X):
    """Aggregate TF-IDF weight per term (distinctiveness-weighted, not raw freq)."""
    import numpy as np
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    return {t: float(s) for t, s in zip(terms, sums)}


def top_terms(df, weights):
    # document-frequency for the readable % column, ranked by tf-idf weight
    docfreq = Counter()
    for t in df["text"]:
        docfreq.update(set(tokenize(t)))
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:TOP_TERMS]
    rows = [{"rank": i + 1, "term": w, "tfidf_weight": round(s, 2),
             "doc_count": docfreq.get(w, 0),
             "doc_pct": round(100 * docfreq.get(w, 0) / len(df), 1)}
            for i, (w, s) in enumerate(ranked)]
    out = pd.DataFrame(rows)
    out.to_csv(cfg.TOPIC_DIR / "top_terms.csv", index=False)
    return out


def wordcloud_fig(weights):
    top = dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:150])
    wc = WordCloud(width=1400, height=750, background_color="white",
                   colormap="viridis", max_words=120, prefer_horizontal=0.9)
    wc.generate_from_frequencies(top)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Figure S5.4 — Dominant terms in COVID-19 literature, 2024–2026 "
                 "(domain-generic words removed)", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIGURES_DIR / "wordclouds" / f"FigS5_4_wordcloud.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def topterms_bar(tt):
    fig, ax = plt.subplots(figsize=(7, 8))
    d = tt.iloc[::-1]
    ax.barh(d["term"], d["doc_pct"], color="#3b6ea5")
    ax.set_xlabel("% of 2024–2026 abstracts containing term")
    ax.set_title("Figure S5.4b — Top 30 recent terms")
    fig.savefig(cfg.FIGURES_DIR / "wordclouds" / "FigS5_4b_topterms_bar.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def cluster_topics(df, vec, X):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    rows = []
    for c in range(N_CLUSTERS):
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:10]
        keywords = ", ".join(terms[i] for i in top_idx)
        size = int((labels == c).sum())
        rows.append({"cluster": c + 1, "n_docs": size,
                     "pct": round(100 * size / len(df), 1), "top_terms": keywords})
    out = pd.DataFrame(rows).sort_values("n_docs", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.to_csv(cfg.TOPIC_DIR / "topic_clusters.csv", index=False)
    return out


def main():
    df = load_corpus()
    print(f"Corpus: {len(df)} abstracts (2024-2026)")
    vec, X = build_tfidf(df)
    weights = term_weights(vec, X)
    tt = top_terms(df, weights)
    wordcloud_fig(weights)
    topterms_bar(tt)
    print("\nTop 15 terms (by TF-IDF weight):")
    print(tt.head(15).to_string(index=False))
    clusters = cluster_topics(df, vec, X)
    print("\nTopic clusters:")
    print(clusters[["rank", "n_docs", "pct", "top_terms"]].to_string(index=False))
    print("\nSaved: top_terms.csv, topic_clusters.csv, FigS5_4 wordcloud + bar")


if __name__ == "__main__":
    main()
