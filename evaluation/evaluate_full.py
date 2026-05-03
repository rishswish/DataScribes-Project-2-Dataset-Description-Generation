"""
Full evaluation of LLM-generated dataset descriptions.

Covers all three evaluation categories from the project proposal:
  1. Text Similarity Metrics  — ROUGE-1, ROUGE-2, ROUGE-L, METEOR, BERTScore
  2. Retrieval Evaluation     — NDCG@5 and NDCG@10 via TF-IDF simulation
  3. Qualitative Evaluation   — manual scoring spreadsheet (10-dataset sample)

Reads:  ~/Downloads/generated_descriptions_all_200.csv  (or data/ copy)
Writes: data/text_similarity_results.csv
        data/retrieval_results.csv
        data/manual_evaluation_sample.csv
        data/manual_evaluation_sample.xlsx
        data/full_evaluation_summary.csv  (one-page summary)
"""

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

CSV_CANDIDATES = [
    Path("data/generated_descriptions_all_200.csv"),
    Path.home() / "Downloads/generated_descriptions_all_200.csv",
]

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_csv() -> Path:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"CSV not found. Tried: {CSV_CANDIDATES}")


def clean(text) -> str:
    if not text or (isinstance(text, float) and math.isnan(text)):
        return ""
    return str(text).strip()


def dcg(relevances: list) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg(relevances: list) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(relevances) / ideal_dcg if ideal_dcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Similarity  (ROUGE + METEOR + BERTScore)
# ─────────────────────────────────────────────────────────────────────────────

def compute_text_similarity(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/3] Computing text similarity metrics (ROUGE, METEOR, BERTScore)...")

    scorable = df[df["generated_description"].notna() & df["original_description"].notna()].copy()
    scorable = scorable[scorable["original_description"].str.strip() != ""]
    print(f"      {len(scorable)} rows have both original and generated descriptions")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1, rouge2, rougeL, meteors = [], [], [], []

    for _, row in scorable.iterrows():
        ref = clean(row["original_description"])
        hyp = clean(row["generated_description"])

        scores = scorer.score(ref, hyp)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)

        ref_tokens = word_tokenize(ref.lower())
        hyp_tokens = word_tokenize(hyp.lower())
        try:
            m = meteor_score.meteor_score([ref_tokens], hyp_tokens)
        except Exception:
            m = 0.0
        meteors.append(m)

    scorable["rouge1"] = rouge1
    scorable["rouge2"] = rouge2
    scorable["rougeL"] = rougeL
    scorable["meteor"] = meteors

    # BERTScore (batched — can take ~30 seconds first run due to model download)
    print("      Running BERTScore (downloads ~400 MB model on first run — please wait)...")
    from bert_score import score as bertscore
    refs = scorable["original_description"].tolist()
    hyps = scorable["generated_description"].tolist()
    P, R, F1 = bertscore(hyps, refs, lang="en", verbose=False)
    scorable["bertscore_precision"] = P.numpy().tolist()
    scorable["bertscore_recall"] = R.numpy().tolist()
    scorable["bertscore_f1"] = F1.numpy().tolist()

    out_cols = [
        "dataset_id", "source", "title",
        "rouge1", "rouge2", "rougeL", "meteor",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
    ]
    result = scorable[out_cols].round(4)
    result.to_csv(OUTPUT_DIR / "text_similarity_results.csv", index=False)
    print(f"      Saved to data/text_similarity_results.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Evaluation  (NDCG via TF-IDF simulation)
# ─────────────────────────────────────────────────────────────────────────────

QUERIES = [
    ("NYC taxi trip data",          ["nyc", "taxi", "trip", "cab", "fare", "ride"]),
    ("weather wind speed dataset",  ["weather", "wind", "speed", "temperature", "climate", "forecast"]),
    ("housing property values",     ["housing", "property", "real estate", "home", "price", "value", "zoning"]),
    ("crime incident reports",      ["crime", "incident", "arrest", "complaint", "police", "felony"]),
    ("public health disease data",  ["health", "disease", "hospital", "covid", "vaccination", "death"]),
    ("school education enrollment", ["school", "education", "student", "enrollment", "grade", "teacher"]),
    ("traffic collision accidents",  ["traffic", "collision", "accident", "crash", "vehicle", "road"]),
    ("restaurant food inspection",  ["restaurant", "food", "inspection", "violation", "health", "permit"]),
]


def retrieval_score_for_query(query: str, keywords: list, corpus_texts: list) -> list:
    """
    Compute relevance of each document to the query using TF-IDF cosine similarity.
    Returns a list of (index, similarity) pairs sorted by similarity desc.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus_texts + [query])
    except ValueError:
        return [(i, 0.0) for i in range(len(corpus_texts))]

    doc_vecs = matrix[:-1]
    query_vec = matrix[-1]
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return ranked


def keyword_relevance(text: str, keywords: list) -> int:
    """Binary relevance: 1 if any keyword appears in text, else 0."""
    t = text.lower()
    return int(any(kw in t for kw in keywords))


def compute_retrieval_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/3] Running retrieval evaluation (TF-IDF + NDCG)...")

    success = df[df["generated_description"].notna()].copy()

    original_corpus = [
        clean(r["title"]) + " " + clean(r["original_description"])
        for _, r in success.iterrows()
    ]
    generated_corpus = [
        clean(r["title"]) + " " + clean(r["generated_description"])
        for _, r in success.iterrows()
    ]

    records = []
    for query, keywords in QUERIES:
        orig_ranked = retrieval_score_for_query(query, keywords, original_corpus)
        gen_ranked = retrieval_score_for_query(query, keywords, generated_corpus)

        for k in [5, 10]:
            orig_top = [keyword_relevance(original_corpus[i], keywords) for i, _ in orig_ranked[:k]]
            gen_top = [keyword_relevance(generated_corpus[i], keywords) for i, _ in gen_ranked[:k]]

            records.append({
                "query": query,
                "k": k,
                "ndcg_original": round(ndcg(orig_top), 4),
                "ndcg_generated": round(ndcg(gen_top), 4),
                "delta": round(ndcg(gen_top) - ndcg(orig_top), 4),
            })

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "retrieval_results.csv", index=False)
    print(f"      Saved to data/retrieval_results.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Qualitative Evaluation  (manual scoring spreadsheet)
# ─────────────────────────────────────────────────────────────────────────────

def build_manual_eval_sample(df: pd.DataFrame):
    print("\n[3/3] Building manual evaluation sample...")

    success = df[df["generated_description"].notna()].copy()

    nyc = success[success["source"] == "nyc_open_data"].head(5)
    gov = success[success["source"] == "data_gov"].head(5)
    sample = pd.concat([nyc, gov], ignore_index=True)

    sample = sample[[
        "dataset_id", "source", "title",
        "original_description", "generated_description", "generation_model",
    ]].copy()

    sample["manual_score_readability"] = ""
    sample["manual_score_accuracy"] = ""
    sample["manual_score_usefulness"] = ""
    sample["manual_notes"] = ""

    sample.to_csv(OUTPUT_DIR / "manual_evaluation_sample.csv", index=False)
    try:
        sample.to_excel(OUTPUT_DIR / "manual_evaluation_sample.xlsx", index=False)
        print(f"      Saved CSV and XLSX to data/")
    except Exception:
        print(f"      Saved CSV to data/ (XLSX skipped — openpyxl not available)")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(sim: pd.DataFrame, ret: pd.DataFrame, df: pd.DataFrame):
    print("\n" + "=" * 62)
    print("FULL EVALUATION SUMMARY")
    print("=" * 62)

    total = len(df)
    success = df["generated_description"].notna().sum()
    print(f"\n  Datasets processed : {total}")
    print(f"  Successful         : {success} ({100*success/total:.1f}%)")
    print(f"  Failed             : {total - success}")

    print("\n  — Text Similarity (vs. original description) —")
    for src in ["nyc_open_data", "data_gov", "ALL"]:
        sub = sim if src == "ALL" else sim[sim["source"] == src]
        if sub.empty:
            continue
        label = src if src != "ALL" else "overall"
        print(f"\n  [{label}]")
        print(f"    ROUGE-1  : {sub['rouge1'].mean():.4f}")
        print(f"    ROUGE-2  : {sub['rouge2'].mean():.4f}")
        print(f"    ROUGE-L  : {sub['rougeL'].mean():.4f}")
        print(f"    METEOR   : {sub['meteor'].mean():.4f}")
        print(f"    BERTScore F1: {sub['bertscore_f1'].mean():.4f}")

    print("\n  — Retrieval (NDCG) —")
    for k in [5, 10]:
        sub = ret[ret["k"] == k]
        avg_orig = sub["ndcg_original"].mean()
        avg_gen = sub["ndcg_generated"].mean()
        avg_delta = sub["delta"].mean()
        direction = "↑" if avg_delta > 0 else ("↓" if avg_delta < 0 else "→")
        print(f"    NDCG@{k}  original={avg_orig:.4f}  generated={avg_gen:.4f}  delta={avg_delta:+.4f} {direction}")

    print("\n  — Per-query NDCG@10 breakdown —")
    for _, row in ret[ret["k"] == 10].iterrows():
        delta = row["delta"]
        flag = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"    {row['query'][:40]:<40}  orig={row['ndcg_original']:.3f}  gen={row['ndcg_generated']:.3f}  {flag}{abs(delta):.3f}")

    summary_rows = []
    for src in ["nyc_open_data", "data_gov"]:
        sub = sim[sim["source"] == src]
        summary_rows.append({
            "source": src,
            "rouge1": round(sub["rouge1"].mean(), 4),
            "rouge2": round(sub["rouge2"].mean(), 4),
            "rougeL": round(sub["rougeL"].mean(), 4),
            "meteor": round(sub["meteor"].mean(), 4),
            "bertscore_f1": round(sub["bertscore_f1"].mean(), 4),
        })
    summary_rows.append({
        "source": "OVERALL",
        "rouge1": round(sim["rouge1"].mean(), 4),
        "rouge2": round(sim["rouge2"].mean(), 4),
        "rougeL": round(sim["rougeL"].mean(), 4),
        "meteor": round(sim["meteor"].mean(), 4),
        "bertscore_f1": round(sim["bertscore_f1"].mean(), 4),
    })
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "full_evaluation_summary.csv", index=False)
    print(f"\n  Saved full_evaluation_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_path = find_csv()
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows  |  {df['generated_description'].notna().sum()} successful generations")

    sim_results = compute_text_similarity(df)
    ret_results = compute_retrieval_evaluation(df)
    build_manual_eval_sample(df)
    print_summary(sim_results, ret_results, df)


if __name__ == "__main__":
    main()
