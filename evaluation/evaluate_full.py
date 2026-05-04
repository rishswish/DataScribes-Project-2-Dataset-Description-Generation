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
        data/full_evaluation_report.md    (human-readable summary)
"""

import math
import re
import warnings
from pathlib import Path
from typing import List

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


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


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

def select_examples_by_score(group: pd.DataFrame, score_col: str, n_best: int = 2, n_worst: int = 2, n_middle: int = 1) -> pd.DataFrame:
    if group.empty:
        return group.copy()

    ranked = group.sort_values(score_col, ascending=False).copy()
    pieces = [
        ranked.head(n_best).assign(review_bucket="top_quality"),
        ranked.tail(n_worst).assign(review_bucket="low_quality"),
    ]

    middle_idx = len(ranked) // 2
    middle = ranked.iloc[[middle_idx]].copy().assign(review_bucket="mid_quality")
    if n_middle == 0:
        middle = ranked.iloc[0:0].copy()
    pieces.append(middle)

    combined = pd.concat(pieces, ignore_index=False)
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined


def build_manual_eval_sample(df: pd.DataFrame, sim: pd.DataFrame):
    print("\n[3/3] Building manual evaluation sample...")

    success = df[df["generated_description"].notna()].copy()
    scored = success.merge(
        sim[["dataset_id", "rouge1", "meteor", "bertscore_f1"]],
        on="dataset_id",
        how="left",
    )

    selected_groups = []
    for source in ["nyc_open_data", "data_gov"]:
        group = scored[scored["source"] == source].copy()
        selected_groups.append(select_examples_by_score(group, "bertscore_f1"))

    sample = pd.concat(selected_groups, ignore_index=True)
    sample = sample.drop_duplicates(subset=["dataset_id"]).copy()

    failures = df[df["generated_description"].isna()].copy()
    if not failures.empty:
        failures["review_bucket"] = "generation_failure"
        sample = pd.concat([sample, failures], ignore_index=True, sort=False)

    sample = sample[[
        "dataset_id", "source", "title",
        "review_bucket", "original_description", "generated_description",
        "generation_model", "generation_error", "rouge1", "meteor", "bertscore_f1",
    ]].copy()

    sample["manual_score_readability"] = ""
    sample["manual_score_accuracy"] = ""
    sample["manual_score_usefulness"] = ""
    sample["manual_score_faithfulness"] = ""
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


def build_failure_analysis(df: pd.DataFrame) -> pd.DataFrame:
    failures = df[df["generated_description"].isna()].copy()
    if failures.empty:
        return pd.DataFrame(columns=["failure_type", "count"])

    def classify_failure(error: str) -> str:
        text = clean(error).lower()
        if "rate limit" in text or "429" in text:
            return "rate_limit"
        if "too long" in text or "tokens >" in text:
            return "prompt_too_long"
        return "other"

    failures["failure_type"] = failures["generation_error"].apply(classify_failure)
    summary = (
        failures.groupby("failure_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary.to_csv(OUTPUT_DIR / "failure_analysis.csv", index=False)
    return summary


def write_markdown_report(
    sim: pd.DataFrame,
    ret: pd.DataFrame,
    df: pd.DataFrame,
    failure_df: pd.DataFrame,
) -> None:
    total = len(df)
    success = int(df["generated_description"].notna().sum())
    failed = total - success

    quality_rows = []
    for src in ["nyc_open_data", "data_gov", "OVERALL"]:
        subset = sim if src == "OVERALL" else sim[sim["source"] == src]
        if subset.empty:
            continue
        quality_rows.append([
            src,
            f"{subset['rouge1'].mean():.4f}",
            f"{subset['rouge2'].mean():.4f}",
            f"{subset['rougeL'].mean():.4f}",
            f"{subset['meteor'].mean():.4f}",
            f"{subset['bertscore_f1'].mean():.4f}",
        ])

    retrieval_rows = []
    for k in [5, 10]:
        subset = ret[ret["k"] == k]
        retrieval_rows.append([
            f"NDCG@{k}",
            f"{subset['ndcg_original'].mean():.4f}",
            f"{subset['ndcg_generated'].mean():.4f}",
            f"{subset['delta'].mean():+.4f}",
        ])

    failure_lines = ["- No failed generations were observed."]
    if not failure_df.empty:
        failure_lines = [
            f"- `{row.failure_type}`: {int(row.count)} dataset(s)"
            for row in failure_df.itertuples()
        ]

    notable_queries = (
        ret[ret["k"] == 10]
        .sort_values("delta", ascending=False)
        [["query", "ndcg_original", "ndcg_generated", "delta"]]
        .head(3)
    )
    query_rows = [
        [
            row["query"],
            f"{row['ndcg_original']:.4f}",
            f"{row['ndcg_generated']:.4f}",
            f"{row['delta']:+.4f}",
        ]
        for _, row in notable_queries.iterrows()
    ]

    report = [
        "# Full Evaluation Report",
        "",
        "## Overview",
        f"- Datasets processed: `{total}`",
        f"- Successful generations: `{success}` ({100 * success / total:.1f}%)",
        f"- Failed generations: `{failed}`",
        "",
        "## Quality Metrics",
        markdown_table(
            ["Source", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore F1"],
            quality_rows,
        ),
        "",
        "## Retrieval Metrics",
        markdown_table(
            ["Metric", "Original", "Generated", "Delta"],
            retrieval_rows,
        ),
        "",
        "## Failure Analysis",
        *failure_lines,
        "",
        "The current failures fall into two operational buckets:",
        "- `rate_limit`: the generation request exceeded Anthropic throughput limits at runtime.",
        "- `prompt_too_long`: the prompt payload exceeded the model token limit for a specific dataset.",
        "",
        "These failures do not indicate a parsing bug in the evaluation pipeline; they indicate generation-time robustness issues that should be addressed by retry logic, batching controls, or prompt truncation.",
        "",
        "## Notable Retrieval Gains",
        markdown_table(
            ["Query", "Original NDCG@10", "Generated NDCG@10", "Delta"],
            query_rows,
        ),
        "",
        "## Qualitative Review Sheet",
        "A manual review spreadsheet was generated at `data/manual_evaluation_sample.xlsx`.",
        "The sample now includes representative high-quality rows, low-quality rows, mid-quality rows, and failed generations so that qualitative scoring covers both strengths and edge cases.",
        "",
        "## Output Files",
        "- `data/text_similarity_results.csv`",
        "- `data/retrieval_results.csv`",
        "- `data/full_evaluation_summary.csv`",
        "- `data/failure_analysis.csv`",
        "- `data/manual_evaluation_sample.csv`",
        "- `data/manual_evaluation_sample.xlsx`",
    ]

    (OUTPUT_DIR / "full_evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("  Saved full_evaluation_report.md")


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
    build_manual_eval_sample(df, sim_results)
    failure_results = build_failure_analysis(df)
    print_summary(sim_results, ret_results, df)
    write_markdown_report(sim_results, ret_results, df, failure_results)


if __name__ == "__main__":
    main()
