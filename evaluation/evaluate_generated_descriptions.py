import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "which", "with", "within", "into", "about", "using",
    "used", "use", "can", "may", "will", "data", "dataset", "datasets",
    "new", "york", "city", "nyc", "gov", "open", "public",
}


def safe_import_meteor():
    try:
        from nltk.translate.meteor_score import meteor_score
        return meteor_score, None
    except Exception as exc:
        return None, str(exc)


def safe_import_bertscore():
    try:
        from bert_score import score
        return score, None
    except Exception as exc:
        return None, str(exc)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().replace("\n", " ")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(reference: str, candidate: str, n: int) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    ref_ngrams = ngrams(ref_tokens, n)
    cand_ngrams = ngrams(cand_tokens, n)

    if not ref_ngrams or not cand_ngrams:
        return 0.0

    overlap = sum((ref_ngrams & cand_ngrams).values())
    precision = overlap / sum(cand_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0

    prev = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        curr = [0]
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0

    overlap = lcs_length(ref_tokens, cand_tokens)
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compression_ratio(reference: str, candidate: str) -> float:
    ref_len = len(reference)
    if ref_len == 0:
        return 0.0
    return len(candidate) / ref_len


def meteor_score_value(
    reference: str,
    candidate: str,
    meteor_func,
) -> Optional[float]:
    if meteor_func is None:
        return None
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    try:
        return float(meteor_func([ref_tokens], cand_tokens))
    except Exception:
        return None


def compute_bertscore_values(
    references: List[str],
    candidates: List[str],
    bertscore_func,
) -> List[Optional[float]]:
    if bertscore_func is None:
        return [None] * len(references)
    if not references:
        return []

    try:
        _, _, f1 = bertscore_func(
            candidates,
            references,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return [float(value) for value in f1.tolist()]
    except Exception:
        return [None] * len(references)


def finite_values(series: pd.Series) -> List[float]:
    values: List[float] = []
    for value in series.dropna().tolist():
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def metric_stats(series: pd.Series) -> Dict[str, float]:
    values = finite_values(series)
    if not values:
        return {
            "valid_count": 0,
            "valid_rate": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "zero_rate": float("nan"),
        }

    zero_count = sum(1 for value in values if value == 0.0)
    return {
        "valid_count": len(values),
        "valid_rate": len(values) / len(series) if len(series) else 0.0,
        "mean": sum(values) / len(values),
        "median": float(pd.Series(values).median()),
        "min": min(values),
        "max": max(values),
        "zero_rate": zero_count / len(values),
    }


def extract_salient_terms(text: str, limit: int) -> List[str]:
    counts = Counter(
        token for token in tokenize(text)
        if token not in STOPWORDS and len(token) > 2
    )
    return [token for token, _ in counts.most_common(limit)]


def build_query(title: str, original_description: str) -> str:
    title_terms = [
        token for token in tokenize(title)
        if token not in STOPWORDS and len(token) > 2
    ][:5]
    desc_terms = extract_salient_terms(original_description, limit=5)

    merged = []
    seen = set()
    for token in title_terms + desc_terms:
        if token not in seen:
            seen.add(token)
            merged.append(token)
    return " ".join(merged[:8])


class BM25Index:
    def __init__(self, docs: List[str]):
        self.docs = [tokenize(doc) for doc in docs]
        self.doc_freq = Counter()
        self.doc_lengths = [len(doc) for doc in self.docs]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = []

        for doc in self.docs:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freq[term] += 1

    def idf(self, term: str) -> float:
        doc_count = len(self.docs)
        freq = self.doc_freq.get(term, 0)
        if doc_count == 0:
            return 0.0
        return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        terms = tokenize(query)
        tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0

        for term in terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / self.avg_doc_len) if self.avg_doc_len else 1.0
            score += self.idf(term) * (numerator / denominator)
        return score

    def rank(self, query: str) -> List[int]:
        scores = [(idx, self.score(query, idx)) for idx in range(len(self.docs))]
        scores.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in scores]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(df: pd.DataFrame, text_column: str) -> Dict[str, float]:
    docs = (df["title"].fillna("") + " " + df[text_column].fillna("")).tolist()
    queries = df["pseudo_query"].tolist()
    dataset_ids = df["dataset_id"].tolist()
    index = BM25Index(docs)

    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []

    for i, query in enumerate(queries):
        ranked = index.rank(query)
        target_id = dataset_ids[i]
        ranked_ids = [dataset_ids[idx] for idx in ranked]

        rank = next((pos + 1 for pos, dataset_id in enumerate(ranked_ids) if dataset_id == target_id), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        recall_at_1.append(1.0 if rank and rank <= 1 else 0.0)
        recall_at_5.append(1.0 if rank and rank <= 5 else 0.0)
        recall_at_10.append(1.0 if rank and rank <= 10 else 0.0)

    return {
        "mrr": mean(reciprocal_ranks),
        "recall_at_1": mean(recall_at_1),
        "recall_at_5": mean(recall_at_5),
        "recall_at_10": mean(recall_at_10),
    }


def error_bucket(error_text: str) -> str:
    lowered = error_text.lower()
    if "rate limit" in lowered or "429" in lowered:
        return "rate_limit"
    if "prompt is too long" in lowered or "too long" in lowered:
        return "prompt_too_long"
    if not lowered.strip():
        return "none"
    return "other"


def compute_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    meteor_func, meteor_import_error = safe_import_meteor()
    bertscore_func, bertscore_import_error = safe_import_bertscore()

    result["original_description"] = result["original_description"].apply(clean_text)
    result["generated_description"] = result["generated_description"].apply(clean_text)
    result["generation_error"] = result["generation_error"].apply(clean_text)
    result["title"] = result["title"].apply(clean_text)

    result["has_generated_description"] = result["generated_description"].str.len() > 0
    result["has_generation_error"] = result["generation_error"].str.len() > 0
    result["error_bucket"] = result["generation_error"].apply(error_bucket)
    result["original_char_len"] = result["original_description"].str.len()
    result["generated_char_len"] = result["generated_description"].str.len()
    result["compression_ratio"] = result.apply(
        lambda row: compression_ratio(row["original_description"], row["generated_description"]),
        axis=1,
    )
    result["rouge1_f1"] = result.apply(
        lambda row: rouge_n_f1(row["original_description"], row["generated_description"], 1),
        axis=1,
    )
    result["rouge2_f1"] = result.apply(
        lambda row: rouge_n_f1(row["original_description"], row["generated_description"], 2),
        axis=1,
    )
    result["rougeL_f1"] = result.apply(
        lambda row: rouge_l_f1(row["original_description"], row["generated_description"]),
        axis=1,
    )
    result["meteor"] = result.apply(
        lambda row: meteor_score_value(
            row["original_description"],
            row["generated_description"],
            meteor_func,
        ),
        axis=1,
    )
    result["pseudo_query"] = result.apply(
        lambda row: build_query(row["title"], row["original_description"]),
        axis=1,
    )
    successful = result["has_generated_description"]
    bertscore_values = compute_bertscore_values(
        result.loc[successful, "original_description"].tolist(),
        result.loc[successful, "generated_description"].tolist(),
        bertscore_func,
    )
    result["bertscore_f1"] = None
    result.loc[successful, "bertscore_f1"] = bertscore_values
    result.attrs["meteor_available"] = meteor_func is not None
    result.attrs["meteor_import_error"] = meteor_import_error
    result.attrs["bertscore_available"] = bertscore_func is not None
    result.attrs["bertscore_import_error"] = bertscore_import_error
    return result


def summarize_subset(df: pd.DataFrame, label: str) -> Dict[str, float]:
    successful = df[df["has_generated_description"]]
    meteor_stats = metric_stats(successful["meteor"])
    bertscore_stats = metric_stats(successful["bertscore_f1"])
    return {
        "segment": label,
        "row_count": int(len(df)),
        "generation_success_rate": float(df["has_generated_description"].mean()) if len(df) else 0.0,
        "generation_error_rate": float(df["has_generation_error"].mean()) if len(df) else 0.0,
        "avg_original_char_len": float(df["original_char_len"].mean()) if len(df) else 0.0,
        "avg_generated_char_len": float(successful["generated_char_len"].mean()) if len(successful) else 0.0,
        "avg_compression_ratio": float(successful["compression_ratio"].mean()) if len(successful) else 0.0,
        "avg_rouge1_f1": float(successful["rouge1_f1"].mean()) if len(successful) else 0.0,
        "avg_rouge2_f1": float(successful["rouge2_f1"].mean()) if len(successful) else 0.0,
        "avg_rougeL_f1": float(successful["rougeL_f1"].mean()) if len(successful) else 0.0,
        "avg_meteor": meteor_stats["mean"],
        "avg_bertscore_f1": bertscore_stats["mean"],
        "meteor_valid_count": meteor_stats["valid_count"],
        "meteor_valid_rate": meteor_stats["valid_rate"],
        "meteor_median": meteor_stats["median"],
        "meteor_min": meteor_stats["min"],
        "meteor_max": meteor_stats["max"],
        "meteor_zero_rate": meteor_stats["zero_rate"],
        "bertscore_valid_count": bertscore_stats["valid_count"],
        "bertscore_valid_rate": bertscore_stats["valid_rate"],
        "bertscore_median": bertscore_stats["median"],
        "bertscore_min": bertscore_stats["min"],
        "bertscore_max": bertscore_stats["max"],
    }


def metric_display(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    return f"{value:.4f}"


def build_suspicious_metric_notes(overall: pd.Series) -> List[str]:
    notes: List[str] = []
    if overall["meteor_valid_count"] == 0:
        notes.append("METEOR produced no valid scores.")
    elif pd.notna(overall["meteor_zero_rate"]) and overall["meteor_zero_rate"] >= 0.95:
        notes.append("METEOR is near-zero on almost every scored row, which suggests the configuration may still need validation.")

    if overall["bertscore_valid_count"] == 0:
        notes.append("BERTScore produced no valid scores.")
    elif pd.notna(overall["avg_bertscore_f1"]) and overall["avg_bertscore_f1"] < 0.30:
        notes.append("BERTScore is unusually low for semantically related summaries, so the model settings should be sanity-checked against example rows.")

    return notes


def write_markdown_report(
    output_path: Path,
    summaries: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    error_df: pd.DataFrame,
    metric_availability: Dict[str, object],
) -> None:
    overall = summaries[summaries["segment"] == "overall"].iloc[0]
    overall_retrieval = retrieval_df[retrieval_df["segment"] == "overall"].set_index("corpus")
    suspicious_notes = build_suspicious_metric_notes(overall)

    lines = [
        "# Generated Description Evaluation Report",
        "",
        "## Reliability Metrics",
        f"- Evaluated rows: {int(overall['row_count'])}",
        f"- Generation success rate: {overall['generation_success_rate']:.2%}",
        f"- Generation error rate: {overall['generation_error_rate']:.2%}",
        "",
        "## Quality Metrics",
        f"- Average ROUGE-1 F1: {overall['avg_rouge1_f1']:.4f}",
        f"- Average ROUGE-2 F1: {overall['avg_rouge2_f1']:.4f}",
        f"- Average ROUGE-L F1: {overall['avg_rougeL_f1']:.4f}",
        f"- Average METEOR: {metric_display(overall['avg_meteor'])}",
        f"- Average BERTScore F1: {metric_display(overall['avg_bertscore_f1'])}",
        "",
        "## Metric Coverage",
        f"- METEOR available: {metric_availability['meteor_available']}",
        f"- METEOR valid rows: {int(overall['meteor_valid_count'])}",
        f"- METEOR valid-row rate: {overall['meteor_valid_rate']:.2%}",
        f"- METEOR median/min/max: {metric_display(overall['meteor_median'])} / {metric_display(overall['meteor_min'])} / {metric_display(overall['meteor_max'])}",
        f"- BERTScore available: {metric_availability['bertscore_available']}",
        f"- BERTScore valid rows: {int(overall['bertscore_valid_count'])}",
        f"- BERTScore valid-row rate: {overall['bertscore_valid_rate']:.2%}",
        f"- BERTScore median/min/max: {metric_display(overall['bertscore_median'])} / {metric_display(overall['bertscore_min'])} / {metric_display(overall['bertscore_max'])}",
        "",
        "## Retrieval Proxy",
        "Pseudo-queries are built from dataset titles plus salient terms from the original descriptions.",
        f"- Original description corpus MRR: {overall_retrieval.loc['original_description', 'mrr']:.4f}",
        f"- Generated description corpus MRR: {overall_retrieval.loc['generated_description', 'mrr']:.4f}",
        f"- Original description Recall@10: {overall_retrieval.loc['original_description', 'recall_at_10']:.2%}",
        f"- Generated description Recall@10: {overall_retrieval.loc['generated_description', 'recall_at_10']:.2%}",
        "",
        "## Reliability Error Breakdown",
    ]

    if metric_availability["meteor_import_error"]:
        lines.append(f"- METEOR import note: {metric_availability['meteor_import_error']}")
    if metric_availability["bertscore_import_error"]:
        lines.append(f"- BERTScore import note: {metric_availability['bertscore_import_error']}")
    if suspicious_notes:
        for note in suspicious_notes:
            lines.append(f"- Quality metric note: {note}")
    if metric_availability["meteor_import_error"] or metric_availability["bertscore_import_error"] or suspicious_notes:
        lines.append("")
        lines.append("## Errors")

    if error_df.empty:
        lines.append("- No generation errors found.")
    else:
        for _, row in error_df.iterrows():
            lines.append(
                f"- `{row['source']}` `{row['error_bucket']}`: {int(row['count'])}"
            )

    lines.extend([
        "",
        "## By Source",
        "",
        "| Segment | Success Rate | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | METEOR Valid | BERTScore F1 | BERTScore Valid | Avg Generated Chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for _, row in summaries.iterrows():
        if row["segment"] == "overall":
            continue
        lines.append(
            f"| {row['segment']} | {row['generation_success_rate']:.2%} | "
            f"{row['avg_rouge1_f1']:.4f} | {row['avg_rouge2_f1']:.4f} | "
            f"{row['avg_rougeL_f1']:.4f} | {metric_display(row['avg_meteor'])} | "
            f"{int(row['meteor_valid_count'])} | {metric_display(row['avg_bertscore_f1'])} | "
            f"{int(row['bertscore_valid_count'])} | "
            f"{row['avg_generated_char_len']:.1f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated dataset descriptions.")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to a CSV with original and generated descriptions.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation",
        help="Directory for evaluation outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, quoting=csv.QUOTE_MINIMAL)
    metrics_df = compute_row_metrics(df)
    metric_availability = {
        "meteor_available": bool(metrics_df.attrs.get("meteor_available")),
        "meteor_import_error": metrics_df.attrs.get("meteor_import_error"),
        "bertscore_available": bool(metrics_df.attrs.get("bertscore_available")),
        "bertscore_import_error": metrics_df.attrs.get("bertscore_import_error"),
    }

    summaries = [
        summarize_subset(metrics_df, "overall"),
        summarize_subset(metrics_df[metrics_df["source"] == "nyc_open_data"], "nyc_open_data"),
        summarize_subset(metrics_df[metrics_df["source"] == "data_gov"], "data_gov"),
    ]
    summaries_df = pd.DataFrame(summaries)

    retrieval_rows = []
    for label, subset in [
        ("overall", metrics_df[metrics_df["has_generated_description"]]),
        ("nyc_open_data", metrics_df[(metrics_df["source"] == "nyc_open_data") & (metrics_df["has_generated_description"])]),
        ("data_gov", metrics_df[(metrics_df["source"] == "data_gov") & (metrics_df["has_generated_description"])]),
    ]:
        if subset.empty:
            continue
        for corpus in ["original_description", "generated_description"]:
            row = {"segment": label, "corpus": corpus}
            row.update(evaluate_retrieval(subset, corpus))
            retrieval_rows.append(row)
    retrieval_df = pd.DataFrame(retrieval_rows)

    error_df = (
        metrics_df[metrics_df["has_generation_error"]]
        .groupby(["source", "error_bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )

    metrics_df.to_csv(output_dir / "generated_description_row_metrics.csv", index=False)
    summaries_df.to_csv(output_dir / "generated_description_summary.csv", index=False)
    retrieval_df.to_csv(output_dir / "generated_description_retrieval.csv", index=False)
    error_df.to_csv(output_dir / "generated_description_errors.csv", index=False)
    write_markdown_report(
        output_dir / "generated_description_evaluation_report.md",
        summaries_df,
        retrieval_df,
        error_df,
        metric_availability,
    )

    print(f"Saved row metrics to {output_dir / 'generated_description_row_metrics.csv'}")
    print(f"Saved summary metrics to {output_dir / 'generated_description_summary.csv'}")
    print(f"Saved retrieval metrics to {output_dir / 'generated_description_retrieval.csv'}")
    print(f"Saved error breakdown to {output_dir / 'generated_description_errors.csv'}")
    print(f"Saved markdown report to {output_dir / 'generated_description_evaluation_report.md'}")


if __name__ == "__main__":
    main()
