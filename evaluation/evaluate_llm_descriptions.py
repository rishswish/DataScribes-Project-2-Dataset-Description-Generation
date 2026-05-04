"""
Automatic evaluation of LLM-generated dataset descriptions.

Reads: generated_descriptions_all_200.csv (from Downloads or data/ directory)
Writes: data/evaluation_results.csv, data/evaluation_summary.csv

Metrics computed per row:
  - success / failure
  - generated and original word counts
  - sentence count in generated description
  - novelty score: fraction of generated words NOT in original (1.0 = fully novel)
  - title coverage: whether the generated description mentions the dataset title keywords
"""

import re
import sys
from pathlib import Path

import pandas as pd


CSV_CANDIDATES = [
    Path("data/generated_descriptions_all_200.csv"),
    Path.home() / "Downloads/generated_descriptions_all_200.csv",
]


def find_csv():
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find generated_descriptions_all_200.csv. "
        f"Tried: {[str(p) for p in CSV_CANDIDATES]}"
    )


def word_set(text: str) -> set:
    if not text or pd.isna(text):
        return set()
    return set(re.findall(r"[a-z]+", text.lower()))


def sentence_count(text: str) -> int:
    if not text or pd.isna(text):
        return 0
    return len(re.findall(r"[.!?]+", text.strip()))


def word_count(text: str) -> int:
    if not text or pd.isna(text):
        return 0
    return len(text.split())


def novelty_score(generated: str, original: str) -> float:
    """Fraction of unique words in generated that do not appear in original."""
    gen_words = word_set(generated)
    if not gen_words:
        return 0.0
    orig_words = word_set(original)
    novel = gen_words - orig_words
    return round(len(novel) / len(gen_words), 3)


def title_coverage(generated: str, title: str) -> bool:
    """True if at least one significant title word appears in the generated description."""
    if not generated or pd.isna(generated) or not title or pd.isna(title):
        return False
    stopwords = {"the", "a", "an", "of", "in", "and", "or", "for", "to", "by", "on", "at", "with"}
    title_words = {w for w in word_set(title) if w not in stopwords and len(w) > 2}
    gen_words = word_set(generated)
    return bool(title_words & gen_words)


def classify_error(error: str) -> str:
    if pd.isna(error):
        return None
    if "rate_limit" in error:
        return "rate_limit"
    if "too long" in error or "tokens >" in error:
        return "prompt_too_long"
    return "other"


def main():
    csv_path = find_csv()
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    df["success"] = df["generated_description"].notna()
    df["error_type"] = df["generation_error"].apply(classify_error)

    df["original_word_count"] = df["original_description"].apply(word_count)
    df["generated_word_count"] = df["generated_description"].apply(word_count)
    df["generated_sentence_count"] = df["generated_description"].apply(sentence_count)

    df["novelty_score"] = df.apply(
        lambda r: novelty_score(r["generated_description"], r["original_description"]), axis=1
    )
    df["title_coverage"] = df.apply(
        lambda r: title_coverage(r["generated_description"], r["title"]), axis=1
    )

    df["follows_length_guideline"] = df["generated_sentence_count"].between(3, 5)

    # ── Per-row results ──────────────────────────────────────────────────────
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    row_cols = [
        "dataset_id", "source", "title",
        "success", "error_type",
        "original_word_count", "generated_word_count", "generated_sentence_count",
        "follows_length_guideline", "novelty_score", "title_coverage",
    ]
    results = df[row_cols].copy()
    results.to_csv(output_dir / "evaluation_results.csv", index=False)
    print(f"Saved per-row results to data/evaluation_results.csv ({len(results)} rows)")

    # ── Summary table ────────────────────────────────────────────────────────
    def summarise(group):
        return pd.Series({
            "total_datasets": len(group),
            "successful": group["success"].sum(),
            "failed": (~group["success"]).sum(),
            "success_rate_pct": round(group["success"].mean() * 100, 1),
            "avg_original_words": round(group["original_word_count"].mean(), 1),
            "avg_generated_words": round(group[group["success"]]["generated_word_count"].mean(), 1),
            "avg_generated_sentences": round(group[group["success"]]["generated_sentence_count"].mean(), 2),
            "pct_follows_3_5_sentences": round(group[group["success"]]["follows_length_guideline"].mean() * 100, 1),
            "avg_novelty_score": round(group[group["success"]]["novelty_score"].mean(), 3),
            "pct_title_covered": round(group[group["success"]]["title_coverage"].mean() * 100, 1),
        })

    summary_rows = []
    for source, group in df.groupby("source", sort=True):
        row = summarise(group).to_dict()
        row["source"] = source
        summary_rows.append(row)

    overall = summarise(df)
    overall["source"] = "OVERALL"
    summary = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([overall])], ignore_index=True)

    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)

    # ── Print report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"\n[{row['source']}]")
        print(f"  Datasets:            {int(row['total_datasets'])} total, {int(row['successful'])} success, {int(row['failed'])} failed")
        print(f"  Success rate:        {row['success_rate_pct']}%")
        print(f"  Avg original words:  {row['avg_original_words']}")
        print(f"  Avg generated words: {row['avg_generated_words']}")
        print(f"  Avg sentences:       {row['avg_generated_sentences']}")
        print(f"  Follows 3-5 sent:    {row['pct_follows_3_5_sentences']}%")
        print(f"  Novelty score:       {row['avg_novelty_score']} (1.0 = fully new vocab)")
        print(f"  Title coverage:      {row['pct_title_covered']}%")

    if df["error_type"].notna().any():
        print("\n[FAILURES]")
        for _, row in df[df["error_type"].notna()][["dataset_id", "source", "error_type", "title"]].iterrows():
            print(f"  {row['dataset_id']} ({row['source']}) — {row['error_type']}: {row['title']}")

    print(f"\nSaved summary to data/evaluation_summary.csv")


if __name__ == "__main__":
    main()
