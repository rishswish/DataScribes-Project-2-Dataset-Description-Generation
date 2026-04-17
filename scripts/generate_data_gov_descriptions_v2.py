import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_dict(value: str) -> Dict[str, Any]:
    if value is None or pd.isna(value):
        return {}
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def choose_domain_phrase(title: str, keywords: List[str], original_description: str) -> str:
    text = f"{title} {' '.join(keywords)} {original_description}".lower()

    if "climate" in text or "weather" in text or "environment" in text:
        return "climate or environmental data"
    if "transport" in text or "traffic" in text or "vehicle" in text:
        return "transportation-related public data"
    if "crime" in text or "offense" in text or "police" in text:
        return "public safety data"
    if "health" in text or "disease" in text or "cdc" in text:
        return "public health data"
    if "energy" in text:
        return "energy-related public data"
    return "public sector open data"


def bucket_columns(columns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    numeric_cols = []
    date_cols = []
    text_cols = []

    for c in columns:
        name = c.get("name")
        inferred = c.get("inferred_sample_type")
        if not name:
            continue

        lower_name = name.lower()

        if "date" in lower_name or inferred == "date_like":
            date_cols.append(name)
        elif inferred == "numeric_like":
            if "year" in lower_name:
                text_cols.append(name)
            else:
                numeric_cols.append(name)
        else:
            text_cols.append(name)

    return {
        "numeric": numeric_cols[:5],
        "date": date_cols[:4],
        "text": text_cols[:6],
    }


def build_description(llm_input: Dict[str, Any]) -> str:
    title = llm_input.get("title", "")
    original_description = (llm_input.get("original_description") or "").strip()
    keywords = llm_input.get("keywords", [])[:8]
    declared_column_count = llm_input.get("declared_column_count", 0)
    sample_row_count = llm_input.get("sample_row_count", 0)
    top_columns = llm_input.get("top_columns_for_prompt", [])[:20]

    domain_phrase = choose_domain_phrase(title, keywords, original_description)
    buckets = bucket_columns(top_columns)

    parts = []
    parts.append(f"{title} is a Data.gov dataset about {domain_phrase}.")
    parts.append(
        f"The dataset contains {declared_column_count} identified columns, and the current pipeline inspected {sample_row_count} sample rows to understand its structure."
    )

    example_cols = [c.get("name") for c in top_columns[:6] if c.get("name")]
    if example_cols:
        parts.append(f"Some representative fields are {', '.join(example_cols)}.")

    if buckets["numeric"]:
        parts.append(
            f"It includes quantitative fields such as {', '.join(buckets['numeric'])}, which may support aggregation, comparisons, and descriptive analysis."
        )

    if buckets["date"]:
        parts.append(
            f"It also includes time-related fields such as {', '.join(buckets['date'])}, which may support temporal analysis."
        )

    if buckets["text"]:
        parts.append(
            f"Other fields such as {', '.join(buckets['text'][:4])} help describe identifiers, categories, or descriptive attributes in the dataset."
        )

    parts.append(
        "This dataset may be useful for exploratory analysis, reporting, and understanding patterns in the underlying public data domain."
    )
    return " ".join(parts)


def main():
    input_path = Path("data/descriptions/data_gov_llm_inputs.parquet")
    output_path = Path("data/descriptions/data_gov_generated_descriptions_v2.parquet")

    df = pd.read_parquet(input_path)
    rows = []

    for _, row in df.iterrows():
        llm_input = safe_dict(row["llm_input_json"])
        generated_description = build_description(llm_input)

        rows.append({
            "dataset_id": row["dataset_id"],
            "title": row["title"],
            "description_type": "user_focused",
            "generation_model": "template_baseline_v2",
            "generated_description": generated_description,
            "llm_input_json": row["llm_input_json"],
        })

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"Saved generated descriptions to {output_path}")
    print(out_df[['title', 'generation_model']].head())


if __name__ == "__main__":
    main()