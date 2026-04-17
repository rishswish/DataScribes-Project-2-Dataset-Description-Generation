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


def build_description(llm_input: Dict[str, Any]) -> str:
    title = llm_input.get("title", "")
    original_description = (llm_input.get("original_description") or "").strip()
    keywords = llm_input.get("keywords", [])[:6]
    declared_column_count = llm_input.get("declared_column_count", 0)
    sample_row_count = llm_input.get("sample_row_count", 0)
    top_columns = llm_input.get("top_columns_for_prompt", [])[:8]

    col_names = [c.get("name") for c in top_columns if c.get("name")]
    col_preview = ", ".join(col_names[:6])

    numeric_cols = [
        c["name"] for c in top_columns
        if c.get("inferred_sample_type") == "numeric_like"
    ][:4]

    date_cols = [
        c["name"] for c in top_columns
        if c.get("inferred_sample_type") == "date_like"
    ][:3]

    parts: List[str] = []

    first = f"{title} is a NYC Open Data dataset"
    if keywords:
        first += f" related to {', '.join(keywords[:3])}"
    first += "."
    parts.append(first)

    if original_description:
        parts.append(
            f"It appears to describe {original_description[:350].rstrip()}."
        )

    structure = f"The dataset has {declared_column_count} declared columns"
    if sample_row_count:
        structure += f", and the current pipeline inspected {sample_row_count} sample rows"
    structure += "."
    parts.append(structure)

    if col_preview:
        parts.append(f"Example columns include {col_preview}.")

    if numeric_cols:
        parts.append(
            f"It contains numeric-like fields such as {', '.join(numeric_cols)}, which may support counting, aggregation, or trend analysis."
        )

    if date_cols:
        parts.append(
            f"It also includes date-like fields such as {', '.join(date_cols)}, which may support time-based analysis."
        )

    parts.append(
        "This dataset may be useful for exploratory analysis, reporting, and understanding patterns in the underlying NYC public data domain."
    )

    return " ".join(parts)


def main():
    input_path = Path("data/descriptions/nyc_open_data_llm_inputs.parquet")
    output_path = Path("data/descriptions/nyc_open_data_generated_descriptions.parquet")

    df = pd.read_parquet(input_path)

    rows = []
    for _, row in df.iterrows():
        llm_input = safe_dict(row["llm_input_json"])
        generated_description = build_description(llm_input)

        rows.append({
            "dataset_id": row["dataset_id"],
            "title": row["title"],
            "description_type": "user_focused",
            "generation_model": "template_baseline_v1",
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