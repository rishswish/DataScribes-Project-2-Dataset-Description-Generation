import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        try:
            out = value.tolist()
            if isinstance(out, list):
                return out
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def safe_dict_from_json(text: str) -> Dict[str, Any]:
    if text is None or pd.isna(text):
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def trim_text(text: str, max_len: int = 2000) -> str:
    if text is None:
        return ""
    return str(text).strip()[:max_len]


def build_summary(row: pd.Series) -> Dict[str, Any]:
    column_names = safe_list(row["column_names"])
    column_types_raw = safe_list(row["column_types_raw"])
    keywords = safe_list(row["keywords"])
    sample_profile = safe_dict_from_json(row["sample_columns_profile_json"])

    top_columns = []
    for idx, col in enumerate(column_names[:25]):
        declared_type = column_types_raw[idx] if idx < len(column_types_raw) else "unknown"
        sample_info = sample_profile.get(col, {})
        top_columns.append({
            "name": col,
            "declared_type": declared_type,
            "inferred_sample_type": sample_info.get("inferred_sample_type", "unknown"),
            "example_values": sample_info.get("example_values", []),
            "missing_count_in_sample": sample_info.get("missing_count", None),
            "unique_count_in_sample": sample_info.get("unique_count_sample", None),
        })

    llm_input = {
        "dataset_id": row["dataset_id"],
        "source": row["source"],
        "title": row["title"],
        "original_description": trim_text(row["original_description"]),
        "keywords": keywords[:20],
        "declared_column_count": int(row["declared_column_count"]),
        "sample_row_count": int(row["sample_row_count"]),
        "sample_column_count": int(row["sample_column_count"]),
        "top_columns_for_prompt": top_columns,
        "task": "Generate a clear user-focused dataset description.",
    }

    return {
        "dataset_id": row["dataset_id"],
        "title": row["title"],
        "llm_input_json": json.dumps(llm_input, ensure_ascii=False),
    }


def main():
    input_path = Path("data/profiles/data_gov_profiles.parquet")
    output_path = Path("data/descriptions/data_gov_llm_inputs.parquet")

    df = pd.read_parquet(input_path)
    out_records = [build_summary(row) for _, row in df.iterrows()]
    out_df = pd.DataFrame(out_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"Saved LLM inputs to {output_path}")
    print(out_df[['title']].head())


if __name__ == "__main__":
    main()