import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_load_sample_rows(sample_rows_json: str) -> List[Dict[str, Any]]:
    if pd.isna(sample_rows_json) or sample_rows_json is None:
        return []
    try:
        rows = json.loads(sample_rows_json)
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def normalize_list_field(value) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    # pandas/parquet sometimes gives array-like objects
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
        except Exception:
            pass

    # if saved as JSON string
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


def infer_basic_type(values: List[Any]) -> str:
    cleaned = [v for v in values if v is not None and str(v).strip() != ""]
    if not cleaned:
        return "unknown"

    as_str = [str(v).strip() for v in cleaned]

    num_count = 0
    for v in as_str:
        try:
            float(v)
            num_count += 1
        except Exception:
            pass

    if num_count == len(as_str):
        return "numeric_like"

    date_count = 0
    for v in as_str:
        parsed = pd.to_datetime(v, errors="coerce")
        if not pd.isna(parsed):
            date_count += 1

    if date_count == len(as_str):
        return "date_like"

    return "text_like"


def profile_sample_rows(sample_rows_json: str) -> Dict[str, Any]:
    rows = safe_load_sample_rows(sample_rows_json)

    if not rows:
        return {
            "sample_row_count": 0,
            "sample_column_count": 0,
            "sample_columns_profile": {},
        }

    df = pd.DataFrame(rows)

    column_profiles = {}
    for col in df.columns:
        values = df[col].tolist()
        non_null = df[col].notna().sum()
        missing = int(df[col].isna().sum())
        unique_count = int(df[col].nunique(dropna=True))
        example_values = [x for x in df[col].dropna().astype(str).head(3).tolist()]

        column_profiles[col] = {
            "non_null_count": int(non_null),
            "missing_count": missing,
            "unique_count_sample": unique_count,
            "inferred_sample_type": infer_basic_type(values),
            "example_values": example_values,
        }

    return {
        "sample_row_count": int(len(df)),
        "sample_column_count": int(len(df.columns)),
        "sample_columns_profile": column_profiles,
    }


def build_profile_record(row: pd.Series) -> Dict[str, Any]:
    column_names = normalize_list_field(row["column_names"])
    column_types_raw = normalize_list_field(row["column_types_raw"])
    keywords = normalize_list_field(row["keywords"])

    sample_profile = profile_sample_rows(row["sample_rows_json"])

    return {
        "dataset_id": row["dataset_id"],
        "source": row["source"],
        "title": row["title"],
        "original_description": row["original_description"],
        "keywords": keywords,
        "column_names": column_names,
        "column_types_raw": column_types_raw,
        "declared_column_count": len(column_names),
        "sample_row_count": sample_profile["sample_row_count"],
        "sample_column_count": sample_profile["sample_column_count"],
        "sample_columns_profile_json": json.dumps(sample_profile["sample_columns_profile"], ensure_ascii=False),
    }


def main():
    input_path = Path("data/metadata/nyc_open_data_metadata_with_samples.parquet")
    output_path = Path("data/profiles/nyc_open_data_profiles.parquet")

    df = pd.read_parquet(input_path)
    profile_records = [build_profile_record(row) for _, row in df.iterrows()]
    out_df = pd.DataFrame(profile_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"Saved profiles to {output_path}")
    print(out_df[['title', 'declared_column_count', 'sample_row_count', 'sample_column_count']].head())


if __name__ == "__main__":
    main()