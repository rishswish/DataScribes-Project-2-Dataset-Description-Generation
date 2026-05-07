from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from autoddg import AutoDDG


REQUIRED_COLUMNS = ["dataset_id", "title", "source", "sample_csv"]


def validate_input_dataframe(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input parquet is missing required columns: {missing_cols}"
        )


def generate_description_for_row(
    row: pd.Series,
    autoddg: AutoDDG,
) -> Dict[str, Any]:
    dataset_id = row.get("dataset_id")
    title = row.get("title")
    source = row.get("source")
    sample_csv = row.get("sample_csv")

    if pd.isna(sample_csv) or not isinstance(sample_csv, str) or not sample_csv.strip():
        raise ValueError(
            f"Row for dataset_id={dataset_id} is missing a valid sample_csv value."
        )

    prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)

    return {
        "dataset_id": dataset_id,
        "title": title,
        "source": source,
        "prompt": prompt,
        "generated_description": description,
        "status": "success",
        "error_message": None,
    }


def generate_descriptions(
    input_parquet: str,
    output_parquet: str,
    model_name: str = "gpt-4o-mini",
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    input_path = Path(input_parquet)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    validate_input_dataframe(df)

    if max_records is not None:
        df = df.head(max_records).copy()

    client = OpenAI()
    autoddg = AutoDDG(client=client, model_name=model_name)

    results: List[Dict[str, Any]] = []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        dataset_id = row.get("dataset_id", f"row_{idx}")
        print(f"[{idx}/{total}] Generating description for: {dataset_id}")

        try:
            result = generate_description_for_row(row, autoddg)
        except Exception as e:
            result = {
                "dataset_id": dataset_id,
                "title": row.get("title"),
                "source": row.get("source"),
                "prompt": None,
                "generated_description": None,
                "status": "error",
                "error_message": str(e),
            }

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_path, index=False)

    return results_df