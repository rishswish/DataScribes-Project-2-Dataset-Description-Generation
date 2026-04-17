import json
from pathlib import Path
import pandas as pd


def main():
    input_path = Path("data/raw/data_gov_metadata.json")
    output_path = Path("data/metadata/data_gov_metadata.parquet")

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved parquet to {output_path}")
    print(df[["dataset_id", "title", "source", "download_url"]].head())


if __name__ == "__main__":
    main()