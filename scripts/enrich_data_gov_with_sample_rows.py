import json
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def fetch_csv_sample_rows(url: str, limit: int = 5):
    response = requests.get(
        url,
        timeout=90,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), nrows=limit)
    return df.to_dict(orient="records")


def main():
    input_path = Path("data/raw/data_gov_metadata.json")
    output_path = Path("data/raw/data_gov_metadata_with_samples.json")

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    enriched = []

    for i, record in enumerate(records, start=1):
        dataset_id = record["dataset_id"]
        download_url = record.get("download_url")

        print(f"[{i}/{len(records)}] dataset_id={dataset_id}")
        print(f"  download_url={download_url}")

        if not download_url:
            record["sample_rows_json"] = json.dumps(
                {"error": "no_csv_download_url"},
                ensure_ascii=False,
            )
            enriched.append(record)
            continue

        try:
            sample_rows = fetch_csv_sample_rows(download_url, limit=5)
            record["sample_rows_json"] = json.dumps(sample_rows, ensure_ascii=False)

            if not record.get("column_names"):
                record["column_names"] = list(sample_rows[0].keys()) if sample_rows else []

            print(f"  fetched {len(sample_rows)} sample rows")

        except Exception as e:
            print(f"  failed: {e}")
            record["sample_rows_json"] = json.dumps(
                {"error": str(e)},
                ensure_ascii=False,
            )

        enriched.append(record)
        time.sleep(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Saved enriched metadata to {output_path}")


if __name__ == "__main__":
    main()