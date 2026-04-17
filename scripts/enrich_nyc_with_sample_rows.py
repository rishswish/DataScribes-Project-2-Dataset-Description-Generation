import json
import time
from pathlib import Path

import requests


def fetch_sample_rows(dataset_id: str, limit: int = 5):
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit={limit}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    input_path = Path("data/raw/nyc_open_data_metadata.json")
    output_path = Path("data/raw/nyc_open_data_metadata_with_samples.json")

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    enriched = []

    for i, record in enumerate(records, start=1):
        dataset_id = record["dataset_id"]
        print(f"[{i}/{len(records)}] fetching sample rows for {dataset_id}")

        try:
            sample_rows = fetch_sample_rows(dataset_id, limit=5)
            record["sample_rows_json"] = json.dumps(sample_rows, ensure_ascii=False)
        except Exception as e:
            print(f"  failed for {dataset_id}: {e}")
            record["sample_rows_json"] = json.dumps(
                {"error": str(e)}, ensure_ascii=False
            )

        enriched.append(record)
        time.sleep(0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Saved enriched metadata to {output_path}")


if __name__ == "__main__":
    main()