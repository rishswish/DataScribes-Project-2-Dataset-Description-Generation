import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SOCRATA_CATALOG_URL = "https://data.cityofnewyork.us/api/views.json"
SOCRATA_VIEW_URL_TEMPLATE = "https://data.cityofnewyork.us/api/views/{dataset_id}.json"


def safe_get(dct: Dict[str, Any], key: str, default=None):
    return dct[key] if key in dct else default


def extract_keywords(category: Optional[str], tags: Optional[List[str]]) -> List[str]:
    out = []
    if category:
        out.append(str(category))
    if tags:
        out.extend([str(t) for t in tags if t is not None])

    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def extract_column_names(columns: Optional[List[Dict[str, Any]]]) -> List[str]:
    names = []
    for col in columns or []:
        name = safe_get(col, "name")
        if name:
            names.append(str(name))
    return names


def extract_column_types(columns: Optional[List[Dict[str, Any]]]) -> List[str]:
    types_ = []
    for col in columns or []:
        dtype = safe_get(col, "dataTypeName")
        types_.append(str(dtype) if dtype is not None else "unknown")
    return types_


def extract_download_url(dataset_id: Optional[str]) -> Optional[str]:
    if not dataset_id:
        return None
    return f"https://data.cityofnewyork.us/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"


def extract_landing_page_url(dataset_id: Optional[str]) -> Optional[str]:
    if not dataset_id:
        return None
    return f"https://data.cityofnewyork.us/d/{dataset_id}"


def fetch_dataset_detail(dataset_id: str) -> Dict[str, Any]:
    url = SOCRATA_VIEW_URL_TEMPLATE.format(dataset_id=dataset_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def normalize_dataset(summary_entry: Dict[str, Any], detail_entry: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = safe_get(summary_entry, "id")
    title = safe_get(summary_entry, "name")
    description = safe_get(summary_entry, "description")
    category = safe_get(summary_entry, "category")
    tags = safe_get(summary_entry, "tags", [])
    rows_updated_at = safe_get(summary_entry, "rowsUpdatedAt")

    detail_columns = safe_get(detail_entry, "columns", [])
    license_info = safe_get(detail_entry, "license")
    license_name = None
    if isinstance(license_info, dict):
        license_name = safe_get(license_info, "name")

    row_count = safe_get(detail_entry, "rowsUpdatedAt", None)

    normalized = {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "source": "nyc_open_data",
        "title": str(title) if title is not None else None,
        "original_description": str(description) if description is not None else None,
        "keywords": extract_keywords(category, tags),
        "column_names": extract_column_names(detail_columns),
        "column_types_raw": extract_column_types(detail_columns),
        "download_url": extract_download_url(str(dataset_id)) if dataset_id is not None else None,
        "landing_page_url": extract_landing_page_url(str(dataset_id)) if dataset_id is not None else None,
        "record_count_estimate": None,
        "last_updated": str(rows_updated_at) if rows_updated_at is not None else None,
        "license": str(license_name) if license_name is not None else None,
        "sample_rows_json": None,
        "raw_metadata_json": json.dumps(
            {
                "summary_entry": summary_entry,
                "detail_entry": detail_entry,
            },
            ensure_ascii=False,
        ),
    }
    return normalized


def fetch_nyc_open_data(limit: int = 10) -> List[Dict[str, Any]]:
    response = requests.get(SOCRATA_CATALOG_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError("Expected NYC Open Data catalog response to be a list.")

    trimmed = payload[:limit]
    normalized_records = []

    for entry in trimmed:
        dataset_id = safe_get(entry, "id")
        if not dataset_id:
            continue

        detail_entry = fetch_dataset_detail(str(dataset_id))
        normalized_records.append(normalize_dataset(entry, detail_entry))

    return normalized_records


def save_raw_json(records: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main(limit: int = 10, output_path: str = "data/raw/nyc_open_data_metadata.json") -> None:
    records = fetch_nyc_open_data(limit=limit)
    save_raw_json(records, output_path)
    print(f"Saved {len(records)} NYC Open Data records to {output_path}")


if __name__ == "__main__":
    main()