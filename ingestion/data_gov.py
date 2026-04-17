import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DATA_GOV_PACKAGE_SEARCH_URL = "https://catalog.data.gov/api/3/action/package_search?rows=10&start=0"


def safe_get(dct: Dict[str, Any], key: str, default=None):
    return dct[key] if key in dct else default


def extract_tags(tags: List[Dict[str, Any]]) -> List[str]:
    out = []
    for tag in tags or []:
        if isinstance(tag, dict):
            name = tag.get("display_name") or tag.get("name")
            if name:
                out.append(str(name))
    return out


def extract_resources(resources: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    csv_url = None
    landing_url = None

    for r in resources or []:
        if not landing_url and r.get("url"):
            landing_url = str(r["url"])

        fmt = (r.get("format") or "").lower()
        url = r.get("url")
        if url and ("csv" in fmt or str(url).lower().endswith(".csv")):
            csv_url = str(url)
            break

    return {
        "download_url": csv_url,
        "landing_page_url": landing_url,
    }


def extract_column_names_from_resources(resources: List[Dict[str, Any]]) -> List[str]:
    return []


def extract_column_types_from_resources(resources: List[Dict[str, Any]]) -> List[str]:
    return []

def normalize_dataset(entry: Dict[str, Any]) -> Dict[str, Any]:
    dcat = entry.get("dcat", {}) or {}

    dataset_id = entry.get("identifier") or dcat.get("identifier")
    title = entry.get("title") or dcat.get("title")
    notes = entry.get("description") or dcat.get("description")
    tags = entry.get("keyword") or dcat.get("keyword") or []
    modified = dcat.get("modified")
    license_url = dcat.get("license")
    landing_page = dcat.get("landingPage")

    distributions = dcat.get("distribution", []) or []

    csv_url = None
    for dist in distributions:
        if not isinstance(dist, dict):
            continue
        access_url = dist.get("accessURL") or dist.get("downloadURL")
        media_type = (dist.get("mediaType") or "").lower()
        fmt = (dist.get("format") or "").lower()

        if access_url and (
            "csv" in media_type or
            "csv" in fmt or
            str(access_url).lower().endswith(".csv")
        ):
            csv_url = access_url
            break

    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "source": "data_gov",
        "title": str(title) if title is not None else None,
        "original_description": str(notes) if notes is not None else None,
        "keywords": [str(x) for x in tags] if isinstance(tags, list) else [],
        "column_names": [],
        "column_types_raw": [],
        "download_url": csv_url,
        "landing_page_url": landing_page,
        "record_count_estimate": None,
        "last_updated": str(modified) if modified is not None else None,
        "license": str(license_url) if license_url is not None else None,
        "sample_rows_json": None,
        "raw_metadata_json": json.dumps(entry, ensure_ascii=False),
    }

def fetch_data_gov_metadata(limit: int = 10) -> List[Dict[str, Any]]:
    url = "https://catalog.data.gov/search"
    params = {
        "q": "",
        "per_page": limit,
    }

    print("REQUEST URL:", url)
    print("PARAMS:", params)

    response = requests.get(
        url,
        params=params,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    print("STATUS CODE:", response.status_code)
    print("FINAL URL:", response.url)

    response.raise_for_status()
    payload = response.json()

    results = payload["results"]
    return [normalize_dataset(entry) for entry in results]

def save_raw_json(records: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main(limit: int = 10, output_path: str = "data/raw/data_gov_metadata.json") -> None:
    records = fetch_data_gov_metadata(limit=limit)
    save_raw_json(records, output_path)
    print(f"Saved {len(records)} Data.gov records to {output_path}")


if __name__ == "__main__":
    main()