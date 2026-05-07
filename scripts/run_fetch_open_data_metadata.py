from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests
from pyspark.sql import SparkSession
from pyspark.sql import types as T


NYC_CATALOG_URL = "https://data.cityofnewyork.us/api/views.json"
NYC_VIEW_URL_TEMPLATE = "https://data.cityofnewyork.us/api/views/{dataset_id}.json"

DATA_GOV_SEARCH_URL = "https://catalog.data.gov/search"


def safe_get(dct: Dict[str, Any], key: str, default=None):
    return dct[key] if key in dct else default


def fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> Any:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            print(f"[WARN] fetch failed attempt={attempt}/{retries} url={response.url if 'response' in locals() else url}: {e}")
            if attempt < retries:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch URL after {retries} attempts: {url}") from last_error


def extract_keywords(category: Optional[str], tags: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if category:
        out.append(str(category))
    if tags:
        out.extend([str(t) for t in tags if t is not None])

    seen = set()
    deduped: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def extract_column_names(columns: Optional[List[Dict[str, Any]]]) -> List[str]:
    names: List[str] = []
    for col in columns or []:
        name = safe_get(col, "name")
        if name:
            names.append(str(name))
    return names


def extract_column_types(columns: Optional[List[Dict[str, Any]]]) -> List[str]:
    types_: List[str] = []
    for col in columns or []:
        dtype = safe_get(col, "dataTypeName")
        types_.append(str(dtype) if dtype is not None else "unknown")
    return types_


def normalize_nyc_dataset(summary_entry: Dict[str, Any], detail_entry: Dict[str, Any]) -> Dict[str, Any]:
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

    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "source": "nyc_open_data",
        "title": str(title) if title is not None else None,
        "original_description": str(description) if description is not None else None,
        "keywords_json": json.dumps(extract_keywords(category, tags), ensure_ascii=False),
        "column_names_json": json.dumps(extract_column_names(detail_columns), ensure_ascii=False),
        "column_types_raw_json": json.dumps(extract_column_types(detail_columns), ensure_ascii=False),
        "download_url": f"https://data.cityofnewyork.us/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD" if dataset_id else None,
        "landing_page_url": f"https://data.cityofnewyork.us/d/{dataset_id}" if dataset_id else None,
        "record_count_estimate": None,
        "last_updated": str(rows_updated_at) if rows_updated_at is not None else None,
        "license": str(license_name) if license_name is not None else None,
        "sample_rows_json": None,
        "raw_metadata_json": json.dumps(
            {"summary_entry": summary_entry, "detail_entry": detail_entry},
            ensure_ascii=False,
        ),
    }


def fetch_nyc_metadata(limit: int) -> List[Dict[str, Any]]:
    payload = fetch_json(NYC_CATALOG_URL)

    if not isinstance(payload, list):
        raise ValueError("Expected NYC catalog response to be a list.")

    trimmed = payload[:limit]
    rows: List[Dict[str, Any]] = []

    for idx, entry in enumerate(trimmed, start=1):
        dataset_id = safe_get(entry, "id")
        if not dataset_id:
            continue

        print(f"[INFO] NYC detail {idx}/{len(trimmed)} dataset_id={dataset_id}")
        detail = fetch_json(NYC_VIEW_URL_TEMPLATE.format(dataset_id=dataset_id))
        rows.append(normalize_nyc_dataset(entry, detail))

    return rows


def normalize_data_gov_dataset(entry: Dict[str, Any]) -> Dict[str, Any]:
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
            "csv" in media_type
            or "csv" in fmt
            or str(access_url).lower().endswith(".csv")
        ):
            csv_url = access_url
            break

    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "source": "data_gov",
        "title": str(title) if title is not None else None,
        "original_description": str(notes) if notes is not None else None,
        "keywords_json": json.dumps([str(x) for x in tags] if isinstance(tags, list) else [], ensure_ascii=False),
        "column_names_json": json.dumps([], ensure_ascii=False),
        "column_types_raw_json": json.dumps([], ensure_ascii=False),
        "download_url": csv_url,
        "landing_page_url": landing_page,
        "record_count_estimate": None,
        "last_updated": str(modified) if modified is not None else None,
        "license": str(license_url) if license_url is not None else None,
        "sample_rows_json": None,
        "raw_metadata_json": json.dumps(entry, ensure_ascii=False),
    }


def fetch_data_gov_metadata(limit: int) -> List[Dict[str, Any]]:
    params = {
        "q": "",
        "per_page": limit,
    }

    payload = fetch_json(DATA_GOV_SEARCH_URL, params=params)

    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("Expected Data.gov search response with a 'results' key.")

    results = payload["results"]
    if not isinstance(results, list):
        raise ValueError("Expected Data.gov 'results' to be a list.")

    rows = [normalize_data_gov_dataset(entry) for entry in results[:limit]]
    return rows


def build_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("dataset_id", T.StringType(), True),
            T.StructField("source", T.StringType(), True),
            T.StructField("title", T.StringType(), True),
            T.StructField("original_description", T.StringType(), True),
            T.StructField("keywords_json", T.StringType(), True),
            T.StructField("column_names_json", T.StringType(), True),
            T.StructField("column_types_raw_json", T.StringType(), True),
            T.StructField("download_url", T.StringType(), True),
            T.StructField("landing_page_url", T.StringType(), True),
            T.StructField("record_count_estimate", T.LongType(), True),
            T.StructField("last_updated", T.StringType(), True),
            T.StructField("license", T.StringType(), True),
            T.StructField("sample_rows_json", T.StringType(), True),
            T.StructField("raw_metadata_json", T.StringType(), True),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_parquet", required=True, help="HDFS output parquet path")
    parser.add_argument("--nyc_limit", type=int, default=20, help="Number of NYC datasets")
    parser.add_argument("--data_gov_limit", type=int, default=20, help="Number of Data.gov datasets")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("fetch_open_data_metadata").getOrCreate()

    print(f"[INFO] Fetching NYC metadata: {args.nyc_limit}")
    nyc_rows = fetch_nyc_metadata(limit=args.nyc_limit)

    print(f"[INFO] Fetching Data.gov metadata: {args.data_gov_limit}")
    data_gov_rows = fetch_data_gov_metadata(limit=args.data_gov_limit)

    all_rows = nyc_rows + data_gov_rows
    if not all_rows:
        raise ValueError("No metadata rows were fetched from either source.")

    df = spark.createDataFrame(all_rows, schema=build_schema())

    print(f"[INFO] Writing {df.count()} combined rows to {args.output_parquet}")
    df.write.mode("overwrite").parquet(args.output_parquet)

    df.groupBy("source").count().show(truncate=False)
    df.select("dataset_id", "source", "title", "download_url").show(20, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
