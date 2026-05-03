import json
import requests
from pyspark.sql import SparkSession


# ----------------------------
# NYC OPEN DATA HELPERS
# ----------------------------

NYC_CATALOG_URL = "https://data.cityofnewyork.us/api/views.json"
NYC_VIEW_URL_TEMPLATE = "https://data.cityofnewyork.us/api/views/{dataset_id}.json"


def safe_get(dct, key, default=None):
    return dct[key] if key in dct else default


def extract_keywords(category, tags):
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


def extract_column_names(columns):
    names = []
    for col in columns or []:
        name = safe_get(col, "name")
        if name:
            names.append(str(name))
    return names


def extract_column_types(columns):
    types_ = []
    for col in columns or []:
        dtype = safe_get(col, "dataTypeName")
        types_.append(str(dtype) if dtype is not None else "unknown")
    return types_


def extract_nyc_download_url(dataset_id):
    if not dataset_id:
        return None
    return f"https://data.cityofnewyork.us/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"


def extract_nyc_landing_page_url(dataset_id):
    if not dataset_id:
        return None
    return f"https://data.cityofnewyork.us/d/{dataset_id}"


def fetch_nyc_dataset_detail(dataset_id):
    url = NYC_VIEW_URL_TEMPLATE.format(dataset_id=dataset_id)
    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.json()


def normalize_nyc_dataset(summary_entry, detail_entry):
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
        "download_url": extract_nyc_download_url(str(dataset_id)) if dataset_id is not None else None,
        "landing_page_url": extract_nyc_landing_page_url(str(dataset_id)) if dataset_id is not None else None,
        "record_count_estimate": None,
        "last_updated": str(rows_updated_at) if rows_updated_at is not None else None,
        "license": str(license_name) if license_name is not None else None,
        "sample_rows_json": None,
        "raw_metadata_json": json.dumps(
            {"summary_entry": summary_entry, "detail_entry": detail_entry},
            ensure_ascii=False,
        ),
    }


def fetch_nyc_metadata(limit=100):
    response = requests.get(NYC_CATALOG_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError("Expected NYC Open Data catalog response to be a list.")

    trimmed = payload[:limit]
    records = []

    for entry in trimmed:
        dataset_id = safe_get(entry, "id")
        if not dataset_id:
            continue
        detail_entry = fetch_nyc_dataset_detail(str(dataset_id))
        records.append(normalize_nyc_dataset(entry, detail_entry))

    return records


# ----------------------------
# DATA.GOV HELPERS
# ----------------------------

DATA_GOV_SEARCH_URL = "https://catalog.data.gov/search"


def normalize_data_gov_dataset(entry):
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


def fetch_data_gov_metadata(limit=100):
    params = {
        "q": "",
        "per_page": limit,
    }

    response = requests.get(
        DATA_GOV_SEARCH_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()

    results = payload["results"]
    return [normalize_data_gov_dataset(entry) for entry in results]


# ----------------------------
# MAIN
# ----------------------------

def main():
    spark = SparkSession.builder.appName("DataprocIngestCombinedMetadata").getOrCreate()

    nyc_limit = 100
    data_gov_limit = 100

    print(f"Fetching {nyc_limit} NYC datasets...")
    nyc_records = fetch_nyc_metadata(limit=nyc_limit)
    print(f"Fetched NYC records: {len(nyc_records)}")

    print(f"Fetching {data_gov_limit} Data.gov datasets...")
    data_gov_records = fetch_data_gov_metadata(limit=data_gov_limit)
    print(f"Fetched Data.gov records: {len(data_gov_records)}")

    combined_records = nyc_records + data_gov_records
    print(f"Total combined records: {len(combined_records)}")

    df = spark.createDataFrame(combined_records)

    output_path = "hdfs:///user/km6579_nyu_edu/data/metadata/combined_metadata.parquet"
    df.write.mode("overwrite").parquet(output_path)

    df.groupBy("source").count().show(truncate=False)
    print(f"Saved combined metadata to {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()