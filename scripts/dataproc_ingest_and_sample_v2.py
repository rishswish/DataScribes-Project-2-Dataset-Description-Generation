import json
import math
import requests
from io import StringIO

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


NYC_CATALOG_URL = "https://data.cityofnewyork.us/api/views.json"
NYC_VIEW_URL_TEMPLATE = "https://data.cityofnewyork.us/api/views/{dataset_id}.json"
NYC_JSON_RESOURCE_TEMPLATE = "https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit={limit}"

DATA_GOV_SEARCH_URL = "https://catalog.data.gov/search"


def safe_get(dct, key, default=None):
    return dct[key] if key in dct else default


# ----------------------------
# STAGE 1: FETCH LIGHTWEIGHT LISTS
# ----------------------------

def fetch_nyc_index(limit=100):
    resp = requests.get(NYC_CATALOG_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    payload = resp.json()
    trimmed = payload[:limit]

    records = []
    for entry in trimmed:
        dataset_id = safe_get(entry, "id")
        if not dataset_id:
            continue
        records.append({
            "dataset_id": str(dataset_id),
            "source": "nyc_open_data",
            "summary_json": json.dumps(entry, ensure_ascii=False),
            "download_url": f"https://data.cityofnewyork.us/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD",
        })
    return records


def fetch_data_gov_index(limit=100):
    params = {"q": "", "per_page": limit}
    resp = requests.get(
        DATA_GOV_SEARCH_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    payload = resp.json()

    records = []
    for entry in payload["results"]:
        dcat = entry.get("dcat", {}) or {}
        dataset_id = entry.get("identifier") or dcat.get("identifier")
        if not dataset_id:
            continue

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

        records.append({
            "dataset_id": str(dataset_id),
            "source": "data_gov",
            "summary_json": json.dumps(entry, ensure_ascii=False),
            "download_url": csv_url,
        })
    return records


# ----------------------------
# STAGE 2: PARALLEL DETAIL + SAMPLE FETCH
# ----------------------------

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


def fetch_nyc_json_sample_rows(dataset_id, limit=5):
    # Uses Socrata JSON API to fetch exactly N rows — much faster than downloading full CSV
    url = NYC_JSON_RESOURCE_TEMPLATE.format(dataset_id=dataset_id, limit=limit)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None, [], []
        cols = list(rows[0].keys())
        return None, rows, cols
    except Exception as e:
        return str(e), [], []


def fetch_csv_sample_rows(url, limit=5):
    # Streams the CSV and stops after collecting enough lines — avoids downloading the full file
    if not url:
        return None, [], []

    try:
        resp = requests.get(url, timeout=30, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        lines = []
        for line in resp.iter_lines():
            if line:
                lines.append(line.decode("utf-8", errors="replace"))
            if len(lines) >= limit + 1:  # header + limit rows
                break
        resp.close()

        if not lines:
            return None, [], []

        df = pd.read_csv(StringIO("\n".join(lines)))
        rows = df.to_dict(orient="records")
        cols = list(df.columns)
        return None, rows, cols
    except Exception as e:
        return str(e), [], []


def process_partition(rows):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    out = []

    for row in rows:
        dataset_id = row["dataset_id"]
        source = row["source"]
        summary = json.loads(row["summary_json"])
        download_url = row.get("download_url")

        try:
            if source == "nyc_open_data":
                detail_url = NYC_VIEW_URL_TEMPLATE.format(dataset_id=dataset_id)
                detail_resp = session.get(detail_url, timeout=30)
                detail_resp.raise_for_status()
                detail = detail_resp.json()

                category = safe_get(summary, "category")
                tags = safe_get(summary, "tags", [])
                rows_updated_at = safe_get(summary, "rowsUpdatedAt")
                license_info = safe_get(detail, "license")
                license_name = license_info.get("name") if isinstance(license_info, dict) else None
                detail_columns = safe_get(detail, "columns", [])

                # Use JSON API instead of CSV for NYC sample rows
                sample_error, sample_rows, sample_cols = fetch_nyc_json_sample_rows(dataset_id, limit=5)

                out.append({
                    "dataset_id": dataset_id,
                    "source": source,
                    "title": safe_get(summary, "name"),
                    "original_description": safe_get(summary, "description"),
                    "keywords_json": json.dumps(extract_keywords(category, tags), ensure_ascii=False),
                    "column_names_json": json.dumps(extract_column_names(detail_columns), ensure_ascii=False),
                    "column_types_raw_json": json.dumps(extract_column_types(detail_columns), ensure_ascii=False),
                    "download_url": download_url,
                    "landing_page_url": f"https://data.cityofnewyork.us/d/{dataset_id}",
                    "record_count_estimate": None,
                    "last_updated": str(rows_updated_at) if rows_updated_at is not None else None,
                    "license": license_name,
                    "sample_rows_json": json.dumps(sample_rows, ensure_ascii=False) if sample_rows else None,
                    "sample_error": sample_error,
                    "raw_metadata_json": json.dumps({"summary_entry": summary, "detail_entry": detail}, ensure_ascii=False),
                })

            else:
                dcat = summary.get("dcat", {}) or {}
                tags = summary.get("keyword") or dcat.get("keyword") or []
                modified = dcat.get("modified")
                license_url = dcat.get("license")
                landing_page = dcat.get("landingPage")
                title = summary.get("title") or dcat.get("title")
                description = summary.get("description") or dcat.get("description")

                # Stream CSV for Data.gov sample rows
                sample_error, sample_rows, sample_cols = fetch_csv_sample_rows(download_url, limit=5)

                out.append({
                    "dataset_id": dataset_id,
                    "source": source,
                    "title": title,
                    "original_description": description,
                    "keywords_json": json.dumps([str(x) for x in tags] if isinstance(tags, list) else [], ensure_ascii=False),
                    "column_names_json": json.dumps(sample_cols, ensure_ascii=False),
                    "column_types_raw_json": json.dumps([], ensure_ascii=False),
                    "download_url": download_url,
                    "landing_page_url": landing_page,
                    "record_count_estimate": None,
                    "last_updated": str(modified) if modified is not None else None,
                    "license": str(license_url) if license_url is not None else None,
                    "sample_rows_json": json.dumps(sample_rows, ensure_ascii=False) if sample_rows else None,
                    "sample_error": sample_error,
                    "raw_metadata_json": json.dumps(summary, ensure_ascii=False),
                })

        except Exception as e:
            out.append({
                "dataset_id": dataset_id,
                "source": source,
                "title": None,
                "original_description": None,
                "keywords_json": json.dumps([], ensure_ascii=False),
                "column_names_json": json.dumps([], ensure_ascii=False),
                "column_types_raw_json": json.dumps([], ensure_ascii=False),
                "download_url": download_url,
                "landing_page_url": None,
                "record_count_estimate": None,
                "last_updated": None,
                "license": None,
                "sample_rows_json": None,
                "sample_error": f"partition_error: {str(e)}",
                "raw_metadata_json": row["summary_json"],
            })

    return iter(out)


def main():
    spark = SparkSession.builder.appName("DataprocIngestAndSampleCombined").getOrCreate()
    sc = spark.sparkContext

    nyc_limit = 100
    data_gov_limit = 100

    print(f"Fetching NYC index: {nyc_limit}")
    nyc_index = fetch_nyc_index(limit=nyc_limit)

    print(f"Fetching Data.gov index: {data_gov_limit}")
    data_gov_index = fetch_data_gov_index(limit=data_gov_limit)

    combined_index = nyc_index + data_gov_index
    print(f"Total index size: {len(combined_index)}")

    # Increased partitions for better parallelism across cluster workers
    partitions = max(40, math.ceil(len(combined_index) / 5))
    rdd = sc.parallelize(combined_index, partitions)
    enriched_rdd = rdd.mapPartitions(process_partition)

    schema = StructType([
        StructField("dataset_id", StringType(), True),
        StructField("source", StringType(), True),
        StructField("title", StringType(), True),
        StructField("original_description", StringType(), True),
        StructField("keywords_json", StringType(), True),
        StructField("column_names_json", StringType(), True),
        StructField("column_types_raw_json", StringType(), True),
        StructField("download_url", StringType(), True),
        StructField("landing_page_url", StringType(), True),
        StructField("record_count_estimate", StringType(), True),
        StructField("last_updated", StringType(), True),
        StructField("license", StringType(), True),
        StructField("sample_rows_json", StringType(), True),
        StructField("sample_error", StringType(), True),
        StructField("raw_metadata_json", StringType(), True),
    ])

    df = spark.createDataFrame(enriched_rdd, schema=schema)

    output_path = "hdfs:///user/km6579_nyu_edu/data/metadata/combined_metadata_with_samples_v2.parquet"
    df.write.mode("overwrite").parquet(output_path)

    df.groupBy("source").count().show(truncate=False)
    print(f"Saved enriched combined metadata to {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
