from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests
from pyspark.sql import SparkSession
from pyspark.sql import types as T


SOCRATA_CATALOG_URL = "https://api.us.socrata.com/api/catalog/v1"


def safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_tags(tags: Any) -> Optional[str]:
    if tags is None:
        return None
    if isinstance(tags, list):
        return json.dumps(tags, ensure_ascii=False)
    return str(tags)


def fetch_catalog_page(
    limit: int,
    offset: int,
    search_context: str = "nyc",
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    params = {
        "only": "datasets",
        "limit": limit,
        "offset": offset,
        "search_context": search_context,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(SOCRATA_CATALOG_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            return payload.get("results", [])
        except Exception as e:
            last_error = e
            print(
                f"[WARN] fetch failed for offset={offset}, attempt={attempt}/{retries}: {e}"
            )
            if attempt < retries:
                time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Failed to fetch catalog page at offset={offset} after {retries} attempts"
    ) from last_error


def parse_catalog_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in results:
        resource = item.get("resource", {})
        classification = item.get("classification", {})
        metadata = item.get("metadata", {})

        row = {
            "dataset_id": resource.get("id"),
            "title": resource.get("name"),
            "source": "nyc_open_data",
            "description": resource.get("description"),
            "permalink": resource.get("permalink"),
            "type": resource.get("type"),
            "created_at": resource.get("createdAt"),
            "updated_at": resource.get("updatedAt"),
            "page_views_total": safe_get(metadata, ["page_views", "page_views_total"]),
            "download_count": metadata.get("downloads"),
            "tags_json": normalize_tags(classification.get("tags")),
            "domain": safe_get(item, ["metadata", "domain"]),
        }
        rows.append(row)

    return rows


def fetch_nyc_metadata_rows(
    total_limit: int,
    page_size: int,
    search_context: str = "nyc",
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    offset = 0

    while len(all_rows) < total_limit:
        remaining = total_limit - len(all_rows)
        current_page_size = min(page_size, remaining)

        print(
            f"[INFO] Fetching catalog page: offset={offset}, page_size={current_page_size}"
        )
        results = fetch_catalog_page(
            limit=current_page_size,
            offset=offset,
            search_context=search_context,
        )

        if not results:
            print("[INFO] No more catalog results returned; stopping early.")
            break

        parsed_rows = parse_catalog_results(results)
        all_rows.extend(parsed_rows)

        offset += current_page_size

    return all_rows


def build_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("dataset_id", T.StringType(), True),
            T.StructField("title", T.StringType(), True),
            T.StructField("source", T.StringType(), True),
            T.StructField("description", T.StringType(), True),
            T.StructField("permalink", T.StringType(), True),
            T.StructField("type", T.StringType(), True),
            T.StructField("created_at", T.StringType(), True),
            T.StructField("updated_at", T.StringType(), True),
            T.StructField("page_views_total", T.LongType(), True),
            T.StructField("download_count", T.LongType(), True),
            T.StructField("tags_json", T.StringType(), True),
            T.StructField("domain", T.StringType(), True),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_parquet",
        required=True,
        help="HDFS or local parquet output path",
    )
    parser.add_argument(
        "--total_limit",
        type=int,
        default=100,
        help="Total number of datasets to fetch",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=50,
        help="Catalog API page size per request",
    )
    parser.add_argument(
        "--search_context",
        default="nyc",
        help="Socrata search context",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.appName("fetch_nyc_metadata").getOrCreate()

    rows = fetch_nyc_metadata_rows(
        total_limit=args.total_limit,
        page_size=args.page_size,
        search_context=args.search_context,
    )

    if not rows:
        raise ValueError("No NYC metadata rows were fetched.")

    schema = build_schema()
    df = spark.createDataFrame(rows, schema=schema)

    print(f"[INFO] Writing {df.count()} metadata rows to: {args.output_parquet}")
    df.write.mode("overwrite").parquet(args.output_parquet)

    print("[INFO] Done.")
    df.show(10, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
