
from __future__ import annotations

import argparse
import csv
import io
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.sql import types as T
import sys
csv.field_size_limit(sys.maxsize)

def fetch_csv_sample_text(
    download_url: str,
    max_rows: int = 5,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    request_timeout: int = 20,
    max_elapsed_seconds: int = 30,
) -> Optional[str]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        start_time = time.time()

        try:
            with requests.get(
                download_url,
                timeout=request_timeout,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as response:
                response.raise_for_status()

                lines: List[str] = []

                for line in response.iter_lines(decode_unicode=True):
                    if time.time() - start_time > max_elapsed_seconds:
                        raise TimeoutError(
                            f"sample fetch exceeded {max_elapsed_seconds}s"
                        )

                    if line is None:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    lines.append(line)

                    # header + max_rows data rows
                    if len(lines) >= max_rows + 1:
                        break

                if not lines:
                    return None

                reader = csv.reader(io.StringIO("\n".join(lines)))
                rows = list(reader)

                if not rows:
                    return None

                header = rows[0]
                data_rows = rows[1:]

                output = io.StringIO()
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(data_rows)

                sample_csv = output.getvalue().strip()
                return sample_csv if sample_csv else None

        except Exception as e:
            last_error = e
            print(
                f"[WARN] sample fetch failed attempt={attempt}/{retries} "
                f"url={download_url}: {e}"
            )
            if attempt < retries:
                time.sleep(sleep_seconds)

    print(f"[ERROR] giving up on url={download_url}: {last_error}")
    return None


def build_output_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("dataset_id", T.StringType(), True),
            T.StructField("source", T.StringType(), True),
            T.StructField("title", T.StringType(), True),
            T.StructField("original_description", T.StringType(), True),
            T.StructField("download_url", T.StringType(), True),
            T.StructField("landing_page_url", T.StringType(), True),
            T.StructField("sample_csv", T.StringType(), True),
            T.StructField("sample_fetch_status", T.StringType(), True),
            T.StructField("error_message", T.StringType(), True),
        ]
    )


def normalize_output_rows(output_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    required_cols = [
        "dataset_id",
        "source",
        "title",
        "original_description",
        "download_url",
        "landing_page_url",
        "sample_csv",
        "sample_fetch_status",
        "error_message",
    ]

    out_pd = pd.DataFrame(output_rows)

    for col in required_cols:
        if col not in out_pd.columns:
            out_pd[col] = None

    out_pd = out_pd[required_cols]

    for col in required_cols:
        out_pd[col] = out_pd[col].where(pd.notnull(out_pd[col]), None)

    return out_pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="HDFS input metadata parquet")
    parser.add_argument("--output_parquet", required=True, help="HDFS output sample parquet")
    parser.add_argument("--max_rows", type=int, default=5, help="Max sample rows per dataset")
    parser.add_argument("--request_timeout", type=int, default=20, help="Per-request timeout in seconds")
    parser.add_argument("--max_elapsed_seconds", type=int, default=30, help="Per-dataset wall-clock limit in seconds")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("fetch_open_data_samples").getOrCreate()

    df = spark.read.parquet(args.input_parquet)

    metadata_pd = (
        df.select(
            "dataset_id",
            "source",
            "title",
            "original_description",
            "download_url",
            "landing_page_url",
        )
        .toPandas()
    )

    output_rows: List[Dict[str, Any]] = []
    total = len(metadata_pd)

    print(f"[INFO] Total metadata rows to process: {total}")

    for idx, row in metadata_pd.iterrows():
        dataset_id = row.get("dataset_id")
        source = row.get("source")
        title = row.get("title")
        original_description = row.get("original_description")
        download_url = row.get("download_url")
        landing_page_url = row.get("landing_page_url")

        print(
            f"[INFO] Fetching sample {idx + 1}/{total} "
            f"for source={source} dataset_id={dataset_id}"
        )

        sample_csv = None
        error_message = None
        status = "error"

        try:
            if download_url is None or not str(download_url).strip():
                raise ValueError("missing_download_url")

            sample_csv = fetch_csv_sample_text(
                download_url=str(download_url),
                max_rows=args.max_rows,
                request_timeout=args.request_timeout,
                max_elapsed_seconds=args.max_elapsed_seconds,
            )

            if sample_csv is None or not str(sample_csv).strip():
                raise ValueError("empty_or_unreadable_sample")

            status = "success"

        except Exception as e:
            error_message = str(e)
            print(
                f"[ERROR] source={source} dataset_id={dataset_id} "
                f"error={error_message}"
            )

        output_rows.append(
            {
                "dataset_id": None if dataset_id is None else str(dataset_id),
                "source": None if source is None else str(source),
                "title": None if title is None else str(title),
                "original_description": None if original_description is None else str(original_description),
                "download_url": None if download_url is None else str(download_url),
                "landing_page_url": None if landing_page_url is None else str(landing_page_url),
                "sample_csv": sample_csv,
                "sample_fetch_status": status,
                "error_message": error_message,
            }
        )

    if not output_rows:
        raise ValueError("No output rows were produced.")

    out_pd = normalize_output_rows(output_rows)
    schema = build_output_schema()
    out_spark = spark.createDataFrame(out_pd, schema=schema)

    out_spark.write.mode("overwrite").parquet(args.output_parquet)

    print(f"[INFO] Wrote {len(out_pd)} sampled datasets to {args.output_parquet}")
    out_spark.groupBy("source", "sample_fetch_status").count().show(truncate=False)
    out_spark.select(
        "dataset_id",
        "source",
        "title",
        "sample_fetch_status",
        "error_message",
    ).show(100, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
