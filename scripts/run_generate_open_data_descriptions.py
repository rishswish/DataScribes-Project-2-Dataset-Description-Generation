from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

from autoddg import AutoDDG


def run_cmd(cmd: List[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def shrink_sample_csv(
    sample_csv: Optional[str],
    max_chars: int = 8000,
    max_lines: int = 8,
) -> Optional[str]:
    if sample_csv is None:
        return None

    text = str(sample_csv).strip()
    if not text:
        return None

    lines = text.splitlines()
    if not lines:
        return None

    header = lines[0]
    data_lines = lines[1 : 1 + max_lines]
    shrunk = "\n".join([header] + data_lines)

    if len(shrunk) > max_chars:
        shrunk = shrunk[:max_chars]

    return shrunk.strip() if shrunk.strip() else None


def copy_hdfs_parquet_to_local(hdfs_path: str, local_parent_dir: str) -> str:
    local_path = os.path.join(local_parent_dir, Path(hdfs_path).name)
    if os.path.exists(local_path):
        shutil.rmtree(local_path, ignore_errors=True)

    run_cmd(["hdfs", "dfs", "-get", hdfs_path, local_parent_dir])
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="HDFS prepared records parquet")
    parser.add_argument("--output_local_parquet", required=True, help="Local output parquet path")
    parser.add_argument("--model_name", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max_chars", type=int, default=8000, help="Max chars sent to model")
    parser.add_argument("--max_lines", type=int, default=8, help="Max lines sent to model")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_input_parquet = copy_hdfs_parquet_to_local(args.input_parquet, tmpdir)
        records_pd = pd.read_parquet(local_input_parquet)

        client = OpenAI()
        autoddg = AutoDDG(client=client, model_name=args.model_name)

        output_rows: List[Dict[str, Any]] = []
        total = len(records_pd)

        print(f"[INFO] Total prepared records to describe: {total}")

        for idx, row in records_pd.iterrows():
            dataset_id = row.get("dataset_id")
            source = row.get("source")
            title = row.get("title")
            original_description = row.get("original_description")
            download_url = row.get("download_url")
            landing_page_url = row.get("landing_page_url")
            sample_csv = row.get("sample_csv")

            print(
                f"[INFO] Generating description {idx + 1}/{total} "
                f"for source={source} dataset_id={dataset_id}"
            )

            generated_description = None
            generation_status = "error"
            error_message = None

            try:
                shrunk_sample_csv = shrink_sample_csv(
                    sample_csv=sample_csv,
                    max_chars=args.max_chars,
                    max_lines=args.max_lines,
                )

                if shrunk_sample_csv is None:
                    raise ValueError("missing_or_empty_sample_csv_after_shrink")

                _, generated_description = autoddg.describe_dataset(
                    dataset_sample=shrunk_sample_csv
                )
                generation_status = "success"

            except Exception as e:
                error_message = str(e)
                print(
                    f"[ERROR] source={source} dataset_id={dataset_id} "
                    f"error={error_message}"
                )

            output_rows.append(
                {
                    "dataset_id": None if pd.isna(dataset_id) else str(dataset_id),
                    "source": None if pd.isna(source) else str(source),
                    "title": None if pd.isna(title) else str(title),
                    "original_description": None if pd.isna(original_description) else str(original_description),
                    "download_url": None if pd.isna(download_url) else str(download_url),
                    "landing_page_url": None if pd.isna(landing_page_url) else str(landing_page_url),
                    "sample_csv": None if pd.isna(sample_csv) else str(sample_csv),
                    "generated_description": generated_description,
                    "generation_status": generation_status,
                    "error_message": error_message,
                }
            )

    out_df = pd.DataFrame(output_rows)
    output_path = Path(args.output_local_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"[INFO] Wrote {len(out_df)} rows locally to {output_path}")
    print(out_df.groupby(["source", "generation_status"]).size().reset_index(name="count"))


if __name__ == "__main__":
    main()
