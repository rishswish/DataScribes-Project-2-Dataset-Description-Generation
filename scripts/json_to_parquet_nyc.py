import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pyspark.sql import SparkSession
from ingestion.schema import NORMALIZED_METADATA_SCHEMA


def main():
    spark = (
        SparkSession.builder
        .appName("NYCOpenDataMetadataToParquet")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .getOrCreate()
    )

    with open("data/raw/nyc_open_data_metadata.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    df = spark.createDataFrame(records, schema=NORMALIZED_METADATA_SCHEMA)

    # keep only lightweight operations
    print("row_count =", df.count())
    df.printSchema()

    output_path = "data/metadata/nyc_open_data_metadata.parquet"
    df.write.mode("overwrite").parquet(output_path)

    print(f"Saved parquet to {output_path}")
    spark.stop()


if __name__ == "__main__":
    main()