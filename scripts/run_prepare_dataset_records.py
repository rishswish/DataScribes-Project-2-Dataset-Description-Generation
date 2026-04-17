import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


REQUIRED_INPUT_COLUMNS = ["dataset_id", "title", "source", "sample_csv"]
OUTPUT_COLUMNS = ["dataset_id", "title", "source", "sample_csv"]


def validate_input_dataframe(df) -> None:
    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input parquet is missing required columns: {missing_cols}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="Input parquet path")
    parser.add_argument("--output_parquet", required=True, help="Output parquet path")
    parser.add_argument("--max_records", type=int, default=None, help="Optional row limit")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("prepare_dataset_records").getOrCreate()

    df = spark.read.parquet(args.input_parquet)
    validate_input_dataframe(df)

    records_df = (
        df.select(*OUTPUT_COLUMNS)
        .withColumn("dataset_id", F.col("dataset_id").cast("string"))
        .withColumn("title", F.coalesce(F.col("title").cast("string"), F.lit("")))
        .withColumn("source", F.coalesce(F.col("source").cast("string"), F.lit("")))
        .withColumn("sample_csv", F.coalesce(F.col("sample_csv").cast("string"), F.lit("")))
        .filter(F.trim(F.col("sample_csv")) != "")
    )

    if args.max_records is not None:
        records_df = records_df.limit(args.max_records)

    records_df.write.mode("overwrite").parquet(args.output_parquet)

    print("\nDone.")
    print(f"Total prepared records: {records_df.count()}")
    records_df.show(10, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()