import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="HDFS sampled metadata parquet")
    parser.add_argument("--output_parquet", required=True, help="HDFS prepared output parquet")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("prepare_open_data_records").getOrCreate()

    df = spark.read.parquet(args.input_parquet)

    prepared_df = (
        df.filter(F.col("sample_fetch_status") == "success")
        .filter(F.col("sample_csv").isNotNull())
        .filter(F.trim(F.col("sample_csv")) != "")
        .select(
            F.col("dataset_id").cast("string").alias("dataset_id"),
            F.col("title").cast("string").alias("title"),
            F.col("source").cast("string").alias("source"),
            F.col("sample_csv").cast("string").alias("sample_csv"),
            F.col("original_description").cast("string").alias("original_description"),
            F.col("download_url").cast("string").alias("download_url"),
            F.col("landing_page_url").cast("string").alias("landing_page_url"),
        )
    )

    prepared_df.write.mode("overwrite").parquet(args.output_parquet)

    print("[INFO] Prepared records written.")
    print(f"[INFO] Output path: {args.output_parquet}")
    print(f"[INFO] Prepared row count: {prepared_df.count()}")

    prepared_df.groupBy("source").count().show(truncate=False)
    prepared_df.select("dataset_id", "source", "title").show(100, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
