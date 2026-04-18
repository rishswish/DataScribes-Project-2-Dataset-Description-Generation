import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="HDFS generated descriptions parquet")
    parser.add_argument("--output_parquet", required=True, help="HDFS evaluation parquet")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("build_evaluation_table_spark").getOrCreate()

    df = spark.read.parquet(args.input_parquet)

    eval_df = (
        df.withColumn(
            "has_original_description",
            F.when(
                F.col("original_description").isNotNull() &
                (F.trim(F.col("original_description")) != ""),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        .withColumn(
            "original_description_len",
            F.length(F.coalesce(F.col("original_description"), F.lit("")))
        )
        .withColumn(
            "generated_description_len",
            F.length(F.coalesce(F.col("generated_description"), F.lit("")))
        )
        .select(
            "dataset_id",
            "source",
            "title",
            "has_original_description",
            "original_description",
            "generated_description",
            "original_description_len",
            "generated_description_len",
            "generation_status",
            "error_message",
        )
    )

    eval_df.write.mode("overwrite").parquet(args.output_parquet)

    print("[INFO] Evaluation table written.")
    print(f"[INFO] Output path: {args.output_parquet}")
    print(f"[INFO] Row count: {eval_df.count()}")

    eval_df.groupBy("source", "generation_status").count().show(truncate=False)
    eval_df.groupBy("source", "has_original_description").count().show(truncate=False)

    eval_df.select(
        "dataset_id",
        "source",
        "title",
        "original_description_len",
        "generated_description_len",
    ).show(50, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
