from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


REQUIRED_INPUT_COLUMNS = ["dataset_id", "title", "source", "sample_csv"]
OUTPUT_COLUMNS = ["dataset_id", "title", "source", "sample_csv"]


def validate_input_dataframe(df: DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input parquet is missing required columns: {missing_cols}")


def prepare_dataset_records(
    spark: SparkSession,
    input_parquet: str,
    output_parquet: str,
    max_records: Optional[int] = None,
) -> DataFrame:
    df = spark.read.parquet(input_parquet)
    validate_input_dataframe(df)

    records_df = (
        df.select(*OUTPUT_COLUMNS)
        .withColumn("dataset_id", F.col("dataset_id").cast("string"))
        .withColumn("title", F.coalesce(F.col("title").cast("string"), F.lit("")))
        .withColumn("source", F.coalesce(F.col("source").cast("string"), F.lit("")))
        .withColumn("sample_csv", F.coalesce(F.col("sample_csv").cast("string"), F.lit("")))
        .filter(F.trim(F.col("sample_csv")) != "")
    )

    if max_records is not None:
        records_df = records_df.limit(max_records)

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_df.write.mode("overwrite").parquet(output_parquet)
    return records_df