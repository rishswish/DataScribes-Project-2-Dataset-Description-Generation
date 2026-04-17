import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType


def safe_load_rows(sample_rows_json: str):
    if sample_rows_json is None:
        return []
    try:
        obj = json.loads(sample_rows_json)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def get_sample_row_count(sample_rows_json: str) -> int:
    rows = safe_load_rows(sample_rows_json)
    return len(rows)


def get_sample_column_count(sample_rows_json: str) -> int:
    rows = safe_load_rows(sample_rows_json)
    if not rows:
        return 0
    first = rows[0]
    if isinstance(first, dict):
        return len(first.keys())
    return 0


def build_sample_profile_json(sample_rows_json: str) -> str:
    rows = safe_load_rows(sample_rows_json)
    if not rows:
        return json.dumps({})

    first = rows[0]
    if not isinstance(first, dict):
        return json.dumps({})

    profile = {}
    columns = list(first.keys())

    for c in columns:
        values = []
        missing = 0

        for row in rows:
            val = row.get(c)
            if val is None or str(val).strip() == "":
                missing += 1
            else:
                values.append(str(val))

        numeric_like = True
        for v in values:
            try:
                float(v)
            except Exception:
                numeric_like = False
                break

        inferred = "numeric_like" if values and numeric_like else "text_like"

        profile[c] = {
            "missing_count": missing,
            "unique_count_sample": len(set(values)),
            "example_values": values[:3],
            "inferred_sample_type": inferred,
        }

    return json.dumps(profile)


sample_row_count_udf = udf(get_sample_row_count, IntegerType())
sample_column_count_udf = udf(get_sample_column_count, IntegerType())
sample_profile_udf = udf(build_sample_profile_json, StringType())


def main():
    spark = (
        SparkSession.builder
        .appName("DatasetDescriptionCombinedProfiling")
        .getOrCreate()
    )

    input_path = "hdfs:///user/rbp5812_nyu_edu/data/metadata/combined_metadata_with_samples.parquet"    
    output_path = "data/profiles/combined_profiles_spark.parquet"

    df = spark.read.parquet(input_path)

    profiled = (
        df.withColumn("sample_row_count", sample_row_count_udf(col("sample_rows_json")))
          .withColumn("sample_column_count", sample_column_count_udf(col("sample_rows_json")))
          .withColumn("sample_columns_profile_json", sample_profile_udf(col("sample_rows_json")))
    )

    profiled.select(
        "dataset_id",
        "title",
        "source",
        "sample_row_count",
        "sample_column_count"
    ).show(20, truncate=False)

    profiled.write.mode("overwrite").parquet(output_path)

    print(f"Saved Spark profiles to {output_path}")
    spark.stop()


if __name__ == "__main__":
    main()