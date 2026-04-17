from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    LongType,
)

NORMALIZED_METADATA_SCHEMA = StructType([
    StructField("dataset_id", StringType(), True),
    StructField("source", StringType(), True),
    StructField("title", StringType(), True),
    StructField("original_description", StringType(), True),
    StructField("keywords", ArrayType(StringType()), True),
    StructField("column_names", ArrayType(StringType()), True),
    StructField("column_types_raw", ArrayType(StringType()), True),
    StructField("download_url", StringType(), True),
    StructField("landing_page_url", StringType(), True),
    StructField("record_count_estimate", LongType(), True),
    StructField("last_updated", StringType(), True),
    StructField("license", StringType(), True),
    StructField("sample_rows_json", StringType(), True),
    StructField("raw_metadata_json", StringType(), True),
])