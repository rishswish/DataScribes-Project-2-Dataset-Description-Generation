import json
import os
import time

import anthropic
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType


def build_prompt(row) -> str:
    title = row.title or "Untitled Dataset"
    source = row.source or ""
    original_description = (row.original_description or "").strip()[:500]
    keywords = json.loads(row.keywords_json or "[]")
    column_names = json.loads(row.column_names_json or "[]")
    sample_rows = json.loads(row.sample_rows_json or "[]") if row.sample_rows_json else []
    sample_profile = json.loads(row.sample_columns_profile_json or "{}") if row.sample_columns_profile_json else {}

    lines = []
    lines.append(f"Dataset Title: {title}")
    lines.append(f"Source: {source}")

    if original_description:
        lines.append(f"Original Description: {original_description}")

    if keywords:
        lines.append(f"Keywords: {', '.join(keywords[:8])}")

    if column_names:
        lines.append(f"Columns ({len(column_names)} total): {', '.join(column_names[:12])}")

    if sample_profile:
        profile_lines = []
        for col_name, stats in list(sample_profile.items())[:6]:
            examples = stats.get("example_values", [])[:2]
            inferred = stats.get("inferred_sample_type", "")
            profile_lines.append(f"  - {col_name} ({inferred}): e.g. {', '.join(str(v) for v in examples)}")
        if profile_lines:
            lines.append("Column Profiles:\n" + "\n".join(profile_lines))

    if sample_rows and isinstance(sample_rows, list) and sample_rows:
        lines.append(f"Sample row: {json.dumps(sample_rows[0], ensure_ascii=False)[:300]}")

    context = "\n".join(lines)

    prompt = f"""You are a data catalog assistant. Write a clear, concise description (3-5 sentences) for the following dataset.
The description should help a data analyst quickly understand what the dataset contains, what it can be used for, and any notable characteristics.
Do not copy the original description verbatim. Be specific and informative.

{context}

Description:"""

    return prompt


def generate_description_for_partition(rows, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    out = []

    for row in rows:
        try:
            prompt = build_prompt(row)
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            description = message.content[0].text.strip()
            error = None
        except Exception as e:
            description = None
            error = str(e)
            time.sleep(2)

        out.append({
            "dataset_id": row.dataset_id,
            "source": row.source,
            "title": row.title,
            "original_description": row.original_description,
            "generated_description": description,
            "generation_model": "claude-sonnet-4-5",
            "generation_error": error,
        })

    return iter(out)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    spark = SparkSession.builder.appName("DataprocGenerateDescriptions").getOrCreate()

    input_path = "hdfs:///user/km6579_nyu_edu/data/profiles/combined_profiles_spark_v2.parquet"
    output_path = "hdfs:///user/km6579_nyu_edu/data/descriptions/generated_descriptions_sonnet.parquet"

    df = spark.read.parquet(input_path)
    print(f"Loaded {df.count()} rows from profiles")

    # Use 10 partitions — enough parallelism without hammering the API
    rdd = df.rdd.repartition(10)

    api_key_broadcast = spark.sparkContext.broadcast(api_key)

    enriched_rdd = rdd.mapPartitions(
        lambda rows: generate_description_for_partition(rows, api_key_broadcast.value)
    )

    from pyspark.sql.types import StructType, StructField, StringType as ST
    schema = StructType([
        StructField("dataset_id", ST(), True),
        StructField("source", ST(), True),
        StructField("title", ST(), True),
        StructField("original_description", ST(), True),
        StructField("generated_description", ST(), True),
        StructField("generation_model", ST(), True),
        StructField("generation_error", ST(), True),
    ])

    out_df = spark.createDataFrame(enriched_rdd, schema=schema)
    out_df.write.mode("overwrite").parquet(output_path)

    out_df.groupBy("source", "generation_model").count().show(truncate=False)
    print(f"Saved generated descriptions to {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
