import argparse

from src.pipeline.describe.generate_descriptions import generate_descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="Input parquet path")
    parser.add_argument("--output_parquet", required=True, help="Output parquet path")
    parser.add_argument("--model_name", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max_records", type=int, default=None, help="Optional row limit")
    args = parser.parse_args()

    results_df = generate_descriptions(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        model_name=args.model_name,
        max_records=args.max_records,
    )

    print("\nDone.")
    print(results_df[["dataset_id", "title", "source", "status"]])


if __name__ == "__main__":
    main()