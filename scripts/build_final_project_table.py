from pathlib import Path
import pandas as pd


def main():
    profiles = pd.read_parquet("data/profiles/combined_profiles.parquet")
    descriptions = pd.read_parquet("data/descriptions/combined_generated_descriptions.parquet")

    final_df = profiles.merge(
        descriptions[["dataset_id", "description_type", "generation_model", "generated_description"]],
        on="dataset_id",
        how="left",
    )

    output_path = Path("data/final_dataset_description_results.parquet")
    final_df.to_parquet(output_path, index=False)

    print(f"Saved final project table to {output_path}")
    print(final_df[[
        "title",
        "source",
        "declared_column_count",
        "sample_row_count",
        "generation_model"
    ]].head(10).to_string())


if __name__ == "__main__":
    main()