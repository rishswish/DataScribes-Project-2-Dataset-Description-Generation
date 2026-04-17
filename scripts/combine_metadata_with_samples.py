from pathlib import Path
import pandas as pd


def main():
    nyc = pd.read_parquet("data/metadata/nyc_open_data_metadata_with_samples.parquet")
    data_gov = pd.read_parquet("data/metadata/data_gov_metadata_with_samples.parquet")

    combined = pd.concat([nyc, data_gov], ignore_index=True)

    output_path = Path("data/metadata/combined_metadata_with_samples.parquet")
    combined.to_parquet(output_path, index=False)

    print(f"Saved combined metadata with samples to {output_path}")
    print(combined[["title", "source"]].head(10).to_string())


if __name__ == "__main__":
    main()