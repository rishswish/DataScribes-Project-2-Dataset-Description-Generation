from pathlib import Path
import pandas as pd


def main():
    nyc_profiles = pd.read_parquet("data/profiles/nyc_open_data_profiles.parquet")
    data_gov_profiles = pd.read_parquet("data/profiles/data_gov_profiles.parquet")

    combined_profiles = pd.concat(
        [nyc_profiles, data_gov_profiles],
        ignore_index=True
    )

    out_profiles = Path("data/profiles/combined_profiles.parquet")
    out_profiles.parent.mkdir(parents=True, exist_ok=True)
    combined_profiles.to_parquet(out_profiles, index=False)

    nyc_desc = pd.read_parquet("data/descriptions/nyc_open_data_generated_descriptions_v2.parquet")
    data_gov_desc = pd.read_parquet("data/descriptions/data_gov_generated_descriptions_v2.parquet")

    combined_desc = pd.concat(
        [nyc_desc, data_gov_desc],
        ignore_index=True
    )

    out_desc = Path("data/descriptions/combined_generated_descriptions.parquet")
    out_desc.parent.mkdir(parents=True, exist_ok=True)
    combined_desc.to_parquet(out_desc, index=False)

    print(f"Saved combined profiles to {out_profiles}")
    print(f"Saved combined descriptions to {out_desc}")
    print("\nProfile counts by source:")
    print(combined_profiles["source"].value_counts())
    print("\nDescription counts by source:")
    print(combined_desc.merge(
        combined_profiles[["dataset_id", "source"]],
        on="dataset_id",
        how="left"
    )["source"].value_counts())


if __name__ == "__main__":
    main()