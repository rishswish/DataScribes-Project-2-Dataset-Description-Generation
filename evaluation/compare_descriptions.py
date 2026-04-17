from pathlib import Path
import pandas as pd


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip().replace("\n", " ")


def main():
    profiles = pd.read_parquet("data/profiles/nyc_open_data_profiles.parquet")
    v1 = pd.read_parquet("data/descriptions/nyc_open_data_generated_descriptions.parquet")
    v2 = pd.read_parquet("data/descriptions/nyc_open_data_generated_descriptions_v2.parquet")

    base = profiles[["dataset_id", "title", "original_description"]].copy()
    base["original_description"] = base["original_description"].apply(clean_text)

    v1 = v1[["dataset_id", "generated_description"]].rename(
        columns={"generated_description": "generated_description_v1"}
    )
    v1["generated_description_v1"] = v1["generated_description_v1"].apply(clean_text)

    v2 = v2[["dataset_id", "generated_description"]].rename(
        columns={"generated_description": "generated_description_v2"}
    )
    v2["generated_description_v2"] = v2["generated_description_v2"].apply(clean_text)

    out = base.merge(v1, on="dataset_id", how="left").merge(v2, on="dataset_id", how="left")

    out["original_len"] = out["original_description"].str.len()
    out["v1_len"] = out["generated_description_v1"].str.len()
    out["v2_len"] = out["generated_description_v2"].str.len()

    output_path = Path("data/descriptions/nyc_open_data_description_comparison.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved comparison file to {output_path}")
    print(out[[
        "title", "original_len", "v1_len", "v2_len"
    ]].head().to_string())


if __name__ == "__main__":
    main()