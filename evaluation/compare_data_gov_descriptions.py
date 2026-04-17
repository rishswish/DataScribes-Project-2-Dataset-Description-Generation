from pathlib import Path
import pandas as pd


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip().replace("\n", " ")


def main():
    profiles = pd.read_parquet("data/profiles/data_gov_profiles.parquet")
    generated = pd.read_parquet("data/descriptions/data_gov_generated_descriptions_v2.parquet")

    base = profiles[["dataset_id", "title", "original_description"]].copy()
    base["original_description"] = base["original_description"].apply(clean_text)

    generated = generated[["dataset_id", "generated_description"]].copy()
    generated["generated_description"] = generated["generated_description"].apply(clean_text)

    out = base.merge(generated, on="dataset_id", how="left")
    out["original_len"] = out["original_description"].str.len()
    out["generated_len"] = out["generated_description"].str.len()

    output_path = Path("data/descriptions/data_gov_description_comparison.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved comparison file to {output_path}")
    print(out[["title", "original_len", "generated_len"]].head().to_string())


if __name__ == "__main__":
    main()