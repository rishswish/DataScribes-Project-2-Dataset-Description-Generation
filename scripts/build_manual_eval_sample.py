from pathlib import Path
import pandas as pd


def main():
    df = pd.read_parquet("data/final_dataset_description_results.parquet")

    sample = pd.concat([
        df[df["source"] == "nyc_open_data"].head(5),
        df[df["source"] == "data_gov"].head(5),
    ], ignore_index=True)

    sample = sample[[
        "dataset_id",
        "title",
        "source",
        "original_description",
        "generated_description",
        "declared_column_count",
        "sample_row_count",
        "generation_model",
    ]].copy()

    sample["manual_score_clarity"] = ""
    sample["manual_score_accuracy"] = ""
    sample["manual_score_usefulness"] = ""
    sample["manual_notes"] = ""

    output_csv = Path("data/manual_evaluation_sample.csv")
    output_xlsx = Path("data/manual_evaluation_sample.xlsx")

    sample.to_csv(output_csv, index=False)
    sample.to_excel(output_xlsx, index=False)

    print(f"Saved CSV to {output_csv}")
    print(f"Saved XLSX to {output_xlsx}")
    print(sample[["title", "source", "generation_model"]].to_string(index=False))


if __name__ == "__main__":
    main()