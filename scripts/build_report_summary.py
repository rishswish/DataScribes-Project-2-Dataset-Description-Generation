from pathlib import Path
import pandas as pd


def main():
    df = pd.read_parquet("data/final_dataset_description_results.parquet")

    summary_rows = []

    for source, group in df.groupby("source"):
        summary_rows.append({
            "source": source,
            "num_datasets": len(group),
            "avg_declared_column_count": round(group["declared_column_count"].mean(), 2),
            "avg_sample_row_count": round(group["sample_row_count"].mean(), 2),
            "num_with_generated_description": int(group["generated_description"].notna().sum()),
        })

    summary_df = pd.DataFrame(summary_rows)

    output_csv = Path("data/report_summary_table.csv")
    output_xlsx = Path("data/report_summary_table.xlsx")

    summary_df.to_csv(output_csv, index=False)
    summary_df.to_excel(output_xlsx, index=False)

    print(f"Saved CSV to {output_csv}")
    print(f"Saved XLSX to {output_xlsx}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()