import pandas as pd
from pathlib import Path


def main():
    records = [
        {
            "dataset_id": "test_001",
            "title": "Patient BMI Sample",
            "source": "local_test",
            "sample_csv": (
                "Case_ID,Age,BMI\n"
                "C3L-00004,72,22.8\n"
                "C3L-00010,30,34.15\n"
                "C3L-00012,54,27.40\n"
            ),
        },
        {
            "dataset_id": "test_002",
            "title": "City Temperature Sample",
            "source": "local_test",
            "sample_csv": (
                "City,Month,Temperature\n"
                "New York,Jan,2\n"
                "Chicago,Jan,-3\n"
                "Miami,Jan,20\n"
            ),
        },
    ]

    df = pd.DataFrame(records)
    output_path = Path("data/processed/test_dataset_records.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved: {output_path}")
    print(df)


if __name__ == "__main__":
    main()