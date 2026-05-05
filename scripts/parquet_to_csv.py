"""
Convert the generated descriptions parquet (10 part files from DataProc)
to a single CSV file expected by the evaluation scripts.

Reads:  data/descriptions/generated_descriptions_sonnet.parquet/
Writes: data/generated_descriptions_all_200.csv
"""

import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/descriptions/generated_descriptions_sonnet.parquet")
CSV_PATH     = Path("data/generated_descriptions_all_200.csv")


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Parquet folder not found at {PARQUET_PATH}. "
            "Download it from HDFS first:\n"
            "  hdfs dfs -get /user/at6370_nyu_edu/data/descriptions/"
            "generated_descriptions_sonnet.parquet ~/\n"
            "Then copy it into data/descriptions/ locally."
        )

    print(f"Reading parquet from {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows  |  "
          f"{df['generated_description'].notna().sum()} successful generations")

    df.groupby("source")["generated_description"].count().reset_index(
        name="count"
    ).apply(lambda r: print(f"  {r['source']}: {r['count']} rows"), axis=1)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
