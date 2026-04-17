import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ingestion.nyc_open_data import main

if __name__ == "__main__":
    main(limit=10, output_path="data/raw/nyc_open_data_metadata.json")