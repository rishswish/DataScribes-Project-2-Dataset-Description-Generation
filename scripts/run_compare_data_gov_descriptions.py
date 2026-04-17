import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from evaluation.compare_data_gov_descriptions import main

if __name__ == "__main__":
    main()