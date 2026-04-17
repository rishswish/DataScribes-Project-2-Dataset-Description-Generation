import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from profiling.profile_nyc_metadata import main

if __name__ == "__main__":
    main()