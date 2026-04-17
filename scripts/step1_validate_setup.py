from pathlib import Path
import yaml

REQUIRED_DIRS = [
    "configs",
    "ingestion",
    "profiling",
    "llm",
    "evaluation",
    "notebooks",
    "scripts",
    "data/raw",
    "data/metadata",
    "data/profiles",
    "data/descriptions",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "configs/project_config.yaml",
    "ingestion/schema.py",
]

def main():
    root = Path(".")
    missing_dirs = [d for d in REQUIRED_DIRS if not (root / d).exists()]
    missing_files = [f for f in REQUIRED_FILES if not (root / f).exists()]

    if missing_dirs:
        print("Missing directories:")
        for d in missing_dirs:
            print(f"  - {d}")

    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")

    if not missing_dirs and not missing_files:
        print("Folder structure looks good.")

    config_path = root / "configs" / "project_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        print("\nLoaded config:")
        print(f"project_name: {cfg.get('project_name')}")
        print(f"sources: {cfg.get('v1_scope', {}).get('sources')}")
        print(f"dataset_type: {cfg.get('v1_scope', {}).get('dataset_type')}")
        print(f"dataset_count_range: {cfg.get('v1_scope', {}).get('dataset_count_min')} - {cfg.get('v1_scope', {}).get('dataset_count_max')}")
        print(f"engine: {cfg.get('execution', {}).get('engine')}")
        print(f"platform: {cfg.get('execution', {}).get('platform')}")

if __name__ == "__main__":
    main()