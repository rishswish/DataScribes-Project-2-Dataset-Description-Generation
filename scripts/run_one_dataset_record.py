import argparse
import json
from pathlib import Path

from openai import OpenAI
from autoddg import AutoDDG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to dataset record JSON")
    parser.add_argument("--output_json", required=True, help="Path to save result JSON")
    parser.add_argument("--model_name", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    dataset_id = record.get("dataset_id")
    title = record.get("title")
    source = record.get("source")
    sample_csv = record.get("sample_csv")

    if not sample_csv:
        raise ValueError("Input JSON must contain a non-empty 'sample_csv' field.")

    client = OpenAI()
    autoddg = AutoDDG(client=client, model_name=args.model_name)

    prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)

    result = {
        "dataset_id": dataset_id,
        "title": title,
        "source": source,
        "model_name": args.model_name,
        "prompt": prompt,
        "description": description,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nDATASET ID:", dataset_id)
    print("TITLE:", title)
    print("SOURCE:", source)
    print("\nDESCRIPTION:\n")
    print(description)
    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()