"""
BridgeDEUX
Phase 2

CoVoST Text Extractor

Purpose
-------
Extract only the text fields from CoVoST for Machine Translation
benchmarking.

The audio column is removed before iteration because the MT
pipeline does not require audio.

Output:
datasets/cache/covost/de_en/<split>.csv
"""

from pathlib import Path
import csv
from datasets import load_dataset

OUTPUT_DIR = Path("datasets/cache/covost/de_en")


def extract_split(split: str):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{split}.csv"

    print("=" * 70)
    print(f"BridgeDEUX - CoVoST Text Extraction ({split})")
    print("=" * 70)

    print("\nLoading streaming dataset...")

    dataset = load_dataset(
        "fixie-ai/covost2",
        "de_en",
        split=split,
        streaming=True,
    )

    print("\nOriginal columns:")
    print(dataset.column_names)

    # Remove audio if present
    if "audio" in dataset.column_names:
        dataset = dataset.remove_columns("audio")

    print("\nColumns after preprocessing:")
    print(dataset.column_names)

    headers = [
        "id",
        "client_id",
        "file",
        "sentence",
        "translation",
    ]

    total = 0

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)
        writer.writerow(headers)

        for sample in dataset:

            writer.writerow([
                sample.get("id", ""),
                sample.get("client_id", ""),
                sample.get("file", ""),
                sample.get("sentence", ""),
                sample.get("translation", ""),
            ])

            total += 1

            if total % 500 == 0:
                print(f"Extracted {total:,} samples...")

    print("\n" + "=" * 70)
    print("Extraction Complete")
    print("=" * 70)
    print(f"Samples : {total:,}")
    print(f"Saved   : {output_file}")


if __name__ == "__main__":

    extract_split("test")