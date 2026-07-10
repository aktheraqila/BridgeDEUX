import csv
from pathlib import Path

# --------------------------------------------------
# Dataset Path
# --------------------------------------------------

DATASET_PATH = Path(
    "datasets/Choice1_covost_v2.de_en.tsv/covost_v2.de_en.tsv"
)

# --------------------------------------------------
# Inspector
# --------------------------------------------------

def inspect_dataset():

    print("=" * 60)
    print("CoVoST Dataset Inspector")
    print("=" * 60)

    print(f"\nDataset Path:")
    print(DATASET_PATH.resolve())

    if not DATASET_PATH.exists():
        print("\n❌ Dataset not found.")
        return

    print("\n✅ Dataset found.")

    file_size_mb = DATASET_PATH.stat().st_size / (1024 * 1024)

    print(f"File Size : {file_size_mb:.2f} MB")

    with open(
        DATASET_PATH,
        mode="r",
        encoding="utf-8"
    ) as file:

        reader = csv.reader(
            file,
            delimiter="\t"
        )

        headers = next(reader)

        print("\nColumn Names")
        print("-" * 60)

        for i, header in enumerate(headers):
            print(f"{i+1}. {header}")

        print("\nFirst 5 Rows")
        print("-" * 60)

        row_count = 0

        for row in reader:

            if row_count < 5:
                print(row)

            row_count += 1

    print("\nApproximate Statistics")
    print("-" * 60)

    print(f"Columns : {len(headers)}")
    print(f"Rows    : {row_count}")

    print("\nInspection Complete.")


# --------------------------------------------------

if __name__ == "__main__":
    inspect_dataset()