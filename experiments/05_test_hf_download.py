import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

print("=" * 70)
print("BridgeDEUX - Diagnostic: Direct Hugging Face Download")
print("=" * 70)

start_time = time.perf_counter()

try:
    print("\nDownloading Parquet shard...")

    file_path = hf_hub_download(
        repo_id="fixie-ai/covost2",
        filename="de_en/test-00000-of-00008.parquet",
        repo_type="dataset",
    )

    download_time = time.perf_counter() - start_time

    print("\nDownload completed.")
    print(f"Time      : {download_time:.2f} seconds")
    print(f"Cached at : {file_path}")

    print("\nReading only required columns...")

    read_start = time.perf_counter()

    df = pd.read_parquet(
        file_path,
        columns=[
            "sentence",
            "translation"
        ]
    )

    read_time = time.perf_counter() - read_start

    print("\nDataset Information")
    print("-" * 70)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {list(df.columns)}")
    print(f"Read Time : {read_time:.2f} seconds")

    print("\nFirst Three Samples")
    print("-" * 70)
    print(df.head(3))

    print("\nDiagnostic Result")
    print("-" * 70)
    print("SUCCESS")
    print("Direct Parquet download works correctly.")

except Exception as e:
    print("\nFAILED")
    print(e)