import pandas as pd
import re
import unicodedata
from jiwer import wer, cer

FILES = {
    "Parakeet": r"results/parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test/parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test_results.parquet",
    "Whisper": r"results/whisper.cpp (base)_mslt_test/whisper.cpp (base)_mslt_test_results.parquet",
    "W1-HF": r"experiments/results/w1_hf_controlled/predictions.csv",
}

MANIFEST = r"datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"


def normalize(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


manifest = pd.read_parquet(MANIFEST)
manifest_ids = set(
    manifest["id"].astype(str).str.zfill(4)
)

print("=" * 70)
print("FINAL 2275-SAMPLE EVALUATION")
print("=" * 70)
print(f"Manifest: {len(manifest_ids)}")
print()

for name, path in FILES.items():

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if name == "W1-HF":
        id_col = "id"
        ref_col = "reference"
        hyp_col = "prediction"
    else:
        id_col = "sample_id"
        ref_col = "source_text"
        hyp_col = "hypothesis"

    ids = df[id_col].astype(str).str.zfill(4)

    references = [
        normalize(x)
        for x in df[ref_col]
    ]

    predictions = [
        normalize(x)
        for x in df[hyp_col]
    ]

    missing = manifest_ids - set(ids)

    print(f"{name}")
    print("-" * 70)
    print(f"Rows:        {len(df)}")
    print(f"Unique IDs:  {ids.nunique()}")
    print(f"Missing IDs: {len(missing)}")
    print(f"WER:         {wer(references, predictions) * 100:.2f}%")
    print(f"CER:         {cer(references, predictions) * 100:.2f}%")
    print()

print("=" * 70)
print("DONE")
print("=" * 70)
