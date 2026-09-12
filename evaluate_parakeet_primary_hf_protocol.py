"""
Evaluate Parakeet using the PRIMARY PyTorch/Hugging Face KD scoring protocol.

Important:
- This does NOT run Parakeet through Hugging Face.
- It takes the existing Parakeet.cpp predictions and scores them with
  the SAME MSLT T1 reference + normalization + corpus-level WER/CER
  used by evaluate_w0_w1_w2.py.

Expected:
  Test manifest:
    datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet

  Parakeet predictions:
    results/parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test/
      parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test_results.parquet
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd
from jiwer import wer, cer


TEST_PARQUET = Path(
    "datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
)

PARAKEET_PARQUET = Path(
    "results/parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test/"
    "parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test_results.parquet"
)


def normalize_mslt_t1(text: str) -> str:
    text = str(text)

    # Same normalization as evaluate_w0_w1_w2.py:
    # remove MSLT annotation tags
    text = re.sub(r"<[^>]*>", " ", text)

    # remove punctuation
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )

    # normalize whitespace and case
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def main():
    print("=" * 72)
    print("PARAKEET — PRIMARY HF/T1 PROTOCOL EVALUATION")
    print("=" * 72)

    if not TEST_PARQUET.exists():
        raise FileNotFoundError(f"Missing test manifest: {TEST_PARQUET}")

    if not PARAKEET_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing Parakeet predictions: {PARAKEET_PARQUET}"
        )

    test = pd.read_parquet(TEST_PARQUET)
    para = pd.read_parquet(PARAKEET_PARQUET)

    print(f"Manifest rows : {len(test)}")
    print(f"Parakeet rows : {len(para)}")

    required_test = {"id", "t1_reference"}
    required_para = {"sample_id", "hypothesis"}

    missing_test = required_test - set(test.columns)
    missing_para = required_para - set(para.columns)

    if missing_test:
        raise RuntimeError(
            f"Manifest missing columns: {sorted(missing_test)}"
        )

    if missing_para:
        raise RuntimeError(
            f"Parakeet output missing columns: {sorted(missing_para)}"
        )

    # Normalize IDs exactly as the project's final evaluation script does.
    test["join_id"] = test["id"].astype(str).str.zfill(4)
    para["join_id"] = para["sample_id"].astype(str).str.zfill(4)

    print(f"Manifest unique IDs  : {test['join_id'].nunique()}")
    print(f"Parakeet unique IDs  : {para['join_id'].nunique()}")

    if test["join_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs found in test manifest.")

    if para["join_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs found in Parakeet output.")

    merged = test[["join_id", "t1_reference"]].merge(
        para[["join_id", "hypothesis"]],
        on="join_id",
        how="inner",
        validate="one_to_one",
    )

    missing = sorted(
        set(test["join_id"]) - set(para["join_id"])
    )
    extra = sorted(
        set(para["join_id"]) - set(test["join_id"])
    )

    print(f"Matched samples     : {len(merged)}")
    print(f"Missing predictions : {len(missing)}")
    print(f"Extra predictions   : {len(extra)}")

    if missing:
        print("First missing IDs:", missing[:10])

    if extra:
        print("First extra IDs:", extra[:10])

    if len(merged) != len(test):
        raise RuntimeError(
            "Not all 2,275 test samples matched. "
            "Do not report this as the final Parakeet score."
        )

    references = [
        normalize_mslt_t1(x)
        for x in merged["t1_reference"]
    ]

    hypotheses = [
        normalize_mslt_t1(x)
        for x in merged["hypothesis"]
    ]

    corpus_wer = wer(references, hypotheses)
    corpus_cer = cer(references, hypotheses)

    exact = sum(
        ref == hyp
        for ref, hyp in zip(references, hypotheses)
    )

    empty = sum(
        not hyp
        for hyp in hypotheses
    )

    print()
    print("-" * 72)
    print("RESULT")
    print("-" * 72)
    print(f"Samples       : {len(merged)}")
    print(f"WER           : {corpus_wer * 100:.2f}%")
    print(f"CER           : {corpus_cer * 100:.2f}%")
    print(f"Exact matches : {exact}")
    print(f"Empty outputs : {empty}")
    print()
    print("This score uses:")
    print("  Reference : MSLT t1_reference")
    print("  Normalize : normalize_mslt_t1()")
    print("  Metric    : corpus-level jiwer WER/CER")
    print()
    print("It is intended to be compared directly with:")
    print("  W0 = 42.55%")
    print("  W1 = 31.78%")
    print("  W2 = 34.83%")
    print("=" * 72)


if __name__ == "__main__":
    main()
