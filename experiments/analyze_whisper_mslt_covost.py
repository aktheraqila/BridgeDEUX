import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

MSLT = Path(r"results/whisper.cpp (base)_mslt_test/whisper.cpp (base)_mslt_test_results.csv")
COVOST = Path(r"results/whisper.cpp (base)_test/whisper.cpp (base)_test_results.csv")

def normalize(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def analyze(path, name):
    df = pd.read_csv(path)

    df["reference_norm"] = df["source_text"].map(normalize)
    df["word_count"] = df["reference_norm"].str.split().str.len()
    df["duration_sec"] = np.nan

    # Existing benchmark files contain WER but not duration.
    # Start with reference word-count analysis.
    bins = [-1, 3, 7, 15, 30, float("inf")]
    labels = ["1-3", "4-7", "8-15", "16-30", "31+"]

    df["length_bin"] = pd.cut(
        df["word_count"],
        bins=bins,
        labels=labels
    )

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Samples: {len(df)}")
    print(f"Overall mean per-sample WER: {df['wer'].mean() * 100:.2f}%")
    print()

    result = (
        df.groupby("length_bin", observed=False)
        .agg(
            samples=("wer", "size"),
            mean_words=("word_count", "mean"),
            mean_WER=("wer", "mean"),
            median_WER=("wer", "median"),
        )
        .reset_index()
    )

    result["mean_WER"] *= 100

    print(result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    return df, result

mslt_df, mslt_result = analyze(MSLT, "WHISPER — MSLT")
covost_df, covost_result = analyze(COVOST, "WHISPER — CoVoST2")

print("=" * 80)
print("COMPARISON")
print("=" * 80)

comparison = mslt_result.merge(
    covost_result,
    on="length_bin",
    suffixes=("_MSLT", "_CoVoST2")
)

comparison["WER_gap_MSLT_minus_CoVoST2"] = (
    comparison["mean_WER_MSLT"]
    - comparison["mean_WER_CoVoST2"]
)

print(
    comparison[
        [
            "length_bin",
            "samples_MSLT",
            "samples_CoVoST2",
            "mean_WER_MSLT",
            "mean_WER_CoVoST2",
            "WER_gap_MSLT_minus_CoVoST2",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

OUT = Path("experiments/results/whisper_mslt_vs_covost_length_analysis.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUT, index=False)

print()
print(f"Saved: {OUT}")
