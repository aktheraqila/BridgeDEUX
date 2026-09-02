from pathlib import Path

import pandas as pd
from jiwer import wer

from benchmarks.run_asr_benchmark import normalize_mslt_t1


INPUT = Path(
    "datasets/cache/mslt/de_en/dev/parakeet_dev_analyzed.parquet"
)


def normalize(text):
    return normalize_mslt_t1(str(text))


def looks_english(text: str) -> bool:
    """
    Conservative detector for obvious English outputs.
    This is intentionally not used to determine WER.
    """
    text = text.lower().strip()

    if not text:
        return False

    english_phrases = {
        "hello",
        "i see",
        "yeah",
        "yes",
        "no",
        "okay",
        "ok",
        "well",
        "and",
        "so",
        "thank you",
        "thank you very much",
    }

    words = set(text.split())

    if text in english_phrases:
        return True

    return bool(words & english_phrases)


def main():
    print("Loading Parakeet Dev analysis...")
    df = pd.read_parquet(INPUT)

    print(f"Total Dev samples: {len(df)}")

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    df["ref_norm"] = df["clean_transcript"].map(normalize)
    df["hyp_norm"] = df["parakeet_transcript"].map(normalize)

    # ---------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------

    df["hyp_empty"] = df["hyp_norm"].str.strip().eq("")

    df["hyp_words"] = df["hyp_norm"].str.split().str.len()

    df["english_flag"] = df["hyp_norm"].map(looks_english)

    # ---------------------------------------------------------
    # Unfiltered
    # ---------------------------------------------------------

    unfiltered_wer = wer(
        df["ref_norm"].tolist(),
        df["hyp_norm"].tolist(),
    )

    # ---------------------------------------------------------
    # Filter
    #
    # >= 4 hypothesis words
    # non-empty
    # not obvious English
    # ---------------------------------------------------------

    filtered = df[
        (~df["hyp_empty"])
        & (df["hyp_words"] >= 4)
        & (~df["english_flag"])
    ].copy()

    filtered_wer = wer(
        filtered["ref_norm"].tolist(),
        filtered["hyp_norm"].tolist(),
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("PARAKEET FILTER GO/NO-GO")
    print("=" * 60)

    print(f"Total samples       : {len(df)}")
    print(f"Unfiltered WER      : {unfiltered_wer:.4f}")

    print()
    print(f"Filtered samples    : {len(filtered)}")
    print(
        f"Filtered percentage : "
        f"{100 * len(filtered) / len(df):.2f}%"
    )

    print(f"Filtered WER        : {filtered_wer:.4f}")

    print()
    print(f"Empty outputs       : {df['hyp_empty'].sum()}")
    print(f"Short outputs (<4)  : {(df['hyp_words'] < 4).sum()}")
    print(f"English flagged     : {df['english_flag'].sum()}")

    improvement = unfiltered_wer - filtered_wer

    print()
    print(f"WER improvement     : {improvement:+.4f}")

    print()
    print("=" * 60)

    if filtered_wer < unfiltered_wer:
        print("GO")
        print("Filtering improves Parakeet pseudo-label quality.")
    else:
        print("NO-GO")
        print("Filtering does not improve pseudo-label quality.")

    print("=" * 60)

    # Save filtered IDs for W2.
    output = Path(
        "datasets/manifests/mslt_dev_parakeet_filtered_ids.txt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    filtered["id"].astype(str).to_csv(
        output,
        index=False,
        header=False,
    )

    print()
    print(f"Filtered manifest: {output}")


if __name__ == "__main__":
    main()