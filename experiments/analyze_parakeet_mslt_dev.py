import json
from pathlib import Path

import pandas as pd
from jiwer import wer, cer
from benchmarks.run_asr_benchmark import normalize_mslt_t1


DEV_CACHE = Path(
    "datasets/cache/mslt/de_en/dev/mslt_de_en_dev.parquet"
)

PREDICTIONS = Path(
    "datasets/cache/mslt/de_en/dev/parakeet_dev_predictions.jsonl"
)

TRAIN_IDS = Path(
    "datasets/manifests/mslt_dev_train_ids.txt"
)

VAL_IDS = Path(
    "datasets/manifests/mslt_dev_val_ids.txt"
)

OUTPUT = Path(
    "datasets/cache/mslt/de_en/dev/parakeet_dev_analyzed.parquet"
)


def load_predictions():
    rows = []

    with PREDICTIONS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def load_ids(path):
    return {
        x.strip()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }


def main():
    print("Loading MSLT Dev cache...")
    dev = pd.read_parquet(DEV_CACHE)

    print("Loading Parakeet predictions...")
    predictions = load_predictions()

    print("Dev rows:", len(dev))
    print("Prediction rows:", len(predictions))

    # Normalize ID types.
    dev["id"] = dev["id"].astype(str).str.zfill(4)
    predictions["sample_id"] = (
        predictions["sample_id"]
        .astype(str)
        .str.zfill(4)
    )

    # Check duplicate IDs.
    if predictions["sample_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["sample_id"].duplicated(),
            "sample_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate prediction IDs found: {duplicates[:10]}"
        )

    # Check that every Dev sample has a prediction.
    dev_ids = set(dev["id"])
    prediction_ids = set(predictions["sample_id"])

    missing = dev_ids - prediction_ids
    extra = prediction_ids - dev_ids

    if missing:
        raise RuntimeError(
            f"Missing predictions: {len(missing)} "
            f"{sorted(missing)[:10]}"
        )

    if extra:
        raise RuntimeError(
            f"Unexpected prediction IDs: {len(extra)} "
            f"{sorted(extra)[:10]}"
        )

    # Only retain the fields needed from predictions.
    predictions = predictions[
        [
            "sample_id",
            "parakeet_transcript",
            "error",
        ]
    ]

    # Join by sample ID.
    df = dev.merge(
        predictions,
        left_on="id",
        right_on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    # Use the exact normalization protocol from the MSLT Test benchmark.
    references = (
        df["clean_transcript"]
        .fillna("")
        .astype(str)
        .map(normalize_mslt_t1)
    )

    hypotheses = (
        df["parakeet_transcript"]
        .fillna("")
        .astype(str)
        .map(normalize_mslt_t1)
    )

    df["asr_reference"] = references
    df["parakeet_transcript"] = hypotheses


    def calculate_wer(ref: str, hyp: str) -> float:
        if not ref and not hyp:
            return 0.0
        if not ref and hyp:
            return 1.0
        if ref and not hyp:
            return 1.0

        return wer(ref, hyp)


    def calculate_cer(ref: str, hyp: str) -> float:
        if not ref and not hyp:
            return 0.0
        if not ref and hyp:
            return 1.0
        if ref and not hyp:
            return 1.0

        return cer(ref, hyp)


    df["parakeet_wer"] = [
        calculate_wer(ref, hyp)
        for ref, hyp in zip(references, hypotheses)
    ]

    df["parakeet_cer"] = [
        calculate_cer(ref, hyp)
        for ref, hyp in zip(references, hypotheses)
    ]

    # Fixed train/validation assignment.
    train_ids = load_ids(TRAIN_IDS)
    val_ids = load_ids(VAL_IDS)

    if train_ids & val_ids:
        raise RuntimeError(
            "Train/validation ID overlap detected."
        )

    if train_ids | val_ids != dev_ids:
        missing_split = dev_ids - (train_ids | val_ids)
        extra_split = (train_ids | val_ids) - dev_ids

        raise RuntimeError(
            f"Split mismatch. "
            f"Missing={len(missing_split)}, "
            f"Extra={len(extra_split)}"
        )

    df["subset"] = df["id"].map(
        lambda x: (
            "train"
            if x in train_ids
            else "validation"
        )
    )

    # Basic teacher-quality indicators.
    df["parakeet_correct"] = df["parakeet_wer"] == 0.0
    df["parakeet_empty"] = hypotheses.str.len() == 0

    # Save.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    # Summary.
    print()
    print("========== PARAKEET DEV ANALYSIS ==========")
    print("Total samples :", len(df))
    print("Train         :", (df["subset"] == "train").sum())
    print("Validation    :", (df["subset"] == "validation").sum())
    print()
    print(
        "Overall WER   :",
        f"{df['parakeet_wer'].mean():.4f}",
    )
    print(
        "Overall CER   :",
        f"{df['parakeet_cer'].mean():.4f}",
    )
    print(
        "WER = 0       :",
        int(df["parakeet_correct"].sum()),
    )
    print(
        "Empty outputs :",
        int(df["parakeet_empty"].sum()),
    )
    print()
    print("By subset:")
    print(
        df.groupby("subset")[
            ["parakeet_wer", "parakeet_cer"]
        ].mean()
    )
    print()
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()