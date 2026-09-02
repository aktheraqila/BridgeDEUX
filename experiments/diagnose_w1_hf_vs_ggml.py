import re
import unicodedata
from pathlib import Path

import pandas as pd
from jiwer import wer, cer


HF_FILE = Path(
    "experiments/results/w1_hf_controlled/predictions.csv"
)

GGML_FILE = Path(
    "experiments/results/w1_ggml_test/predictions.csv"
)

OUTPUT_DIR = Path(
    "experiments/results/w1_hf_vs_ggml"
)

DIAGNOSTIC_FILE = OUTPUT_DIR / "diagnostic.csv"


def normalize_text(text):
    text = re.sub(
        r"<[^>]*>",
        " ",
        str(text),
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.category(c).startswith("P")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def sample_wer(reference, hypothesis):
    return wer([reference], [hypothesis])


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hf = pd.read_csv(
        HF_FILE,
        dtype={"id": str},
    )

    ggml = pd.read_csv(
        GGML_FILE,
        dtype={"id": str},
    )

    hf["id"] = hf["id"].str.zfill(4)
    ggml["id"] = ggml["id"].str.zfill(4)

    hf = hf.rename(
        columns={
            "prediction": "hf_prediction",
        }
    )

    ggml = ggml.rename(
        columns={
            "prediction": "ggml_prediction",
        }
    )

    merged = hf[
        ["id", "reference", "hf_prediction"]
    ].merge(
        ggml[
            ["id", "ggml_prediction"]
        ],
        on="id",
        how="inner",
    )

    if len(merged) != 2275:
        raise RuntimeError(
            f"Expected 2275 matched samples, got {len(merged)}"
        )

    merged["reference_norm"] = (
        merged["reference"]
        .fillna("")
        .map(normalize_text)
    )

    merged["hf_norm"] = (
        merged["hf_prediction"]
        .fillna("")
        .map(normalize_text)
    )

    merged["ggml_norm"] = (
        merged["ggml_prediction"]
        .fillna("")
        .map(normalize_text)
    )

    merged["hf_wer"] = [
        sample_wer(reference, hypothesis)
        for reference, hypothesis in zip(
            merged["reference_norm"],
            merged["hf_norm"],
        )
    ]

    merged["ggml_wer"] = [
        sample_wer(reference, hypothesis)
        for reference, hypothesis in zip(
            merged["reference_norm"],
            merged["ggml_norm"],
        )
    ]

    merged["wer_difference"] = (
        merged["hf_wer"]
        - merged["ggml_wer"]
    )

    merged["same_prediction"] = (
        merged["hf_norm"]
        == merged["ggml_norm"]
    )

    merged["winner"] = "tie"

    merged.loc[
        merged["hf_wer"] < merged["ggml_wer"],
        "winner",
    ] = "HF"

    merged.loc[
        merged["ggml_wer"] < merged["hf_wer"],
        "winner",
    ] = "GGML"

    references = merged[
        "reference_norm"
    ].tolist()

    hf_predictions = merged[
        "hf_norm"
    ].tolist()

    ggml_predictions = merged[
        "ggml_norm"
    ].tolist()

    hf_corpus_wer = wer(
        references,
        hf_predictions,
    )

    ggml_corpus_wer = wer(
        references,
        ggml_predictions,
    )

    hf_corpus_cer = cer(
        references,
        hf_predictions,
    )

    ggml_corpus_cer = cer(
        references,
        ggml_predictions,
    )

    identical = int(
        merged["same_prediction"].sum()
    )

    different = len(merged) - identical

    hf_better = int(
        (merged["winner"] == "HF").sum()
    )

    ggml_better = int(
        (merged["winner"] == "GGML").sum()
    )

    ties = int(
        (merged["winner"] == "tie").sum()
    )

    print("=" * 70)
    print("W1 HF vs GGML — SAMPLE-BY-SAMPLE DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Matched samples :", len(merged))
    print("Identical       :", identical)
    print("Different       :", different)

    print()
    print("CORPUS METRICS")
    print("-" * 70)

    print(
        f"HF WER          : "
        f"{hf_corpus_wer:.4f} "
        f"({hf_corpus_wer * 100:.2f}%)"
    )

    print(
        f"GGML WER        : "
        f"{ggml_corpus_wer:.4f} "
        f"({ggml_corpus_wer * 100:.2f}%)"
    )

    print(
        f"HF CER          : "
        f"{hf_corpus_cer:.4f} "
        f"({hf_corpus_cer * 100:.2f}%)"
    )

    print(
        f"GGML CER        : "
        f"{ggml_corpus_cer:.4f} "
        f"({ggml_corpus_cer * 100:.2f}%)"
    )

    print()
    print("PER-SAMPLE WINNERS")
    print("-" * 70)

    print("HF better       :", hf_better)
    print("GGML better     :", ggml_better)
    print("Tie             :", ties)

    print()
    print("LARGEST HF ADVANTAGE")
    print("-" * 70)

    hf_advantage = merged.sort_values(
        "wer_difference"
    ).head(10)

    for _, row in hf_advantage.iterrows():
        print()
        print(f"[{row['id']}]")
        print("REF :", row["reference"])
        print("HF  :", row["hf_prediction"])
        print("GGML:", row["ggml_prediction"])
        print(
            f"WER HF={row['hf_wer']:.3f} "
            f"GGML={row['ggml_wer']:.3f}"
        )

    print()
    print("LARGEST GGML ADVANTAGE")
    print("-" * 70)

    ggml_advantage = merged.sort_values(
        "wer_difference",
        ascending=False,
    ).head(10)

    for _, row in ggml_advantage.iterrows():
        print()
        print(f"[{row['id']}]")
        print("REF :", row["reference"])
        print("HF  :", row["hf_prediction"])
        print("GGML:", row["ggml_prediction"])
        print(
            f"WER HF={row['hf_wer']:.3f} "
            f"GGML={row['ggml_wer']:.3f}"
        )

    merged.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("Saved:", DIAGNOSTIC_FILE)


if __name__ == "__main__":
    main()