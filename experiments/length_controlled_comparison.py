import pandas as pd
import re
import unicodedata
from jiwer import wer, cer

FILES = {
    "MSLT": r"results/whisper.cpp (base)_mslt_test/whisper.cpp (base)_mslt_test_results.csv",
    "CoVoST2": r"results/whisper.cpp (base)_test/whisper.cpp (base)_test_results.csv",
}

def normalize(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

bins = [-1, 3, 7, 15, 30, float("inf")]
labels = ["1-3", "4-7", "8-15", "16-30", "31+"]

for name, path in FILES.items():
    df = pd.read_csv(path)

    df["reference"] = df["source_text"].map(normalize)
    df["hypothesis"] = df["hypothesis"].map(normalize)
    df["words"] = df["reference"].str.split().str.len()
    df["length"] = pd.cut(
        df["words"],
        bins=bins,
        labels=labels
    )

    print("=" * 70)
    print(name)
    print("=" * 70)

    for label in labels:
        group = df[df["length"] == label]

        if len(group) == 0:
            continue

        group_wer = wer(
            group["reference"].tolist(),
            group["hypothesis"].tolist()
        ) * 100

        group_cer = cer(
            group["reference"].tolist(),
            group["hypothesis"].tolist()
        ) * 100

        print(
            f"{label:>5} words | "
            f"N={len(group):5d} | "
            f"WER={group_wer:6.2f}% | "
            f"CER={group_cer:6.2f}%"
        )

    print()
