import pandas as pd
import re
import unicodedata
from jiwer import process_words

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

    print("=" * 80)
    print(name)
    print("=" * 80)

    for label in labels:

        group = df[df["length"] == label]

        if len(group) == 0:
            continue

        substitutions = 0
        deletions = 0
        insertions = 0
        reference_words = 0

        for ref, hyp in zip(
            group["reference"],
            group["hypothesis"]
        ):
            result = process_words(ref, hyp)

            substitutions += result.substitutions
            deletions += result.deletions
            insertions += result.insertions

            reference_words += len(ref.split())

        print(
            f"{label:>5} words | "
            f"N={len(group):5d} | "
            f"RefWords={reference_words:6d} | "
            f"S={substitutions / reference_words * 100:6.2f} | "
            f"D={deletions / reference_words * 100:6.2f} | "
            f"I={insertions / reference_words * 100:6.2f}"
        )

    print()
