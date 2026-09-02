import pandas as pd
import re
import unicodedata
from jiwer import wer, cer

MANIFEST = r"datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
RESULTS = r"results/whisper.cpp (base)_mslt_test/whisper.cpp (base)_mslt_test_results.csv"

def normalize(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

manifest = pd.read_parquet(MANIFEST)
results = pd.read_csv(RESULTS)

manifest["id"] = manifest["id"].astype(str).str.zfill(4)
results["id"] = results["sample_id"].astype(str).str.zfill(4)

df = manifest.merge(
    results[["id", "hypothesis"]],
    on="id",
    how="inner"
)

df["reference"] = df["t2_reference"].map(normalize)
df["hypothesis"] = df["hypothesis"].map(normalize)

df["words"] = df["reference"].str.split().str.len()

df["annotated"] = df["t1_reference"].astype(str).str.contains(
    r"<[^>]+>",
    regex=True,
    na=False
)

bins = [-1, 3, 7, 15, 30, float("inf")]
labels = ["1-3", "4-7", "8-15", "16-30", "31+"]

df["length"] = pd.cut(
    df["words"],
    bins=bins,
    labels=labels
)

print("=" * 75)
print("MSLT — ANNOTATED vs UNANNOTATED")
print("=" * 75)

for status in [False, True]:
    g = df[df["annotated"] == status]

    print()
    print("ANNOTATED" if status else "UNANNOTATED")
    print("-" * 75)
    print("Samples:", len(g))
    print(
        f"WER: {wer(g.reference.tolist(), g.hypothesis.tolist()) * 100:.2f}%"
    )
    print(
        f"CER: {cer(g.reference.tolist(), g.hypothesis.tolist()) * 100:.2f}%"
    )

print()
print("=" * 75)
print("WITHIN-LENGTH COMPARISON")
print("=" * 75)

for label in labels:
    print()
    print(label, "words")

    for status in [False, True]:
        g = df[
            (df["annotated"] == status)
            & (df["length"] == label)
        ]

        if len(g) == 0:
            continue

        w = wer(
            g.reference.tolist(),
            g.hypothesis.tolist()
        ) * 100

        c = cer(
            g.reference.tolist(),
            g.hypothesis.tolist()
        ) * 100

        print(
            ("Annotated" if status else "Unannotated")
            + f": N={len(g):4d}, WER={w:6.2f}%, CER={c:6.2f}%"
        )
