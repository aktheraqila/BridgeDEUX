import pandas as pd
import re
from jiwer import process_words

M = r"datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
R = r"results/whisper.cpp (base)_mslt_test/whisper.cpp (base)_mslt_test_results.csv"

m = pd.read_parquet(M)
r = pd.read_csv(R)

m["id"] = m["id"].astype(str).str.zfill(4)
r["id"] = r["sample_id"].astype(str).str.zfill(4)

d = m.merge(r[["id", "hypothesis"]], on="id")

def clean(x):
    x = re.sub(r"<[^>]*>", " ", str(x))
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

d["reference"] = d["t2_reference"].map(clean)
d["hypothesis"] = d["hypothesis"].map(clean)
d["words"] = d["reference"].str.split().str.len()

d["length"] = pd.cut(
    d["words"],
    [-1, 3, 7, 15, 30, float("inf")],
    labels=["1-3", "4-7", "8-15", "16-30", "31+"]
)

patterns = {
    "SPN": r"<SPN\s*/>",
    "UNIN": r"<UNIN\s*/>",
    "NON": r"<NON\s*/>",
    "EU": r"<EU\s*/>",
    "LM": r"<LM>",
    "Laughter": r"\[laughter\]",
}

print("=" * 100)
print("MSLT — ANNOTATION TYPE × UTTERANCE LENGTH")
print("=" * 100)

for length in d["length"].cat.categories:

    print()
    print(f"### {length} words")
    print("-" * 100)

    group = d[d["length"] == length]

    for name, pattern in patterns.items():

        present = group[
            group["t1_reference"].astype(str).str.contains(
                pattern, regex=True, na=False
            )
        ]

        absent = group[
            ~group["t1_reference"].astype(str).str.contains(
                pattern, regex=True, na=False
            )
        ]

        if len(present) == 0:
            continue

        def rate(g):
            I = 0
            N = 0

            for ref, hyp in zip(g["reference"], g["hypothesis"]):
                x = process_words(ref, hyp)
                I += x.insertions
                N += len(ref.split())

            return I / N * 100 if N else 0

        p = rate(present)
        a = rate(absent)

        print(
            f"{name:10s} | "
            f"Present N={len(present):4d}, I={p:7.2f} | "
            f"Absent N={len(absent):4d}, I={a:7.2f} | "
            f"Gap={p-a:+7.2f}"
        )
