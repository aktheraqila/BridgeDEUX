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
d["words"] = d["reference"].str.split().str.len()

d["annotated"] = d["t1_reference"].astype(str).str.contains(
    r"<[^>]+>", regex=True, na=False
)

d["length"] = pd.cut(
    d["words"],
    [-1, 3, 7, 15, 30, float("inf")],
    labels=["1-3", "4-7", "8-15", "16-30", "31+"]
)

rows = []

for length in d["length"].cat.categories:
    for annotated in [False, True]:

        g = d[
            (d["length"] == length) &
            (d["annotated"] == annotated)
        ]

        if len(g) == 0:
            continue

        S = D = I = N = 0

        for ref, hyp in zip(g["reference"], g["hypothesis"]):
            x = process_words(ref, clean(hyp))
            S += x.substitutions
            D += x.deletions
            I += x.insertions
            N += len(ref.split())

        rows.append({
            "length": str(length),
            "group": "Annotated" if annotated else "Unannotated",
            "samples": len(g),
            "reference_words": N,
            "S_per100": S / N * 100,
            "D_per100": D / N * 100,
            "I_per100": I / N * 100,
        })

out = pd.DataFrame(rows)

print("=" * 90)
print("MSLT — LENGTH-CONTROLLED ANNOTATION vs INSERTIONS")
print("=" * 90)
print(
    out.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()
print("=" * 90)
print("ANNOTATED MINUS UNANNOTATED INSERTION RATE")
print("=" * 90)

for length in out["length"].unique():
    a = out[(out.length == length) & (out.group == "Annotated")]
    u = out[(out.length == length) & (out.group == "Unannotated")]

    if len(a) and len(u):
        gap = a.iloc[0].I_per100 - u.iloc[0].I_per100
        print(f"{length:>5} words: {gap:+.2f} insertions / 100 words")
