import pandas as pd
import re
import unicodedata

P = r"datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"

def clean(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

d = pd.read_parquet(P)

d["t1_clean"] = d["t1_reference"].map(clean)
d["t2_clean"] = d["t2_reference"].map(clean)

d["t1_words"] = d["t1_clean"].str.split().str.len()
d["t2_words"] = d["t2_clean"].str.split().str.len()

d["removed_words"] = d["t1_words"] - d["t2_words"]

d["annotated"] = d["t1_reference"].astype(str).str.contains(
    r"<[^>]+>", regex=True, na=False
)

print("=" * 75)
print("MSLT — T1 TO T2 REFERENCE TRANSFORMATION")
print("=" * 75)

print(f"Samples: {len(d)}")
print(f"Annotated samples: {d['annotated'].sum()}")
print(f"Unannotated samples: {(~d['annotated']).sum()}")

print()
print("WORD COUNTS")
print("-" * 75)
print(f"T1 mean words: {d['t1_words'].mean():.2f}")
print(f"T2 mean words: {d['t2_words'].mean():.2f}")
print(f"T1 median words: {d['t1_words'].median():.2f}")
print(f"T2 median words: {d['t2_words'].median():.2f}")

print()
print("T1 → T2 WORD DIFFERENCE")
print("-" * 75)
print(f"Mean words removed: {d['removed_words'].mean():.2f}")
print(f"Median words removed: {d['removed_words'].median():.2f}")
print(f"Samples with words removed: {(d['removed_words'] > 0).sum()}")
print(f"Samples with no change: {(d['removed_words'] == 0).sum()}")
print(f"Samples where T2 is longer: {(d['removed_words'] < 0).sum()}")

print()
print("ANNOTATED vs UNANNOTATED")
print("-" * 75)

for status, label in [(False, "Unannotated"), (True, "Annotated")]:
    g = d[d["annotated"] == status]

    print(
        f"{label:12s} | "
        f"N={len(g):4d} | "
        f"T1={g['t1_words'].mean():6.2f} words | "
        f"T2={g['t2_words'].mean():6.2f} words | "
        f"Removed={g['removed_words'].mean():6.2f}"
    )

print()
print("LARGEST T1 → T2 REDUCTIONS")
print("-" * 75)

cols = [
    "id",
    "t1_reference",
    "t2_reference",
    "t1_words",
    "t2_words",
    "removed_words",
]

print(
    d.sort_values("removed_words", ascending=False)[cols]
    .head(20)
    .to_string(index=False)
)
