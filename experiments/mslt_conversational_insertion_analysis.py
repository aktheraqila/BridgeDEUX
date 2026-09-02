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

# Count conversational phenomena in the original T1 transcript.
d["spn"] = d.t1_reference.astype(str).str.count(r"<SPN\s*/>")
d["unin"] = d.t1_reference.astype(str).str.count(r"<UNIN\s*/>")
d["non"] = d.t1_reference.astype(str).str.count(r"<NON\s*/>")
d["eu"] = d.t1_reference.astype(str).str.count(r"<EU\s*/>")
d["lm"] = d.t1_reference.astype(str).str.count(r"<LM>")
d["laughter"] = d.t1_reference.astype(str).str.count(r"\[laughter\]")

# Repetition proxy:
# consecutive repeated lexical words in T1.
def repeated_words(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
    words = re.findall(r"\b[\wÄÖÜäöüß]+\b", text.lower())
    return sum(a == b for a, b in zip(words, words[1:]))

d["repeated_words"] = d.t1_reference.map(repeated_words)

def errors(ref, hyp):
    x = process_words(str(ref), str(hyp))
    return x.insertions, x.deletions, x.substitutions

d[["insertions", "deletions", "substitutions"]] = d.apply(
    lambda x: pd.Series(errors(x["t2_reference"], x["hypothesis"])),
    axis=1
)

d["ref_words"] = (
    d["t2_reference"]
    .astype(str)
    .str.split()
    .str.len()
)

d["insertion_rate"] = (
    d["insertions"] / d["ref_words"].clip(lower=1) * 100
)

d["any_annotation"] = (
    d.t1_reference.astype(str)
    .str.contains(r"<[^>]+>", regex=True, na=False)
)

print("=" * 80)
print("MSLT — CONVERSATIONAL PHENOMENA vs WHISPER INSERTIONS")
print("=" * 80)

print()
print("ANNOTATION PRESENCE")
print("-" * 80)

for status in [False, True]:
    g = d[d.any_annotation == status]
    print(
        f"{'Annotated' if status else 'Unannotated':12s} | "
        f"N={len(g):4d} | "
        f"Insertions={g.insertions.sum():5d} | "
        f"Insertion rate={g.insertions.sum()/g.ref_words.sum()*100:.2f}/100 words"
    )

print()
print("INDIVIDUAL PHENOMENA")
print("-" * 80)

phenomena = [
    ("SPN", "spn"),
    ("UNIN", "unin"),
    ("NON", "non"),
    ("EU", "eu"),
    ("LM", "lm"),
    ("Laughter", "laughter"),
    ("Repeated words", "repeated_words"),
]

for name, col in phenomena:

    present = d[d[col] > 0]
    absent = d[d[col] == 0]

    if len(present) == 0:
        continue

    p_rate = present.insertions.sum() / present.ref_words.sum() * 100
    a_rate = absent.insertions.sum() / absent.ref_words.sum() * 100

    print(
        f"{name:16s} | "
        f"Present N={len(present):4d}, I={p_rate:6.2f} | "
        f"Absent N={len(absent):4d}, I={a_rate:6.2f}"
    )

print()
print("NUMBER OF PHENOMENA")
print("-" * 80)

d["phenomena_count"] = (
    (d.spn > 0).astype(int)
    + (d.unin > 0).astype(int)
    + (d.non > 0).astype(int)
    + (d.eu > 0).astype(int)
    + (d.lm > 0).astype(int)
    + (d.laughter > 0).astype(int)
    + (d.repeated_words > 0).astype(int)
)

for n in sorted(d.phenomena_count.unique()):
    g = d[d.phenomena_count == n]

    print(
        f"{n} phenomena | "
        f"N={len(g):4d} | "
        f"Insertion rate={g.insertions.sum()/g.ref_words.sum()*100:.2f}/100 words"
    )
