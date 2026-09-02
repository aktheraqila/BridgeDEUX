import pandas as pd
import re
import math
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
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
d["ref_words"] = d["reference"].str.split().str.len()

t1 = d["t1_reference"].astype(str)

d["SPN"] = t1.str.count(r"<SPN\s*/>")
d["UNIN"] = t1.str.count(r"<UNIN\s*/>")
d["NON"] = t1.str.count(r"<NON\s*/>")
d["EU"] = t1.str.count(r"<EU\s*/>")
d["LM"] = t1.str.count(r"<LM>")
d["Laughter"] = t1.str.count(r"\[laughter\]")

def repeated_words(x):
    words = re.findall(r"\b[\wÄÖÜäöüß]+\b", x.lower())
    return sum(a == b for a, b in zip(words, words[1:]))

d["Repeated"] = t1.map(repeated_words)

d["insertions"] = [
    process_words(ref, hyp).insertions
    for ref, hyp in zip(d["reference"], d["hypothesis"])
]

d = d[d["ref_words"] > 0].copy()

predictors = [
    "SPN",
    "UNIN",
    "NON",
    "EU",
    "LM",
    "Laughter",
    "Repeated",
]

X = sm.add_constant(d[predictors].astype(float))

# Offset controls for the number of reference words.
offset = d["ref_words"].map(math.log)

model = NegativeBinomial(
    d["insertions"],
    X,
    offset=offset
)

result = model.fit(
    disp=False,
    maxiter=200
)

print("=" * 90)
print("MSLT — ESTIMATED-DISPERSION NEGATIVE BINOMIAL")
print("=" * 90)

print(result.summary())

print()
print("=" * 90)
print("ESTIMATED DISPERSION")
print("=" * 90)
print(f"Alpha = {result.params['alpha']:.6f}")

print()
print("=" * 90)
print("INCIDENCE RATE RATIOS")
print("=" * 90)

for variable in predictors:
    coef = result.params[variable]
    p = result.pvalues[variable]
    irr = math.exp(coef)

    print(
        f"{variable:12s} | "
        f"IRR={irr:8.3f} | "
        f"p={p:.6f}"
    )
