import pandas as pd
import re
import statsmodels.api as sm

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

d["Repeated"] = t1.map(
    lambda x: sum(
        a == b
        for a, b in zip(
            re.findall(r"\b[\wÄÖÜäöüß]+\b", x.lower()),
            re.findall(r"\b[\wÄÖÜäöüß]+\b", x.lower())[1:]
        )
    )
)

from jiwer import process_words

def get_insertions(hyp, ref):
    return process_words(ref, hyp).insertions

d["insertions"] = [
    get_insertions(h, ref)
    for h, ref in zip(d["hypothesis"], d["reference"])
]

# Remove zero-reference-word rows because they cannot have a meaningful
# word-normalized exposure.
d = d[d["ref_words"] > 0].copy()

predictors = [
    "ref_words",
    "SPN",
    "UNIN",
    "NON",
    "EU",
    "LM",
    "Laughter",
    "Repeated",
]

X = d[predictors].astype(float)
X = sm.add_constant(X)

y = d["insertions"]

model = sm.GLM(
    y,
    X,
    family=sm.families.Poisson(),
    offset=d["ref_words"].map(lambda x: __import__("math").log(x))
)

result = model.fit()

print("=" * 90)
print("MSLT — POISSON REGRESSION: INSERTIONS vs CONVERSATIONAL PHENOMENA")
print("=" * 90)

print(result.summary())

print()
print("=" * 90)
print("INCIDENCE RATE RATIOS")
print("=" * 90)

for variable in predictors:
    coef = result.params[variable]
    p = result.pvalues[variable]
    irr = __import__("math").exp(coef)

    print(
        f"{variable:12s} | "
        f"IRR={irr:8.3f} | "
        f"p={p:.6f}"
    )
