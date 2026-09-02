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

m = pd.read_parquet(MANIFEST)
r = pd.read_csv(RESULTS)

m["id"] = m["id"].astype(str).str.zfill(4)
r["id"] = r["sample_id"].astype(str).str.zfill(4)

d = m.merge(r[["id", "hypothesis"]], on="id")

d["reference"] = d["t2_reference"].map(normalize)
d["hypothesis"] = d["hypothesis"].map(normalize)

tags = {
    "No annotation": None,
    "<SPN/>": r"<SPN\s*/>",
    "<UNIN/>": r"<UNIN\s*/>",
    "<NON/>": r"<NON\s*/>",
    "<EU/>": r"<EU\s*/>",
    "<SU/>": r"<SU\s*/>",
    "<LM>": r"<LM>",
    "<NPS>": r"<NPS>",
    "<AS/>": r"<AS\s*/>",
    "<MP>": r"<MP>",
    "<UNSURE>": r"<UNSURE>",
}

print("=" * 75)
print("MSLT — WHISPER ERROR BY ANNOTATION TYPE")
print("=" * 75)

for name, pattern in tags.items():

    if pattern is None:
        g = d[
            ~d["t1_reference"].astype(str).str.contains(
                r"<[^>]+>",
                regex=True,
                na=False
            )
        ]
    else:
        g = d[
            d["t1_reference"].astype(str).str.contains(
                pattern,
                regex=True,
                na=False
            )
        ]

    if len(g) == 0:
        continue

    w = wer(
        g["reference"].tolist(),
        g["hypothesis"].tolist()
    ) * 100

    c = cer(
        g["reference"].tolist(),
        g["hypothesis"].tolist()
    ) * 100

    print(
        f"{name:15s} | "
        f"N={len(g):4d} | "
        f"WER={w:7.2f}% | "
        f"CER={c:7.2f}%"
    )
