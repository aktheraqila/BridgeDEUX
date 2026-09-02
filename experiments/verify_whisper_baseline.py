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

for name, path in FILES.items():
    df = pd.read_csv(path)

    refs = df["source_text"].map(normalize).tolist()
    hyps = df["hypothesis"].map(normalize).tolist()

    print("=" * 70)
    print(name)
    print("=" * 70)
    print("Samples:", len(df))
    print(f"WER: {wer(refs, hyps) * 100:.2f}%")
    print(f"CER: {cer(refs, hyps) * 100:.2f}%")
    print()

print("=" * 70)
print("REFERENCE")
print("=" * 70)
print("Previous stored mean-per-sample WER:")
print("MSLT    46.74%")
print("CoVoST2 31.71%")
