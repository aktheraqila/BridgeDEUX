import pandas as pd
from jiwer import wer, cer

files = {
    "MSLT": r"results/whisper.cpp (base)_mslt_test/whisper.cpp (base)_mslt_test_results.csv",
    "CoVoST2": r"results/whisper.cpp (base)_test/whisper.cpp (base)_test_results.csv",
}

for name, path in files.items():
    d = pd.read_csv(path)

    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Rows:", len(d))
    print("Stored mean WER:", d["wer"].mean() * 100)
    print("Stored mean CER:", d["cer"].mean() * 100)

    print()
    print("Recalculated corpus WER:",
          wer(d["source_text"].fillna("").tolist(),
              d["hypothesis"].fillna("").tolist()) * 100)

    print("Recalculated corpus CER:",
          cer(d["source_text"].fillna("").tolist(),
              d["hypothesis"].fillna("").tolist()) * 100)

    print()
