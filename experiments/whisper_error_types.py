import pandas as pd
import re
import unicodedata
from jiwer import process_words

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

    references = df["source_text"].map(normalize).tolist()
    hypotheses = df["hypothesis"].map(normalize).tolist()

    substitutions = 0
    deletions = 0
    insertions = 0
    reference_words = 0

    for ref, hyp in zip(references, hypotheses):
        result = process_words(ref, hyp)

        substitutions += result.substitutions
        deletions += result.deletions
        insertions += result.insertions
        reference_words += len(ref.split())

    print("=" * 70)
    print(name)
    print("=" * 70)
    print("Samples:", len(df))
    print("Reference words:", reference_words)
    print("Substitutions:", substitutions)
    print("Deletions:", deletions)
    print("Insertions:", insertions)

    print()
    print("Errors per 100 reference words:")
    print(f"Substitutions: {substitutions / reference_words * 100:.2f}")
    print(f"Deletions:     {deletions / reference_words * 100:.2f}")
    print(f"Insertions:    {insertions / reference_words * 100:.2f}")
