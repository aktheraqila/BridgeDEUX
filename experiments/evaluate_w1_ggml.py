import subprocess
import pandas as pd
from pathlib import Path
import re
import time

PARQUET = Path("datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet")
MODEL = Path("models/w1-kd-hf/ggml-model.bin")
CLI = Path("whisper.cpp/build/bin/whisper-cli.exe")
OUTPUT = Path("experiments/results/w1_ggml_test")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET)

print("=" * 70)
print("W1 GGML — MSLT TEST EVALUATION")
print("=" * 70)
print(f"Samples : {len(df)}")
print(f"Model   : {MODEL}")
print(f"CLI     : {CLI}")
print()

if not MODEL.exists():
    raise FileNotFoundError(MODEL)

if not CLI.exists():
    raise FileNotFoundError(CLI)

results_file = OUTPUT / "predictions.csv"

if results_file.exists():
    results = pd.read_csv(results_file, dtype={"id": str})
    completed = set(results["id"].astype(str))
    print(f"Resuming: {len(completed)} samples already completed")
else:
    results = pd.DataFrame(
        columns=["id", "reference", "prediction", "returncode", "time_sec"]
    )
    completed = set()

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

start_all = time.time()

for i, row in df.iterrows():
    sample_id = str(row["id"])

    if sample_id in completed:
        continue

    audio = Path(row["audio_path"])
    reference = str(row["t1_reference"])

    start = time.time()

    proc = subprocess.run(
        [
            str(CLI),
            "-m", str(MODEL),
            "-f", str(audio),
            "-l", "de",
            "-nt",
            "-np",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    elapsed = time.time() - start

    prediction = proc.stdout.strip()

    new_row = pd.DataFrame([{
        "id": sample_id,
        "reference": reference,
        "prediction": prediction,
        "returncode": proc.returncode,
        "time_sec": elapsed,
    }])

    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(results_file, index=False)

    done = len(results)

    if done % 25 == 0 or done == len(df):
        rate = done / (time.time() - start_all)
        remaining = len(df) - done
        eta = remaining / rate if rate > 0 else 0

        print(
            f"[{done}/{len(df)}] "
            f"rate={rate:.2f}/s "
            f"ETA={eta/60:.1f} min"
        )

print()
print("=" * 70)
print("GGML INFERENCE COMPLETE")
print("=" * 70)
print(f"Predictions: {results_file}")
print(f"Total time : {(time.time() - start_all)/60:.1f} min")
print("=" * 70)