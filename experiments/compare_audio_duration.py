import pandas as pd
import soundfile as sf
from pathlib import Path

MSLT = r"datasets/cache/mslt/de_en/test/mslt_de_asr_test.parquet"
COVOST = r"datasets/cache/covost2_de_en_test.parquet"

def duration(path):
    try:
        return sf.info(path).duration
    except Exception:
        return None

# MSLT
mslt = pd.read_parquet(MSLT)
mslt["duration_sec"] = mslt["audio_path"].map(duration)

# CoVoST2
covost = pd.read_parquet(COVOST)

# CoVoST2 paths in the manifest are stale Linux paths.
# Use the filename to locate the actual local audio.
def find_covost_audio(filename):
    matches = list(Path("datasets").rglob(filename))
    return matches[0] if matches else None

covost["local_audio"] = covost["file_name"].map(
    lambda x: find_covost_audio(Path(str(x)).name)
)
covost["duration_sec"] = covost["local_audio"].map(duration)

print("=" * 70)
print("AUDIO DURATION")
print("=" * 70)

for name, df in [("MSLT", mslt), ("CoVoST2", covost)]:
    d = df["duration_sec"].dropna()

    print(f"\n{name}")
    print("-" * 70)
    print("Samples:", len(df))
    print("Duration found:", len(d))
    print("Duration missing:", df["duration_sec"].isna().sum())

    if len(d):
        print(f"Mean:   {d.mean():.2f} sec")
        print(f"Median: {d.median():.2f} sec")
        print(f"Min:    {d.min():.2f} sec")
        print(f"Max:    {d.max():.2f} sec")

print()
print("MSLT duration quantiles:")
print(mslt["duration_sec"].describe(
    percentiles=[.10, .25, .50, .75, .90]
))

print()
print("CoVoST2 duration quantiles:")
print(covost["duration_sec"].describe(
    percentiles=[.10, .25, .50, .75, .90]
))
