import pandas as pd
import json
import glob
import random
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")

    files.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(files[0])


def load_jsonl_records(file_path):
    records = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                if "sample_id" in data:
                    records[data["sample_id"]] = data

            except json.JSONDecodeError:
                continue

    return records


# ============================================================
# Paths
# ============================================================

analysis_dir = Path("analysis")
results_dir = Path("results")

# ============================================================
# Load Step 3 parquet
# ============================================================

files = sorted(
    analysis_dir.glob("int8_seq_likelihood_exploratory_*.parquet")
)

if not files:
    raise FileNotFoundError(
        "No Step 3 parquet file found."
    )

df = pd.read_parquet(files[-1])


# ============================================================
# STEP 0b — ORACLE CEILING
# ============================================================

pos = df[df["delta_chrf"] > 0]
neg = df[df["delta_chrf"] < 0]

ceiling = pos["delta_chrf"].sum() / len(df)
negative_side = neg["delta_chrf"].sum() / len(df)

print("=" * 60)
print("STEP 0b — ORACLE CEILING")
print("=" * 60)

print(f"Total samples        : {len(df):,}")
print(f"FP32 wins (Δ > 0)    : {len(pos):,}")
print(f"FP32-win percentage  : {len(pos) / len(df) * 100:.2f}%")
print(f"Mean positive Δ      : {pos['delta_chrf'].mean():.6f}")
print(f"Oracle ceiling       : {ceiling:.6f} chrF++")

print()
print(f"INT8 wins (Δ < 0)    : {len(neg):,}")
print(f"INT8-win percentage  : {len(neg) / len(df) * 100:.2f}%")
print(f"Mean negative Δ      : {neg['delta_chrf'].mean():.6f}")
print(f"Negative contribution: {negative_side:.6f} chrF++")

print()
print(f"Net mean Δ           : {df['delta_chrf'].mean():.6f} chrF++")
print(f"Absolute asymmetry   : {abs(ceiling + negative_side):.6f}")
print("=" * 60)


# ============================================================
# Load original FP32 / INT8 JSONL results
# ============================================================

fp32_pattern = str(
    results_dir /
    "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" /
    "*.jsonl.bak"
)

int8_pattern = str(
    results_dir /
    "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" /
    "*.jsonl.bak"
)

fp32_file = find_latest_file(fp32_pattern)
int8_file = find_latest_file(int8_pattern)

print()
print(f"FP32 results : {fp32_file.name}")
print(f"INT8 results : {int8_file.name}")

fp32_records = load_jsonl_records(fp32_file)
int8_records = load_jsonl_records(int8_file)


# ============================================================
# Identify divergent output-length groups
# ============================================================

# delta_output_tokens =
# FP32 output tokens - INT8 output tokens
#
# > 2  -> INT8 output is shorter
# < -2 -> FP32 output is shorter
#
# IMPORTANT:
# These are only length differences.
# We do NOT call them "truncation" yet.

int8_shorter_ids = set(
    df.loc[df["delta_output_tokens"] > 2, "sample_id"]
)

fp32_shorter_ids = set(
    df.loc[df["delta_output_tokens"] < -2, "sample_id"]
)


# ============================================================
# Build inspection records
# ============================================================

def build_inspection_records(sample_ids):
    rows = []

    for sid in sample_ids:

        if sid not in fp32_records or sid not in int8_records:
            continue

        f = fp32_records[sid]
        i = int8_records[sid]

        analysis_row = df[df["sample_id"] == sid].iloc[0]

        rows.append({
            "sample_id": sid,

            # Original input/reference
            "source_text": f.get("source_text", ""),
            "reference_translation": f.get(
                "reference_translation", ""
            ),

            # Actual model outputs
            "fp32_translation": f.get("translation", ""),
            "int8_translation": i.get("translation", ""),

            # Output lengths
            "fp32_output_tokens": f.get("output_tokens", 0),
            "int8_output_tokens": i.get("output_tokens", 0),
            "delta_output_tokens": analysis_row[
                "delta_output_tokens"
            ],

            # Quality comparison
            "fp32_chrf": analysis_row.get(
                "fp32_chrf", None
            ),
            "int8_chrf": analysis_row.get(
                "int8_chrf", None
            ),
            "delta_chrf": analysis_row.get(
                "delta_chrf", None
            ),

            # Model-derived signal
            "seq_loss_hyp": analysis_row.get(
                "seq_loss_hyp", None
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Deterministic sampling
# ============================================================

int8_shorter_df = build_inspection_records(
    int8_shorter_ids
)

fp32_shorter_df = build_inspection_records(
    fp32_shorter_ids
)

# 50 from each group, reproducibly
int8_sample = int8_shorter_df.sample(
    n=min(50, len(int8_shorter_df)),
    random_state=42
)

fp32_sample = fp32_shorter_df.sample(
    n=min(50, len(fp32_shorter_df)),
    random_state=42
)


# ============================================================
# Save inspection files
# ============================================================

int8_output_file = (
    analysis_dir / "inspect_int8_shorter.csv"
)

fp32_output_file = (
    analysis_dir / "inspect_fp32_shorter.csv"
)

combined_output_file = (
    analysis_dir / "inspect_divergent_outputs_100.csv"
)

int8_sample.to_csv(
    int8_output_file,
    index=False,
    encoding="utf-8-sig"
)

fp32_sample.to_csv(
    fp32_output_file,
    index=False,
    encoding="utf-8-sig"
)

combined = pd.concat(
    [
        int8_sample.assign(
            length_group="INT8 shorter"
        ),
        fp32_sample.assign(
            length_group="FP32 shorter"
        ),
    ],
    ignore_index=True
)

combined.to_csv(
    combined_output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 60)
print("DIVERGENT OUTPUT INSPECTION FILES")
print("=" * 60)

print(
    f"INT8-shorter cases available : "
    f"{len(int8_shorter_df):,}"
)

print(
    f"FP32-shorter cases available : "
    f"{len(fp32_shorter_df):,}"
)

print()
print(
    f"Saved INT8-shorter sample : "
    f"{int8_output_file}"
)

print(
    f"Saved FP32-shorter sample : "
    f"{fp32_output_file}"
)

print(
    f"Saved combined sample     : "
    f"{combined_output_file}"
)

print()
print("Sampling seed: 42")
print("Samples: up to 50 from each group")
print()
print(
    "IMPORTANT: 'shorter' describes output length only. "
    "It does NOT establish truncation or failure."
)

print("=" * 60)