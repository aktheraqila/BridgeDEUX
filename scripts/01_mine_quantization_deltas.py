#!/usr/bin/env python3
"""
BridgeDEUX: Step 1 - Quantization Delta Mining (Continuous Distribution)
========================================================================
Parses FP32 and INT8 .jsonl.bak logs, computes continuous sentence-level 
chrF++ and latency deltas, and outputs timestamped statistical artifacts.
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import sacrebleu
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DeltaMiner")


def find_latest_file(pattern: str) -> Path:
    """Finds the most recent file matching a glob pattern."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])


def load_jsonl_bak_records(file_path: Path) -> dict:
    """Loads records from the .bak files using the explicit BridgeDEUX schema."""
    records = {}
    logger.info(f"Loading records from: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sample_id = data["sample_id"]
                
                records[sample_id] = {
                    "sample_id": sample_id,
                    "source": data.get("source_text", ""),
                    "reference": data.get("reference_translation", ""),
                    "translation": data.get("translation", ""),
                    "total_time_ms": float(data.get("total_time_ms", 0.0)),
                    "input_tokens": int(data.get("input_tokens", 0)),
                    "output_tokens": int(data.get("output_tokens", 0))
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed line: {e}")
                continue

    return records


def compute_sentence_chrf(hypothesis: str, reference: str) -> float:
    """Computes sentence-level chrF++ score (0.0 to 100.0)."""
    if not reference or not hypothesis:
        return 0.0
    return float(sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    analysis_dir = repo_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Locate files
    fp32_pattern = str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "*.jsonl.bak")
    int8_pattern = str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "*.jsonl.bak")

    try:
        fp32_file = find_latest_file(fp32_pattern)
        int8_file = find_latest_file(int8_pattern)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Load data
    fp32_data = load_jsonl_bak_records(fp32_file)
    int8_data = load_jsonl_bak_records(int8_file)

    common_ids = set(fp32_data.keys()).intersection(set(int8_data.keys()))
    logger.info(f"Aligned {len(common_ids):,} common samples.")

    # Compute continuous metrics
    rows = []
    for sid in common_ids:
        f_rec = fp32_data[sid]
        i_rec = int8_data[sid]

        ref = f_rec["reference"]
        src = f_rec["source"]
        
        f_chrf = compute_sentence_chrf(f_rec["translation"], ref)
        i_chrf = compute_sentence_chrf(i_rec["translation"], ref)
        
        # Delta calculations
        delta_chrf = f_chrf - i_chrf # Positive means FP32 is better
        delta_latency = f_rec["total_time_ms"] - i_rec["total_time_ms"] # Positive means FP32 is slower

        rows.append({
            "sample_id": sid,
            "source_char_len": len(src),
            "source_word_count": len(src.split()),
            "fp32_chrf": f_chrf,
            "int8_chrf": i_chrf,
            "delta_chrf": delta_chrf,
            "fp32_latency_ms": f_rec["total_time_ms"],
            "int8_latency_ms": i_rec["total_time_ms"],
            "delta_latency_ms": delta_latency
        })

    df = pd.DataFrame(rows)

    # Generate Statistical Summary
    stats = df[["delta_chrf", "delta_latency_ms"]].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99])
    
    # Categorize strictly for reporting visibility (NOT for training)
    exact_match = (df["delta_chrf"] == 0.0).sum()
    fp32_better = (df["delta_chrf"] > 0.0).sum()
    int8_better = (df["delta_chrf"] < 0.0).sum()
    
    summary_table = [
        ["Total Samples", f"{len(df):,}"],
        ["Exact Match (Delta = 0)", f"{exact_match:,} ({(exact_match/len(df))*100:.1f}%)"],
        ["FP32 Better (Delta > 0)", f"{fp32_better:,} ({(fp32_better/len(df))*100:.1f}%)"],
        ["INT8 Better (Delta < 0)", f"{int8_better:,} ({(int8_better/len(df))*100:.1f}%)"],
        ["---", "---"],
        ["Mean Δ chrF++", f"{df['delta_chrf'].mean():.3f}"],
        ["Median Δ chrF++", f"{df['delta_chrf'].median():.3f}"],
        ["Std Dev Δ chrF++", f"{df['delta_chrf'].std():.3f}"],
        ["95th Percentile Δ chrF++", f"{df['delta_chrf'].quantile(0.95):.3f}"],
        ["99th Percentile Δ chrF++", f"{df['delta_chrf'].quantile(0.99):.3f}"],
        ["---", "---"],
        ["Mean FP32 Latency (ms)", f"{df['fp32_latency_ms'].mean():.1f}"],
        ["Mean INT8 Latency (ms)", f"{df['int8_latency_ms'].mean():.1f}"],
        ["Mean Δ Latency (ms)", f"{df['delta_latency_ms'].mean():.1f}"]
    ]

    print("\n" + "="*60)
    print(" STEP 1: CONTINUOUS QUANTIZATION DELTA REPORT")
    print("="*60)
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    print("\nDetailed Percentiles:")
    print(stats.round(3))
    print("="*60 + "\n")

    # Export immutable artifacts
    output_parquet = analysis_dir / f"quantization_deltas_{run_timestamp}.parquet"
    output_csv = analysis_dir / f"quantization_deltas_{run_timestamp}.csv"
    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved dataset artifacts to {analysis_dir.name}/")

    # Plot Distributions (Matplotlib only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: chrF++ Delta Histogram
    axes[0].hist(df["delta_chrf"], bins=100, color="#2b5c8f", edgecolor="black", alpha=0.7)
    axes[0].axvline(0.0, color="black", linestyle="--", label="Zero Delta")
    axes[0].set_title("Distribution of Δ chrF++ (FP32 - INT8)")
    axes[0].set_xlabel("Δ chrF++ (Positive = FP32 is better)")
    axes[0].set_ylabel("Sentence Count")
    axes[0].legend()

    # Plot 2: Latency Delta Histogram
    axes[1].hist(df["delta_latency_ms"], bins=100, color="#d95f02", edgecolor="black", alpha=0.7)
    axes[1].axvline(0.0, color="black", linestyle="--", label="Zero Delta")
    axes[1].set_title("Distribution of Latency Difference")
    axes[1].set_xlabel("Δ Latency ms (Positive = FP32 is slower)")
    axes[1].set_ylabel("Sentence Count")
    axes[1].legend()

    plt.tight_layout()
    plot_path = analysis_dir / f"quantization_distribution_{run_timestamp}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved distribution plot to {plot_path.name}")

if __name__ == "__main__":
    main()