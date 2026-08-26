#!/usr/bin/env python3
"""
BridgeDEUX: Step 2A - Exploratory Data Analysis & Feature Correlation
======================================================================
Performs rigorous statistical EDA on the 13,511 CoVoST 2 sentence deltas.
Computes Pearson, Spearman correlations, effect sizes, and binned distributions
without making premature binary assumptions or training classifiers.
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EDAAnalysis")


def find_latest_file(pattern: str) -> Path:
    """Finds the most recent file matching a glob pattern."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])


def compute_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculates Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(), group2.std()
    mean1, mean2 = group1.mean(), group2.mean()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    analysis_dir = repo_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Locate latest parquet from Step 1 or reload jsonl.bak files to get source text features
    try:
        parquet_pattern = str(analysis_dir / "quantization_deltas_*.parquet")
        latest_parquet = find_latest_file(parquet_pattern)
        logger.info(f"Loading pre-computed deltas from: {latest_parquet.name}")
        df = pd.read_parquet(latest_parquet)
    except FileNotFoundError:
        logger.error("No quantization_deltas parquet file found in analysis/. Run Step 1 script first.")
        sys.exit(1)

    # 2. Re-load source text from jsonl.bak files to extract granular linguistic features
    fp32_pattern = str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "*.jsonl.bak")
    try:
        fp32_file = find_latest_file(fp32_pattern)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    logger.info(f"Extracting granular linguistic features from source text: {fp32_file.name}")
    source_texts = {}
    with open(fp32_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                source_texts[data["sample_id"]] = data.get("source_text", "")
            except (json.JSONDecodeError, KeyError):
                continue

    # Map source text back into dataframe
    df["source_text"] = df["sample_id"].map(source_texts)
    df = df.dropna(subset=["source_text"])

    # Extract granular features for EDA
    df["src_char_len"] = df["source_text"].str.len()
    df["src_word_count"] = df["source_text"].str.split().str.len()
    df["src_avg_word_len"] = df["src_char_len"] / df["src_word_count"].replace(0, 1)
    df["src_num_digits"] = df["source_text"].apply(lambda s: sum(c.isdigit() for c in s))
    df["src_has_numbers"] = (df["src_num_digits"] > 0).astype(int)
    df["src_num_punctuation"] = df["source_text"].apply(lambda s: sum(1 for c in s if c in ",.?!:;-\"'"))
    df["src_has_punctuation"] = (df["src_num_punctuation"] > 0).astype(int)
    df["src_num_uppercase"] = df["source_text"].apply(lambda s: sum(1 for w in s.split() if w and w[0].isupper()))

    feature_cols = [
        "src_char_len", 
        "src_word_count", 
        "src_avg_word_len", 
        "src_num_digits", 
        "src_num_punctuation", 
        "src_num_uppercase"
    ]

    # 3. Compute Pearson and Spearman Correlations
    logger.info("Computing Pearson and Spearman correlations against delta_chrf...")
    corr_results = []
    for feat in feature_cols:
        pearson_r, p_val_p = stats.pearsonr(df["delta_chrf"], df[feat])
        spearman_rho, p_val_s = stats.spearmanr(df["delta_chrf"], df[feat])
        corr_results.append([
            feat, 
            round(pearson_r, 4), 
            f"{p_val_p:.2e}", 
            round(spearman_rho, 4), 
            f"{p_val_s:.2e}"
        ])

    print("\n" + "="*80)
    print(" STEP 2A: CORRELATION ANALYSIS (Feature vs. Δ chrF++)")
    print("="*80)
    print("Note: Positive delta means FP32 is better (INT8 lost quality).")
    print(tabulate(
        corr_results, 
        headers=["Feature", "Pearson (r)", "p-value (Pearson)", "Spearman (rho)", "p-value (Spearman)"], 
        tablefmt="fancy_grid"
    ))

    # 4. Group Differences & Effect Sizes (Binary Features)
    print("\n" + "="*80)
    print(" GROUP DIFFERENCES & EFFECT SIZES (Binary Features)")
    print("="*80)
    binary_groups = [
        ("Has Numbers vs. No Numbers", "src_has_numbers"),
        ("Has Punctuation vs. No Punctuation", "src_has_punctuation")
    ]
    
    group_report = []
    for label, col in binary_groups:
        g0 = df[df[col] == 0]["delta_chrf"]
        g1 = df[df[col] == 1]["delta_chrf"]
        d = compute_cohens_d(g1, g0)
        group_report.append([
            label, 
            round(g0.mean(), 3), 
            round(g1.mean(), 3), 
            round(g1.mean() - g0.mean(), 3), 
            round(d, 3)
        ])

    print(tabulate(
        group_report, 
        headers=["Condition", "Mean Delta (Group 0)", "Mean Delta (Group 1)", "Difference (1 - 0)", "Cohen's d"], 
        tablefmt="fancy_grid"
    ))

    # 5. Length Bins Analysis (Checking for non-linear behavioral shifts)
    df["length_bin"] = pd.cut(
        df["src_char_len"], 
        bins=[0, 25, 50, 100, 1000], 
        labels=["Short (0-25)", "Medium (26-50)", "Long (51-100)", "Very Long (100+)"]
    )
    
    bin_stats = df.groupby("length_bin", observed=False)["delta_chrf"].agg(
        Count="count",
        Mean_Delta="mean",
        Median_Delta="median",
        Std_Dev="std",
        P95=lambda x: x.quantile(0.95),
        P99=lambda x: x.quantile(0.99)
    ).reset_index()

    print("\n" + "="*80)
    print(" STRATIFIED ANALYSIS BY SOURCE LENGTH BINS")
    print("="*80)
    print(tabulate(bin_stats.round(3), headers="keys", tablefmt="fancy_grid", showindex=False))
    print("="*80 + "\n")

    # 6. Save Updated Artifacts
    output_parquet = analysis_dir / f"eda_correlation_report_{run_timestamp}.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved extended EDA dataset to {output_parquet.name}")

    # 7. Generate Non-Seaborn EDA Plots (Matplotlib)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean delta by word count
    word_trends = df.groupby("src_word_count")["delta_chrf"].mean()
    axes[0].plot(word_trends.index, word_trends.values, marker="o", linestyle="-", color="#2b5c8f")
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].set_title("Mean Δ chrF++ Across Source Word Counts")
    axes[0].set_xlabel("Source Word Count")
    axes[0].set_ylabel("Mean Δ chrF++ (Positive = FP32 better)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Box or scatter distribution of lengths vs deltas
    axes[1].scatter(df["src_char_len"], df["delta_chrf"], alpha=0.3, color="#d95f02", s=10)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].set_title("Source Character Length vs. Δ chrF++")
    axes[1].set_xlabel("Source Character Length")
    axes[1].set_ylabel("Δ chrF++ Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = analysis_dir / f"eda_correlation_trends_{run_timestamp}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved correlation trend plot to {plot_path.name}")


if __name__ == "__main__":
    main()