#!/usr/bin/env python3
"""
BridgeDEUX: Step 2B - Token-Level & Model-Derived EDA
======================================================
Investigates whether INT8 translation degradation is correlated with 
model-specific mechanics (subword fragmentation, token counts, generation loops).
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import sacrebleu
from scipy import stats
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TokenEDA")

def find_latest_file(pattern: str) -> Path:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])

def compute_sentence_chrf(hypothesis: str, reference: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    return float(sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score)

def load_jsonl_bak_records(file_path: Path) -> dict:
    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                sid = data["sample_id"]
                records[sid] = data
            except (json.JSONDecodeError, KeyError):
                continue
    return records

def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    
    fp32_pattern = str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "*.jsonl.bak")
    int8_pattern = str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "*.jsonl.bak")

    try:
        fp32_file = find_latest_file(fp32_pattern)
        int8_file = find_latest_file(int8_pattern)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    fp32_data = load_jsonl_bak_records(fp32_file)
    int8_data = load_jsonl_bak_records(int8_file)

    common_ids = set(fp32_data.keys()).intersection(set(int8_data.keys()))
    logger.info(f"Analyzing {len(common_ids):,} common samples for token-level metrics.")

    rows = []
    for sid in common_ids:
        f_rec = fp32_data[sid]
        i_rec = int8_data[sid]

        ref = f_rec.get("reference_translation", "")
        src = f_rec.get("source_text", "")
        
        f_hyp = f_rec.get("translation", "")
        i_hyp = i_rec.get("translation", "")
        
        f_chrf = compute_sentence_chrf(f_hyp, ref)
        i_chrf = compute_sentence_chrf(i_hyp, ref)
        delta_chrf = f_chrf - i_chrf  # Positive = FP32 is better
        
        word_count = max(len(src.split()), 1)
        
        # PRE-INFERENCE METRICS (Knowable before MT runs)
        input_tokens = f_rec.get("input_tokens", 0)
        fragmentation_ratio = input_tokens / word_count
        
        # POST-INFERENCE METRICS (Observing how INT8 failed)
        f_out_tokens = f_rec.get("output_tokens", 0)
        i_out_tokens = i_rec.get("output_tokens", 0)
        delta_out_tokens = f_out_tokens - i_out_tokens  # Positive = FP32 generated more tokens
        
        # Did INT8 hallucinate (generate way too many tokens) or truncate (stop too early)?
        output_token_ratio = i_out_tokens / max(f_out_tokens, 1)

        rows.append({
            "sample_id": sid,
            "delta_chrf": delta_chrf,
            "input_tokens": input_tokens,
            "subword_fragmentation_ratio": fragmentation_ratio,
            "fp32_output_tokens": f_out_tokens,
            "int8_output_tokens": i_out_tokens,
            "delta_output_tokens": delta_out_tokens,
            "abs_delta_output_tokens": abs(delta_out_tokens),
            "int8_to_fp32_output_ratio": output_token_ratio
        })

    df = pd.DataFrame(rows)

    # 1. Correlation Analysis
    logger.info("Computing correlations for model-derived features...")
    features = [
        "input_tokens", 
        "subword_fragmentation_ratio", 
        "delta_output_tokens", 
        "abs_delta_output_tokens",
        "int8_to_fp32_output_ratio"
    ]
    
    corr_results = []
    for feat in features:
        pearson_r, p_p = stats.pearsonr(df["delta_chrf"], df[feat])
        spearman_rho, p_s = stats.spearmanr(df["delta_chrf"], df[feat])
        corr_results.append([feat, round(pearson_r, 4), f"{p_p:.2e}", round(spearman_rho, 4), f"{p_s:.2e}"])

    print("\n" + "="*80)
    print(" STEP 2B: TOKEN & MODEL-MECHANIC CORRELATIONS (vs. Δ chrF++)")
    print("="*80)
    print(tabulate(corr_results, headers=["Feature", "Pearson (r)", "p-val", "Spearman (rho)", "p-val"], tablefmt="fancy_grid"))

    # 2. Output Token Behavior Deep Dive (Does INT8 truncate or hallucinate?)
    print("\n" + "="*80)
    print(" GENERATION COLLAPSE ANALYSIS")
    print("="*80)
    
    # Categorize INT8 behavior relative to FP32
    df["int8_behavior"] = "Normal (±2 tokens)"
    df.loc[df["delta_output_tokens"] > 2, "int8_behavior"] = "Truncated (Early EOS)"
    df.loc[df["delta_output_tokens"] < -2, "int8_behavior"] = "Hallucinated (Runaway Gen)"
    
    behavior_stats = df.groupby("int8_behavior", observed=False)["delta_chrf"].agg(
        Count="count",
        Mean_Delta_chrF="mean",
        P95_Delta="quantile"
    )
    # Using 0.95 quantile properly inside groupby agg
    behavior_stats["P95_Delta"] = df.groupby("int8_behavior", observed=False)["delta_chrf"].quantile(0.95)
    
    print(tabulate(behavior_stats.reset_index(), headers="keys", tablefmt="fancy_grid", showindex=False))
    print("\n")

if __name__ == "__main__":
    main()