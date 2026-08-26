#!/usr/bin/env python3
"""
BridgeDEUX: Step 5F - Final COMET Statistical Assembly
======================================================
Reads the finalized CheckpointManager Parquet, merges it with the cohort, 
and calculates the definitive 10,000-iteration bootstrap CI.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    
    # 1. Load the original text cohort
    cohort_path = analysis_dir / "scored_qirg_cohort.parquet"
    if not cohort_path.exists():
        print("FATAL: Could not find original cohort.")
        sys.exit(1)
    df_text = pd.read_parquet(cohort_path)
    N_TOTAL = len(df_text)
    
    # 2. Load the newly generated COMET scores from the correct subfolder
    comet_path = analysis_dir / "comet_wmt20_qirg" / "comet_wmt20_qirg_results.parquet"
    if not comet_path.exists():
        print(f"FATAL: Could not find COMET results at {comet_path}")
        sys.exit(1)
        
    print("=" * 80)
    print(" STEP 5F: FINAL COMET STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Loading COMET scores from: {comet_path.name}")
    
    df_scores = pd.read_parquet(comet_path)
    
    # Merge on sample_id
    df_final = pd.merge(df_text, df_scores, on="sample_id", how="inner")
    
    if len(df_final) != N_TOTAL:
        print(f"WARNING: Merged length ({len(df_final)}) does not match expected ({N_TOTAL})!")
        
    # 3. Calculate End-to-End Semantic DiD
    df_final["comet_delta_clean"] = df_final["comet_clean_i"] - df_final["comet_clean_f"]
    df_final["comet_delta_asr"] = df_final["comet_asr_i_e2e"] - df_final["comet_asr_f_e2e"]
    df_final["comet_did"] = df_final["comet_delta_asr"] - df_final["comet_delta_clean"]
    
    mean_clean_f = df_final["comet_clean_f"].mean()
    mean_clean_i = df_final["comet_clean_i"].mean()
    mean_asr_f   = df_final["comet_asr_f_e2e"].mean()
    mean_asr_i   = df_final["comet_asr_i_e2e"].mean()
    mean_did     = df_final["comet_did"].mean()
    
    print("Executing Paired Bootstrap Resampling (B=10,000) for E2E COMET DiD...")
    np.random.seed(42)
    did_array = df_final["comet_did"].values
    boot_means = np.empty(10000)
    for b in range(10000):
        sample = np.random.choice(did_array, size=N_TOTAL, replace=True)
        boot_means[b] = np.mean(sample)
    
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    
    print("\n" + "=" * 80)
    print(" FINAL RESULTS (END-TO-END SEMANTIC PRESERVATION)")
    print("=" * 80)
    print(f"  Clean Input -> FP32: {mean_clean_f:.4f} | INT8: {mean_clean_i:.4f} | Δ: {mean_clean_i - mean_clean_f:+.4f}")
    print(f"  ASR Input   -> FP32: {mean_asr_f:.4f} | INT8: {mean_asr_i:.4f} | Δ: {mean_asr_i - mean_asr_f:+.4f}")
    print("-" * 80)
    print(f"  Semantic Difference-in-Differences (DiD) : {mean_did:+.4f} COMET points")
    print(f"  95% Paired Bootstrap Confidence Interval : [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print("-" * 80)
    
    if ci_lower < 0 and ci_upper < 0:
        print("  Verdict : STATISTICALLY SIGNIFICANT SEMANTIC DEGRADATION.")
    elif ci_lower > 0 and ci_upper > 0:
        print("  Verdict : STATISTICALLY SIGNIFICANT SEMANTIC IMPROVEMENT.")
    else:
        print("  Verdict : NULL RESULT (CI crosses 0. No proven differential semantic effect).")
        
    out_path = analysis_dir / "scored_qirg_cohort_final.parquet"
    df_final.to_parquet(out_path, index=False)
    print("=" * 80)
    print(f"SUCCESS: Final comprehensive dataset saved to: {out_path.name}")

if __name__ == "__main__":
    main()