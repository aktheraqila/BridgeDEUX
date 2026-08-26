#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

def get_latest(pattern):
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    # Load the latest COMET parquet file which contains the texts
    parquet_file = get_latest("analysis/comet_divergent_scores_*.parquet")
    df = pd.read_parquet(parquet_file)
    
    print("="*60)
    print(" 1. RECONCILING THE 39-SEGMENT DISCREPANCY")
    print("="*60)
    # Find rows where chrF++ Delta is exactly 0, but the strings are different
    ghosts = df[df['delta_chrf'] == 0]
    print(f"Found {len(ghosts)} 'Ghost' segments (ΔchrF++ = 0, but strings differ).")
    
    if len(ghosts) > 0:
        print("\nExamples of metric collision (chrF++ blindness):")
        for i, row in ghosts.head(3).iterrows():
            print(f"\nID  : {row['sample_id']}")
            print(f"FP32: '{row['fp32_translation']}'")
            print(f"INT8: '{row['int8_translation']}'")
            
    print("\n" + "="*60)
    print(" 2. BINOMIAL TEST: TRUNCATION ASYMMETRY")
    print("="*60)
    fp32_longer = len(df[df['delta_output_tokens'] > 2])  # INT8 truncated
    int8_longer = len(df[df['delta_output_tokens'] < -2]) # FP32 truncated
    total_truncations = fp32_longer + int8_longer
    
    # Two-sided binomial test (Null hypothesis: p = 0.5)
    p_value = stats.binomtest(int8_longer, n=total_truncations, p=0.5, alternative='two-sided').pvalue
    
    print(f"FP32 Truncations (INT8 longer): {int8_longer}")
    print(f"INT8 Truncations (FP32 longer): {fp32_longer}")
    print(f"Total severe length divergences : {total_truncations}")
    print(f"Binomial Test p-value         : {p_value:.4f}")
    
    if p_value > 0.05:
        print("\nCONCLUSION: Fail to reject the null hypothesis.")
        print("The rates of truncation are statistically indistinguishable.")
        print("We CANNOT claim INT8 or FP32 degenerates more often.")
    else:
        print("\nCONCLUSION: Reject the null hypothesis. The asymmetry is significant.")

if __name__ == "__main__":
    main()