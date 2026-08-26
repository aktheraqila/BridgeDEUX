#!/usr/bin/env python3
"""
BridgeDEUX: The Real Mechanism (dmargin check)
==============================================
Validates whether INT8 noise shrinks the top-2 margin more aggressively 
under ASR input compared to Clean input.
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print(" INT8 EFFECT ON TOP-2 MARGIN (The true cause of flips)")
    print("=" * 60)
    
    try:
        clean = pd.read_parquet("analysis/level2_margin_tokens_clean.parquet")
        asr = pd.read_parquet("analysis/level2_margin_tokens_asr.parquet")
    except FileNotFoundError:
        print("Error: Could not find Level 2 parquet files.")
        return

    for cond, d in [("CLEAN", clean), ("ASR", asr)]:
        # How much did INT8 shrink or grow the gap between #1 and #2?
        # Negative means INT8 shrunk the gap (made the model less confident)
        d["dmargin"] = d.margin_int8 - d.margin_fp32
        
        print(f"\n=== {cond} CONDITION ===")
        print("Overall margin change (INT8 - FP32):")
        print(d.dmargin.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(4))
        
        # We only care about the Danger Zone where flips actually happen
        near = d[d.margin_fp32 < 1.0]
        print(f"\nDanger Zone positions (FP32 Margin < 1.0), n={len(near):,}:")
        print(f"  Mean dmargin   (Shrinkage) : {near.dmargin.mean():+.4f}")
        print(f"  P(dmargin < 0) (Gap Shrunk)  : {100*(near.dmargin<0).mean():.2f}%")
        print(f"  Mean |dmargin| (Volatility)  : {near.dmargin.abs().mean():.4f}")

    print("\n" + "=" * 60)
    print(" INTERPRETATION")
    print("=" * 60)
    print("If ASR has a more negative Mean dmargin in the Danger Zone,")
    print("it proves ASR noise actively destroys model confidence, causing")
    print("the #2 token to overtake the #1 token more frequently.")

if __name__ == "__main__":
    main()