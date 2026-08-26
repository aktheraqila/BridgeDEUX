#!/usr/bin/env python3
"""
BridgeDEUX: Sequence Length and Position Error Accumulation
===========================================================
Tests whether the ASR perturbation increase is actually an artifact of 
sequence length and autoregressive error compounding.
"""

import pandas as pd

def main():
    print("Loading Level 2 token data (this takes just a second)...\n")
    try:
        clean = pd.read_parquet("analysis/level2_margin_tokens_clean.parquet")
        asr = pd.read_parquet("analysis/level2_margin_tokens_asr.parquet")
    except FileNotFoundError:
        print("Error: Could not find Level 2 parquet files. Make sure Level 2 finished.")
        return

    # ---------------------------------------------------------
    # TEST 1: Matched Length Comparison
    # ---------------------------------------------------------
    print("=" * 75)
    print(" TEST 1: MATCHED LENGTH COMPARISON")
    print("=" * 75)
    
    clean_agg = clean.groupby("sample_id").agg(pert=("mean_abs_pert", "mean"), n=("seq_len", "first")).reset_index()
    clean_agg["cond"] = "clean"
    
    asr_agg = asr.groupby("sample_id").agg(pert=("mean_abs_pert", "mean"), n=("seq_len", "first")).reset_index()
    asr_agg["cond"] = "asr"
    
    both = pd.concat([clean_agg, asr_agg])
    both["nbin"] = pd.cut(both.n, [0, 8, 11, 14, 18, 25, 10000])
    
    piv = both.pivot_table(index="nbin", columns="cond", values="pert", aggfunc="mean", observed=True)
    piv["delta_pct"] = 100 * (piv.asr / piv.clean - 1)
    piv["n_clean"] = both[both.cond == "clean"].groupby("nbin", observed=True).size()
    piv["n_asr"]   = both[both.cond == "asr"].groupby("nbin", observed=True).size()
    
    print(piv.round(4).to_string())
    print("\n[Interpretation]")
    print("-> If delta_pct stays near +5-12%, length does NOT explain the ASR penalty.")
    print("-> If delta_pct drops near 0%, length DOES explain it.")

    # ---------------------------------------------------------
    # TEST 2: Error Accumulation by Position
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" TEST 2: ERROR ACCUMULATION BY POSITION")
    print("=" * 75)
    
    for name, d in [("clean", clean), ("asr", asr)]:
        d["pband"] = pd.cut(d.rel_position, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
        print(f"\n{name.upper()}: Mean logit perturbation by relative position")
        print(d.groupby("pband", observed=True).mean_abs_pert.mean().round(4).to_string())
        
    print("\n[Interpretation]")
    print("-> If perturbation rises across the bands, quantization error compounds")
    print("   as the decoder conditions on its own output.")

if __name__ == "__main__":
    main()