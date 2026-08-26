#!/usr/bin/env python3
"""
BridgeDEUX: Step 5C - Methodological DiD Evaluation (chrF++ Corrected)
======================================================================
Calculates genuine chrF++ (word_order=2) and performs a 10,000-iteration 
paired bootstrap on sentence-level Difference-in-Differences (DiD).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import sacrebleu
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def compute_corpus_chrfpp(hyps, refs):
    """Explicitly computes chrF++ by requiring word_order=2."""
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score

def compute_sentence_chrfpp(hyp, ref):
    """Explicitly computes sentence-level chrF++ (word_order=2)."""
    return sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    cohort_path = analysis_dir / "matched_cohort_complete.parquet"
    
    if not cohort_path.exists():
        print("ERROR: Run 05b_run_asr_translations.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(cohort_path)
    N = len(df)
    
    print("=" * 80)
    print(f" STEP 5C: STATISTICAL EVALUATION & DiD (N={N})")
    print("=" * 80)
    
    refs = df["gold_english_reference"].tolist()
    clean_f = df["clean_fp32_translation"].tolist()
    clean_i = df["clean_int8_translation"].tolist()
    asr_f = df["asr_fp32_translation"].tolist()
    asr_i = df["asr_int8_translation"].tolist()

    print("1. Computing genuine Corpus-level chrF++ (word_order=2)...")
    corp_clean_f = compute_corpus_chrfpp(clean_f, refs)
    corp_clean_i = compute_corpus_chrfpp(clean_i, refs)
    corp_asr_f = compute_corpus_chrfpp(asr_f, refs)
    corp_asr_i = compute_corpus_chrfpp(asr_i, refs)
    
    corp_delta_clean = corp_clean_i - corp_clean_f
    corp_delta_asr = corp_asr_i - corp_asr_f
    corp_did = corp_delta_asr - corp_delta_clean

    print("2. Computing Sentence-level chrF++ for statistical testing...")
    chrf_clean_f, chrf_clean_i, chrf_asr_f, chrf_asr_i = [], [], [], []
    
    for i in tqdm(range(N), desc="Scoring sentences"):
        r = refs[i]
        chrf_clean_f.append(compute_sentence_chrfpp(clean_f[i], r))
        chrf_clean_i.append(compute_sentence_chrfpp(clean_i[i], r))
        chrf_asr_f.append(compute_sentence_chrfpp(asr_f[i], r))
        chrf_asr_i.append(compute_sentence_chrfpp(asr_i[i], r))
        
    df["chrf_clean_f"] = chrf_clean_f
    df["chrf_clean_i"] = chrf_clean_i
    df["chrf_asr_f"] = chrf_asr_f
    df["chrf_asr_i"] = chrf_asr_i
    
    # Sentence-level deltas
    df["delta_clean"] = df["chrf_clean_i"] - df["chrf_clean_f"]
    df["delta_asr"] = df["chrf_asr_i"] - df["chrf_asr_f"]
    df["did"] = df["delta_asr"] - df["delta_clean"]
    
    print("3. Executing Paired Bootstrap Resampling (B=10,000)...")
    np.random.seed(42)
    did_array = df["did"].values
    boot_means = np.empty(10000)
    for b in range(10000):
        sample = np.random.choice(did_array, size=N, replace=True)
        boot_means[b] = np.mean(sample)
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    print("\n" + "=" * 80)
    print(" RESULTS PART 1: OVERALL QUALITY & ROBUSTNESS GAP")
    print("=" * 80)
    print("Corpus-Level chrF++ (word_order=2):")
    print(f"  Clean Input -> FP32: {corp_clean_f:.3f} | INT8: {corp_clean_i:.3f} | Δ(INT8-FP32): {corp_delta_clean:+.3f}")
    print(f"  ASR Input   -> FP32: {corp_asr_f:.3f} | INT8: {corp_asr_i:.3f} | Δ(INT8-FP32): {corp_delta_asr:+.3f}")
    print(f"  Corpus DiD  -> {corp_did:+.3f} chrF++ points")
    print("-" * 80)
    print("Behavioral Divergence (Exact String Mismatch):")
    clean_div = (df["clean_fp32_translation"] != df["clean_int8_translation"]).mean() * 100
    asr_div = (df["asr_fp32_translation"] != df["asr_int8_translation"]).mean() * 100
    print(f"  Clean Input : {clean_div:.2f}%")
    print(f"  ASR Input   : {asr_div:.2f}% (Absolute increase of {asr_div - clean_div:+.2f}%)")
    print("-" * 80)
    print("Statistical Significance (Sentence-Level Bootstrap):")
    print(f"  Mean DiD            : {np.mean(did_array):+.4f}")
    print(f"  95% Confidence Int. : [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    if ci_lower < 0 and ci_upper < 0:
        print("  Verdict             : SIGNIFICANT DEGRADATION (CI is entirely below 0).")
    elif ci_lower > 0 and ci_upper > 0:
        print("  Verdict             : SIGNIFICANT IMPROVEMENT (CI is entirely above 0).")
    else:
        print("  Verdict             : NULL RESULT (CI crosses 0. No proven differential effect).")
        
    print("\n" + "=" * 80)
    print(" RESULTS PART 2: WER-STRATIFIED EXPLORATION")
    print("=" * 80)
    
    # Data-driven exploratory bins based on observed percentiles
    bins = [-0.01, 0.15, 0.45, 0.90, float('inf')]
    labels = ["0-15% WER (Q1: Clean/Usable)", "15-45% WER (Q2-Q3: Noisy)", "45-90% WER (Q4: Severe)", ">90% WER (Tail: Pathological)"]
    df["wer_bin"] = pd.cut(df["asr_wer"], bins=bins, labels=labels)
    
    print(f"{'WER Bin':<30} | {'N':<5} | {'ASR Div %':<10} | {'Δ_Clean':<8} | {'Δ_ASR':<8} | {'Mean DiD'}")
    print("-" * 80)
    for label in labels:
        sub = df[df["wer_bin"] == label]
        if len(sub) > 0:
            s_div = (sub["asr_fp32_translation"] != sub["asr_int8_translation"]).mean() * 100
            s_d_clean = sub["delta_clean"].mean()
            s_d_asr = sub["delta_asr"].mean()
            s_did = sub["did"].mean()
            print(f"{label:<30} | {len(sub):<5} | {s_div:8.1f}%  | {s_d_clean:+7.3f}  | {s_d_asr:+7.3f}  | {s_did:+7.3f}")
            
    print("=" * 80)
    
    # Save the fully scored cohort
    out_path = analysis_dir / "scored_qirg_cohort.parquet"
    df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()