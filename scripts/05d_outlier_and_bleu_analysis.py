#!/usr/bin/env python3
"""
BridgeDEUX: Step 5D - BLEU and Outlier Analysis
===============================================
Computes standard BLEU for the corpus and isolates severe sentence-level 
failures to determine if the DiD is driven by catastrophic outliers.
"""

import sys
import pandas as pd
from pathlib import Path
import sacrebleu

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    cohort_path = analysis_dir / "scored_qirg_cohort.parquet"
    
    if not cohort_path.exists():
        print("ERROR: Run 05c_evaluate_qirg_did.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(cohort_path)
    N = len(df)
    
    print("=" * 80)
    print(" STEP 5D: BLEU & OUTLIER ANALYSIS")
    print("=" * 80)
    
    refs = df["gold_english_reference"].tolist()
    clean_f = df["clean_fp32_translation"].tolist()
    clean_i = df["clean_int8_translation"].tolist()
    asr_f = df["asr_fp32_translation"].tolist()
    asr_i = df["asr_int8_translation"].tolist()

    # 1. Calculate Corpus BLEU
    print("1. Computing Corpus-Level BLEU...")
    bleu_clean_f = sacrebleu.corpus_bleu(clean_f, [refs]).score
    bleu_clean_i = sacrebleu.corpus_bleu(clean_i, [refs]).score
    bleu_asr_f = sacrebleu.corpus_bleu(asr_f, [refs]).score
    bleu_asr_i = sacrebleu.corpus_bleu(asr_i, [refs]).score
    
    did_bleu = (bleu_asr_i - bleu_asr_f) - (bleu_clean_i - bleu_clean_f)
    
    print(f"  Clean Input -> FP32: {bleu_clean_f:.2f} | INT8: {bleu_clean_i:.2f} | Δ: {bleu_clean_i - bleu_clean_f:+.2f}")
    print(f"  ASR Input   -> FP32: {bleu_asr_f:.2f} | INT8: {bleu_asr_i:.2f} | Δ: {bleu_asr_i - bleu_asr_f:+.2f}")
    print(f"  Corpus BLEU DiD : {did_bleu:+.2f} points")
    print("-" * 80)

    # 2. Outlier Analysis
    print("2. Catastrophic Outlier Analysis (Sentence-Level)")
    
    # Define a catastrophic failure as INT8 losing 5.0+ more chrF++ points than FP32 under ASR shift
    catastrophic_mask = df["did"] <= -5.0
    severe_mask = df["did"] <= -10.0
    
    num_catastrophic = catastrophic_mask.sum()
    num_severe = severe_mask.sum()
    
    print(f"  Total Sentences                      : {N}")
    print(f"  Sentences where INT8 DiD <= -5.0     : {num_catastrophic} ({(num_catastrophic/N)*100:.2f}%)")
    print(f"  Sentences where INT8 DiD <= -10.0    : {num_severe} ({(num_severe/N)*100:.2f}%)")
    
    if num_catastrophic > 0:
        print("\n  Top 3 Worst Catastrophic Failures (Qualitative Inspection):")
        worst_cases = df[catastrophic_mask].sort_values(by="did").head(3)
        for idx, row in worst_cases.iterrows():
            print(f"\n  --- Sample ID: {row['sample_id']} | DiD: {row['did']:.2f} ---")
            print(f"  Gold Source : {row['gold_german_source']}")
            print(f"  Whisper ASR : {row['whisper_hypothesis']} (WER: {row['asr_wer']:.2f})")
            print(f"  English Ref : {row['gold_english_reference']}")
            print(f"  ASR -> FP32 : {row['asr_fp32_translation']}")
            print(f"  ASR -> INT8 : {row['asr_int8_translation']}")
            
    print("=" * 80)

if __name__ == "__main__":
    main()