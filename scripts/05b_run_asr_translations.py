#!/usr/bin/env python3
"""
BridgeDEUX: Step 5B - Run ASR-Input Translations
================================================
Generates translations from Whisper hypotheses using Marian FP32 and INT8.
Appends the results to the matched cohort for final statistical evaluation.
"""

import sys
import pandas as pd
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    cohort_path = analysis_dir / "matched_cohort_base.parquet"
    
    if not cohort_path.exists():
        print("ERROR: Run 05a_build_matched_cohort_strict.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(cohort_path)
    N = len(df)
    
    print("=" * 75)
    print(f" STEP 5B: RUNNING ASR TRANSLATIONS (N={N})")
    print("=" * 75)
    
    fp32_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended"
    int8_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended_int8"
    
    print("Loading ONNX models...")
    tokenizer = MarianTokenizer.from_pretrained(fp32_dir)
    fp32_model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, provider="CPUExecutionProvider")
    int8_model = ORTModelForSeq2SeqLM.from_pretrained(int8_dir, provider="CPUExecutionProvider")
    
    BATCH_SIZE = 16
    fp32_asr_outs = []
    int8_asr_outs = []
    
    print("\nRunning Inference (FP32 & INT8)...")
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Translating ASR Hypotheses"):
        batch_raw = df["whisper_hypothesis"].iloc[i:i+BATCH_SIZE].tolist()
        
        # Failsafe: if Whisper output was completely empty, feed a period to avoid tokenizer crashes
        batch_clean = [t if str(t).strip() else "." for t in batch_raw]
        
        inputs = tokenizer(batch_clean, return_tensors="pt", padding=True, truncation=True)
        
        # 1. FP32 Generation
        out_f = fp32_model.generate(**inputs, max_length=150)
        fp32_asr_outs.extend(tokenizer.batch_decode(out_f, skip_special_tokens=True))
        
        # 2. INT8 Generation
        out_i = int8_model.generate(**inputs, max_length=150)
        int8_asr_outs.extend(tokenizer.batch_decode(out_i, skip_special_tokens=True))
        
    df["asr_fp32_translation"] = fp32_asr_outs
    df["asr_int8_translation"] = int8_asr_outs
    
    out_path = analysis_dir / "matched_cohort_complete.parquet"
    df.to_parquet(out_path, index=False)
    
    print(f"\nInference complete. Saved full dataset to: {out_path.name}")
    print("=" * 75)

if __name__ == "__main__":
    main()