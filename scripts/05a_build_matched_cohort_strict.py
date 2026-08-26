#!/usr/bin/env python3
"""
BridgeDEUX: Step 5A - Strict Matched Cohort Builder
===================================================
Verifies and joins the 13,411 Whisper hypotheses with the clean FP32 
and INT8 MT results using strict sample_id inner joins and verified schemas.
"""

import sys
import json
import pandas as pd
from pathlib import Path

def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows)

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Exact verified paths
    whisper_path = repo_root / "results" / "whisper.cpp (base)_test" / "whisper.cpp (base)_test_results_20260716_203250_747588.jsonl.bak"
    fp32_path = repo_root / "results" / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test_results_20260802_204633_269550.jsonl.bak"
    int8_path = repo_root / "results" / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test_results_20260807_101154_066233.jsonl.bak"
    
    print("=" * 75)
    print(" STEP 5A: STRICT COHORT BUILDER")
    print("=" * 75)
    
    print("1. Loading datasets...")
    df_w = load_jsonl(whisper_path)
    df_f = load_jsonl(fp32_path)
    df_i = load_jsonl(int8_path)
    
    # Strict Column Assertions based on verified terminal output
    assert all(k in df_w.columns for k in ["sample_id", "source_text", "hypothesis", "wer", "cer"]), "Whisper schema mismatch"
    assert all(k in df_f.columns for k in ["sample_id", "source_text", "reference_translation", "translation"]), "FP32 schema mismatch"
    assert all(k in df_i.columns for k in ["sample_id", "source_text", "reference_translation", "translation"]), "INT8 schema mismatch"

    # Subset and Standardize
    df_w_sub = df_w[["sample_id", "source_text", "hypothesis", "wer", "cer"]].rename(
        columns={"source_text": "whisper_gold_source", "hypothesis": "whisper_hypothesis", "wer": "asr_wer", "cer": "asr_cer"}
    )
    df_f_sub = df_f[["sample_id", "source_text", "reference_translation", "translation"]].rename(
        columns={"source_text": "mt_gold_source_fp32", "reference_translation": "gold_english_reference", "translation": "clean_fp32_translation"}
    )
    df_i_sub = df_i[["sample_id", "source_text", "reference_translation", "translation"]].rename(
        columns={"source_text": "mt_gold_source_int8", "reference_translation": "gold_english_reference_int8", "translation": "clean_int8_translation"}
    )

    print("\n2. Executing strict INNER JOINs on sample_id...")
    cohort = pd.merge(df_w_sub, df_f_sub, on="sample_id", how="inner")
    cohort = pd.merge(cohort, df_i_sub, on="sample_id", how="inner")
    
    print(f"   Common Cohort Size: {len(cohort)} samples")
    
    if len(cohort) != 13411:
        print(f"FATAL: Expected exactly 13,411 matched samples, but got {len(cohort)}.")
        sys.exit(1)
        
    print("\n3. Validating text alignment across datasets...")
    src_mismatch = (cohort["whisper_gold_source"] != cohort["mt_gold_source_fp32"]).sum()
    ref_mismatch = (cohort["gold_english_reference"] != cohort["gold_english_reference_int8"]).sum()
    
    print(f"   Source Text Mismatches (Whisper vs FP32) : {src_mismatch}")
    print(f"   Reference Mismatches (FP32 vs INT8)      : {ref_mismatch}")
    
    if src_mismatch > 0 or ref_mismatch > 0:
        print("FATAL: Source or reference text misalignment detected. Do not proceed.")
        sys.exit(1)

    # Final Cleanup: Keep only the canonical columns
    final_cohort = cohort[[
        "sample_id", "whisper_gold_source", "gold_english_reference", 
        "whisper_hypothesis", "asr_wer", "asr_cer", 
        "clean_fp32_translation", "clean_int8_translation"
    ]].rename(columns={"whisper_gold_source": "gold_german_source"})

    out_path = analysis_dir / "matched_cohort_base.parquet"
    final_cohort.to_parquet(out_path, index=False)
    
    print(f"\nSUCCESS: Immutable matched cohort saved to {out_path.name}")
    print("=" * 75)

if __name__ == "__main__":
    main()