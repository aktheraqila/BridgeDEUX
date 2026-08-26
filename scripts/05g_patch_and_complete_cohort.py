#!/usr/bin/env python3
"""
BridgeDEUX: Step 5G - Patch Missing 100 Samples & Complete Cohort
=================================================================
Isolates the 100 missing CoVoST2 samples, runs ONNX inference on their ASR transcripts
(exactly mirroring Step 5B), scores them (chrF++ & COMET), and merges them to create 
the definitive 13,511 cohort without mutating the original 13,411 audit artifact.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import sacrebleu
from comet import download_model, load_from_checkpoint
import logging
import warnings
from tqdm import tqdm

# Optimum/Transformers for ONNX inference exactly as in Step 5B
from transformers import MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def compute_sentence_chrfpp(hyp, ref):
    return sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    results_dir = repo_root / "results"
    
    existing_cohort_path = analysis_dir / "scored_qirg_cohort_final.parquet"
    if not existing_cohort_path.exists():
        print(f"FATAL: Missing {existing_cohort_path.name}")
        sys.exit(1)
        
    df_existing = pd.read_parquet(existing_cohort_path)
    existing_ids = set(df_existing["sample_id"].astype(str))
    
    print("=" * 80)
    print(" STEP 5G: PATCHING 100 SAMPLES & FINAL COMPILATION")
    print("=" * 80)
    
    # 1. Load full source files
    print("1. Loading full 13,511 datasets...")
    df_w_full = pd.read_parquet(results_dir / "whisper.cpp (base)_test" / "whisper.cpp (base)_test_results.parquet")
    df_f_full = pd.read_parquet(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test_results.parquet")
    df_i_full = pd.read_parquet(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test_results.parquet")
    
    # 2. Isolate missing IDs
    missing_ids = set(df_w_full["sample_id"].astype(str)) - existing_ids
    print(f"   Missing Samples Found: {len(missing_ids)}")
    
    if len(missing_ids) != 100:
        print(f"FATAL: Expected exactly 100 missing samples, found {len(missing_ids)}. Aborting.")
        sys.exit(1)
        
    # 3. Build the base 100-sample dataframe
    df_w_sub = df_w_full[df_w_full["sample_id"].astype(str).isin(missing_ids)][["sample_id", "source_text", "hypothesis", "wer", "cer"]]
    df_w_sub = df_w_sub.rename(columns={"source_text": "gold_german_source", "hypothesis": "whisper_hypothesis", "wer": "asr_wer", "cer": "asr_cer"})
    
    df_f_sub = df_f_full[df_f_full["sample_id"].astype(str).isin(missing_ids)][["sample_id", "reference_translation", "translation"]]
    df_f_sub = df_f_sub.rename(columns={"reference_translation": "gold_english_reference", "translation": "clean_fp32_translation"})
    
    df_i_sub = df_i_full[df_i_full["sample_id"].astype(str).isin(missing_ids)][["sample_id", "translation"]]
    df_i_sub = df_i_sub.rename(columns={"translation": "clean_int8_translation"})
    
    df_patch = pd.merge(df_w_sub, df_f_sub, on="sample_id", how="inner")
    df_patch = pd.merge(df_patch, df_i_sub, on="sample_id", how="inner")
    
    if len(df_patch) != 100:
        print(f"FATAL: Inner join reduced the patch size to {len(df_patch)}. Expected 100. Aborting.")
        sys.exit(1)
    
    # 4. Generate ASR->MT Translations (Exact Step 5B Mirror)
    print("\n2. Executing ONNX Inference on the 100 ASR Hypotheses...")
    fp32_dir = repo_root / "models" / "onnx" / "opus_mt_de_en_opt_extended"
    int8_dir = repo_root / "models" / "onnx" / "opus_mt_de_en_opt_extended_int8"
    
    print("   Loading local ONNX models and tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(fp32_dir)
    fp32_model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, provider="CPUExecutionProvider")
    int8_model = ORTModelForSeq2SeqLM.from_pretrained(int8_dir, provider="CPUExecutionProvider")
    
    BATCH_SIZE = 16
    N_PATCH = len(df_patch)
    fp32_asr_outs = []
    int8_asr_outs = []
    
    for i in range(0, N_PATCH, BATCH_SIZE):
        batch_raw = df_patch["whisper_hypothesis"].iloc[i:i+BATCH_SIZE].tolist()
        # 5B Failsafe: if Whisper output was empty, feed a period
        batch_clean = [t if str(t).strip() else "." for t in batch_raw]
        
        inputs = tokenizer(batch_clean, return_tensors="pt", padding=True, truncation=True)
        
        out_f = fp32_model.generate(**inputs, max_length=150)
        fp32_asr_outs.extend(tokenizer.batch_decode(out_f, skip_special_tokens=True))
        
        out_i = int8_model.generate(**inputs, max_length=150)
        int8_asr_outs.extend(tokenizer.batch_decode(out_i, skip_special_tokens=True))
        
    df_patch["asr_fp32_translation"] = fp32_asr_outs
    df_patch["asr_int8_translation"] = int8_asr_outs
    del fp32_model, int8_model, tokenizer
    
    # 5. Score chrF++
    print("\n3. Calculating chrF++ (word_order=2) for the patch cohort...")
    chrf_cf, chrf_ci, chrf_af, chrf_ai = [], [], [], []
    for _, row in df_patch.iterrows():
        ref = row["gold_english_reference"]
        chrf_cf.append(compute_sentence_chrfpp(row["clean_fp32_translation"], ref))
        chrf_ci.append(compute_sentence_chrfpp(row["clean_int8_translation"], ref))
        chrf_af.append(compute_sentence_chrfpp(row["asr_fp32_translation"], ref))
        chrf_ai.append(compute_sentence_chrfpp(row["asr_int8_translation"], ref))
        
    df_patch["chrf_clean_f"] = chrf_cf
    df_patch["chrf_clean_i"] = chrf_ci
    df_patch["chrf_asr_f"] = chrf_af
    df_patch["chrf_asr_i"] = chrf_ai
    df_patch["delta_clean"] = df_patch["chrf_clean_i"] - df_patch["chrf_clean_f"]
    df_patch["delta_asr"] = df_patch["chrf_asr_i"] - df_patch["chrf_asr_f"]
    df_patch["did"] = df_patch["delta_asr"] - df_patch["delta_clean"]
    
    bins = [-0.01, 0.15, 0.45, 0.90, float('inf')]
    labels = ["0-15% WER (Q1: Clean/Usable)", "15-45% WER (Q2-Q3: Noisy)", "45-90% WER (Q4: Severe)", ">90% WER (Tail: Pathological)"]
    df_patch["wer_bin"] = pd.cut(df_patch["asr_wer"], bins=bins, labels=labels)
    
    # 6. Score COMET
    print("\n4. Loading COMET and running 6 passes on 100 samples (~2 mins)...")
    comet_model_name = "Unbabel/wmt20-comet-da"
    model_path = download_model(comet_model_name)
    comet_model = load_from_checkpoint(model_path)
    comet_model.eval()
    
    passes = {
        "comet_clean_f": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df_patch["gold_german_source"], df_patch["clean_fp32_translation"], df_patch["gold_english_reference"])],
        "comet_clean_i": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df_patch["gold_german_source"], df_patch["clean_int8_translation"], df_patch["gold_english_reference"])],
        "comet_asr_f_e2e": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df_patch["gold_german_source"], df_patch["asr_fp32_translation"], df_patch["gold_english_reference"])],
        "comet_asr_i_e2e": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df_patch["gold_german_source"], df_patch["asr_int8_translation"], df_patch["gold_english_reference"])],
        "comet_asr_f_mt": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df_patch["whisper_hypothesis"], df_patch["asr_fp32_translation"], df_patch["gold_english_reference"])],
        "comet_asr_i_mt": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df_patch["whisper_hypothesis"], df_patch["asr_int8_translation"], df_patch["gold_english_reference"])],
    }
    
    for col_name, data in passes.items():
        out = comet_model.predict(data, batch_size=16, gpus=0)
        df_patch[col_name] = out.scores
        
    df_patch["comet_delta_clean"] = df_patch["comet_clean_i"] - df_patch["comet_clean_f"]
    df_patch["comet_delta_asr"] = df_patch["comet_asr_i_e2e"] - df_patch["comet_asr_f_e2e"]
    df_patch["comet_did"] = df_patch["comet_delta_asr"] - df_patch["comet_delta_clean"]
    
    # Save the 100-sample patch for the audit trail
    patch_artifact = analysis_dir / "qirg_cohort_patch_100.parquet"
    df_patch.to_parquet(patch_artifact, index=False)
    print(f"   Patch artifact saved: {patch_artifact.name}")
    
    # 7. Merge into final Complete Cohort
    print("\n5. Merging patch into Complete Cohort...")
    df_complete = pd.concat([df_existing, df_patch], ignore_index=True)
    
    if len(df_complete) != 13511:
        print(f"FATAL: Final cohort size {len(df_complete)} does not equal 13,511. Aborting save.")
        sys.exit(1)
        
    # Re-sort to maintain determinism
    df_complete = df_complete.sort_values("sample_id").reset_index(drop=True)
    
    complete_artifact = analysis_dir / "scored_qirg_cohort_complete.parquet"
    df_complete.to_parquet(complete_artifact, index=False)
    print(f"   Complete cohort saved: {complete_artifact.name}")
    
    # 8. Recalculate Final Statistics (B=10,000)
    print("\n" + "=" * 80)
    print(f" FINAL STATISTICAL BOOTSTRAP (N={len(df_complete)})")
    print("=" * 80)
    
    mean_clean_f = df_complete["comet_clean_f"].mean()
    mean_clean_i = df_complete["comet_clean_i"].mean()
    mean_asr_f   = df_complete["comet_asr_f_e2e"].mean()
    mean_asr_i   = df_complete["comet_asr_i_e2e"].mean()
    mean_did     = df_complete["comet_did"].mean()
    
    print("Executing Paired Bootstrap Resampling (B=10,000)...")
    np.random.seed(42)
    did_array = df_complete["comet_did"].values
    boot_means = np.empty(10000)
    for b in tqdm(range(10000), desc="Bootstrapping"):
        sample = np.random.choice(did_array, size=len(df_complete), replace=True)
        boot_means[b] = np.mean(sample)
    
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    
    print("\n" + "=" * 80)
    print(" COMPLETE DATASET RESULTS (END-TO-END)")
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
        print("  Verdict : NULL RESULT (No proven differential semantic effect).")

if __name__ == "__main__":
    main()