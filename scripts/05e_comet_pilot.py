#!/usr/bin/env python3
"""
BridgeDEUX: Step 5E - COMET Runtime & Validity Pilot (100 Samples)
==================================================================
Runs a 100-sample deterministic benchmark using 'Unbabel/wmt20-comet-da' to:
1. Verify COMET model inference on CPU.
2. Verify sentence-level score extraction (.scores).
3. Confirm absence of NaNs / Infs.
4. Measure precise wall-clock time to estimate full-corpus runtime.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from comet import download_model, load_from_checkpoint
import warnings

warnings.filterwarnings("ignore")

def validate_scores(scores: list, name: str, expected_n: int):
    arr = np.array(scores)
    assert len(arr) == expected_n, f"{name}: Expected {expected_n} scores, got {len(arr)}"
    assert not np.isnan(arr).any(), f"{name}: Contains NaN values!"
    assert not np.isinf(arr).any(), f"{name}: Contains Inf values!"
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    cohort_path = analysis_dir / "scored_qirg_cohort.parquet"
    
    if not cohort_path.exists():
        print("ERROR: Run 05c_evaluate_qirg_did.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(cohort_path)
    N_TOTAL = len(df)
    
    # 1. Simple, reproducible deterministic sample
    N_PILOT = 100
    df_pilot = df.sample(n=N_PILOT, random_state=42).copy()
    
    print("=" * 80)
    print(f" STEP 5E: COMET PILOT BENCHMARK (N={N_PILOT} Deterministic Samples)")
    print("=" * 80)
    
    comet_model_name = "Unbabel/wmt20-comet-da"
    print(f"Loading '{comet_model_name}' checkpoint...")
    load_start = time.time()
    try:
        model_path = download_model(comet_model_name)
        model = load_from_checkpoint(model_path)
        model.eval()
    except Exception as e:
        print(f"FATAL: Failed to load COMET model: {e}")
        sys.exit(1)
        
    print(f"Model loaded in {time.time() - load_start:.2f} seconds.")
    
    gold_src = df_pilot["gold_german_source"].tolist()
    asr_src  = df_pilot["whisper_hypothesis"].tolist()
    refs     = df_pilot["gold_english_reference"].tolist()
    
    clean_f = df_pilot["clean_fp32_translation"].tolist()
    clean_i = df_pilot["clean_int8_translation"].tolist()
    asr_f   = df_pilot["asr_fp32_translation"].tolist()
    asr_i   = df_pilot["asr_int8_translation"].tolist()
    
    passes = {
        "1. Clean FP32 (E2E)": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, clean_f, refs)],
        "2. Clean INT8 (E2E)": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, clean_i, refs)],
        "3. ASR FP32 (E2E)  ": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, asr_f, refs)],
        "4. ASR INT8 (E2E)  ": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, asr_i, refs)],
        "5. ASR FP32 (MT-In)": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(asr_src, asr_f, refs)],
        "6. ASR INT8 (MT-In)": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(asr_src, asr_i, refs)],
    }
    
    BATCH_SIZE = 16
    results = {}
    
    print(f"\nRunning Inference Across All 6 Passes (Batch Size = {BATCH_SIZE}, CPU Mode)...")
    total_infer_start = time.time()
    
    for name, data in passes.items():
        t0 = time.time()
        out = model.predict(data, batch_size=BATCH_SIZE, gpus=0)
        dt = time.time() - t0
        
        mean_s, min_s, max_s = validate_scores(out.scores, name, N_PILOT)
        results[name] = out.scores
        print(f"  {name} -> Done in {dt:5.2f}s | Mean: {mean_s:+.4f} [Min: {min_s:+.4f}, Max: {max_s:+.4f}]")
        
    total_infer_time = time.time() - total_infer_start
    rate_evals_per_sec = (N_PILOT * 6) / total_infer_time
    
    est_sec_6passes = (N_TOTAL * 6) / rate_evals_per_sec
    est_sec_4passes = (N_TOTAL * 4) / rate_evals_per_sec
    
    print("\n" + "=" * 80)
    print(" PILOT VERIFICATION & RUNTIME PROJECTIONS")
    print("=" * 80)
    print(f"Total Pilot Inference Time (600 evals) : {total_infer_time:.2f} seconds")
    print(f"Processing Throughput                  : {rate_evals_per_sec:.2f} evaluations/sec")
    print("-" * 80)
    print(f"Estimated Full 13,411 (All 6 Passes)   : {est_sec_6passes / 60:.1f} minutes ({est_sec_6passes / 3600:.2f} hours)")
    print(f"Estimated Full 13,411 (Core 4 E2E Only): {est_sec_4passes / 60:.1f} minutes ({est_sec_4passes / 3600:.2f} hours)")
    print("=" * 80)

if __name__ == "__main__":
    main()