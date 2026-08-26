#!/usr/bin/env python3
"""
BridgeDEUX: Step 5E - Framework-Native COMET Evaluator
======================================================
Fault-tolerant COMET inference using the existing BridgeDEUX CheckpointManager.
Safe against shutdowns. Re-run to resume seamlessly.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from comet import download_model, load_from_checkpoint
import logging
import warnings

# Suppress verbose lightning tips
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from benchmarks.checkpoint_manager import CheckpointManager

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    cohort_path = analysis_dir / "scored_qirg_cohort.parquet"
    
    if not cohort_path.exists():
        print("ERROR: Run 05c_evaluate_qirg_did.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(cohort_path)
    N_TOTAL = len(df)
    
    print("=" * 80)
    print(f" STEP 5E: RESUMABLE COMET EVALUATION (Total N={N_TOTAL})")
    print("=" * 80)
    
    # 1. Initialize CheckpointManager using the exact project API
    model_id = "comet_wmt20_qirg"
    try:
        manager = CheckpointManager(
            model_identifier=model_id,
            checkpoint_interval=25,
            output_dir=analysis_dir
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize CheckpointManager: {e}")
        sys.exit(1)
        
    completed_ids = manager.load_completed_samples()
    n_completed = len(completed_ids)
    n_remaining = N_TOTAL - n_completed
    
    print("WAL VERIFICATION (via CheckpointManager):")
    print(f"  Total Cohort Size : {N_TOTAL}")
    print(f"  Completed Samples : {n_completed} ({n_completed / N_TOTAL * 100:.2f}%)")
    print(f"  Remaining Samples : {n_remaining}")
    print("-" * 80)
    
    # 2. Inference Loop
    if n_remaining > 0:
        df_todo = df[~df["sample_id"].isin(completed_ids)].copy()
        
        comet_model_name = "Unbabel/wmt20-comet-da"
        print(f"Loading '{comet_model_name}' checkpoint...")
        try:
            model_path = download_model(comet_model_name)
            model = load_from_checkpoint(model_path)
            model.eval()
        except Exception as e:
            print(f"FATAL: Failed to load COMET model: {e}")
            sys.exit(1)
            
        BATCH_SIZE = 16
        print(f"\nResuming inference for {n_remaining} samples...")
        
        for i in range(0, n_remaining, BATCH_SIZE):
            batch = df_todo.iloc[i : i + BATCH_SIZE]
            
            gold_src = batch["gold_german_source"].tolist()
            asr_src  = batch["whisper_hypothesis"].tolist()
            refs     = batch["gold_english_reference"].tolist()
            
            clean_f = batch["clean_fp32_translation"].tolist()
            clean_i = batch["clean_int8_translation"].tolist()
            asr_f   = batch["asr_fp32_translation"].tolist()
            asr_i   = batch["asr_int8_translation"].tolist()
            
            passes = {
                "comet_clean_f": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, clean_f, refs)],
                "comet_clean_i": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, clean_i, refs)],
                "comet_asr_f_e2e": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, asr_f, refs)],
                "comet_asr_i_e2e": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(gold_src, asr_i, refs)],
                "comet_asr_f_mt": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(asr_src, asr_f, refs)],
                "comet_asr_i_mt": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(asr_src, asr_i, refs)],
            }
            
            batch_results = {"sample_id": batch["sample_id"].tolist()}
            for col_name, data in passes.items():
                out = model.predict(data, batch_size=BATCH_SIZE, gpus=0)
                batch_results[col_name] = out.scores
                
            # Feed individual records to the CheckpointManager
            for j in range(len(batch)):
                record = {
                    "sample_id": batch_results["sample_id"][j],
                    "comet_clean_f": batch_results["comet_clean_f"][j],
                    "comet_clean_i": batch_results["comet_clean_i"][j],
                    "comet_asr_f_e2e": batch_results["comet_asr_f_e2e"][j],
                    "comet_asr_i_e2e": batch_results["comet_asr_i_e2e"][j],
                    "comet_asr_f_mt": batch_results["comet_asr_f_mt"][j],
                    "comet_asr_i_mt": batch_results["comet_asr_i_mt"][j],
                }
                manager.save(record)
                
            processed_so_far = n_completed + i + len(batch)
            if processed_so_far % 100 < BATCH_SIZE or processed_so_far == N_TOTAL:
                pct = processed_so_far / N_TOTAL * 100
                print(f"  Progress: {processed_so_far:,}/{N_TOTAL:,} ({pct:5.1f}%) processed.")
            
        # Ensure any remaining samples in the manager's buffer are flushed
        if manager.has_pending_records:
            manager.flush()
        manager.finalize()

    # ---------------------------------------------------------
    # 3. Final Assembly & Statistical Bootstrap
    # ---------------------------------------------------------
    # Re-verify completeness
    completed_ids = manager.load_completed_samples()
    if len(completed_ids) < N_TOTAL:
        print(f"\n[!] Pausing: Completed {len(completed_ids)}/{N_TOTAL}. Run script again to finish.")
        sys.exit(0)
        
    print("\n" + "=" * 80)
    print(" ALL SAMPLES PROCESSED: COMPILING FINAL DATASET & STATISTICS")
    print("=" * 80)
    
    # Load the Parquet artifact generated by CheckpointManager
    target_parquet = analysis_dir / f"{model_id}_results.parquet"
    if not target_parquet.exists():
        print(f"FATAL: Could not locate the finalized CheckpointManager Parquet artifact at {target_parquet}")
        sys.exit(1)
        
    print(f"Loading COMET results from CheckpointManager artifact: {target_parquet.name}")
    df_scores = pd.read_parquet(target_parquet)
    df_final = pd.merge(df, df_scores, on="sample_id", how="inner")
    
    # Calculate End-to-End Semantic DiD
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