#!/usr/bin/env python3
"""
BridgeDEUX: Step 2 - COMET Evaluation of Divergent Outputs (Q1/Q2 Standard)
===========================================================================
- Loads full valid sample set and strictly verifies 1-to-1 record alignment.
- Verifies that FP32 and INT8 saw the EXACT same source and reference texts.
- Filters by strict string inequality (fp32_translation != int8_translation).
- Evaluates divergent pairs using wmt20-comet-da as an independent neural metric.
- Computes Spearman rank correlation with non-parametric bootstrap 95% CI.
- Performs threshold sensitivity analysis for win/tie classification.
- Saves full row-level scores and reproducible provenance metadata.
"""

import os
import sys
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tabulate import tabulate

warnings.filterwarnings("ignore", category=UserWarning)

def find_latest_file(pattern: str) -> Path:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])

def load_jsonl_bak_records(file_path: Path, target_ids: set) -> dict:
    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sid = data.get("sample_id")
                if sid and sid in target_ids:
                    records[sid] = data
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at {file_path.name}:{line_num}: {e}")
                continue
    return records

def compute_spearman_bootstrap(x: pd.Series, y: pd.Series, n_resamples: int = 1000, ci: float = 0.95):
    rng = np.random.default_rng(42)
    indices = np.arange(len(x))
    rhos = []

    x_arr = x.values
    y_arr = y.values

    for _ in range(n_resamples):
        sample_idx = rng.choice(indices, size=len(x), replace=True)
        r, _ = stats.spearmanr(x_arr[sample_idx], y_arr[sample_idx])
        rhos.append(r)

    alpha = 1.0 - ci
    lower = np.percentile(rhos, (alpha / 2.0) * 100)
    upper = np.percentile(rhos, (1.0 - alpha / 2.0) * 100)
    return lower, upper

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    results_dir = repo_root / "results"

    try:
        parquet_file = find_latest_file(str(analysis_dir / "int8_seq_likelihood_exploratory_*.parquet"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    df = pd.read_parquet(parquet_file)
    valid_df = df[df["generation_status"] == "Generated"].copy()
    total_generated = len(valid_df)
    valid_ids = set(valid_df["sample_id"])

    try:
        fp32_file = find_latest_file(str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "*.jsonl.bak"))
        int8_file = find_latest_file(str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "*.jsonl.bak"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 75)
    print(" STEP 2: COMET EVALUATION (Q1/Q2 PROVENANCE & STRICT VALIDATION)")
    print("=" * 75)
    print(f"Total Valid Parquet Records: {total_generated:,}")
    print(f"FP32 Log Source            : {fp32_file.name}")
    print(f"INT8 Log Source            : {int8_file.name}")
    print("-" * 75)

    print("Loading text records...")
    fp32_data = load_jsonl_bak_records(fp32_file, valid_ids)
    int8_data = load_jsonl_bak_records(int8_file, valid_ids)

    missing_fp32 = valid_ids - set(fp32_data.keys())
    missing_int8 = valid_ids - set(int8_data.keys())

    if missing_fp32 or missing_int8:
        raise ValueError(
            f"Integrity check failed: Missing {len(missing_fp32)} IDs in FP32 log, "
            f"and {len(missing_int8)} IDs in INT8 log."
        )

    print("Validating source/reference text alignment...")
    text_data = []
    mismatched_inputs = []
    
    for sid in valid_ids:
        f_rec = fp32_data[sid]
        i_rec = int8_data[sid]
        
        f_src = f_rec.get("source_text", "")
        i_src = i_rec.get("source_text", "")
        f_ref = f_rec.get("reference_translation", "")
        i_ref = i_rec.get("reference_translation", "")
        
        # STRICT VALIDATION CHECK
        if f_src != i_src or f_ref != i_ref:
            mismatched_inputs.append(sid)
            
        text_data.append({
            "sample_id": sid,
            "source": f_src,
            "reference": f_ref,
            "fp32_translation": f_rec.get("translation", ""),
            "int8_translation": i_rec.get("translation", ""),
        })

    if mismatched_inputs:
        raise ValueError(
            f"Integrity check failed: Found {len(mismatched_inputs)} samples where "
            f"the source text or reference differs between the FP32 and INT8 logs. "
            f"First mismatched ID: {mismatched_inputs[0]}"
        )
    print("Data integrity passed: Source and reference texts align perfectly.")

    text_df = pd.DataFrame(text_data)
    valid_df = valid_df.merge(text_df, on="sample_id")

    # Strict divergence isolation
    divergent_df = valid_df[valid_df["fp32_translation"] != valid_df["int8_translation"]].copy()
    n_divergent = len(divergent_df)
    n_identical = total_generated - n_divergent

    print(f"Identical Text Outputs     : {n_identical:,} (ΔCOMET = 0 by definition)")
    print(f"Divergent Outputs to Score : {n_divergent:,}")
    print("-" * 75)

    if n_divergent == 0:
        print("All generated translations are identical. Nothing to evaluate.")
        return

    fp32_comet_input = []
    int8_comet_input = []

    for _, row in divergent_df.iterrows():
        fp32_comet_input.append({"src": row["source"], "mt": row["fp32_translation"], "ref": row["reference"]})
        int8_comet_input.append({"src": row["source"], "mt": row["int8_translation"], "ref": row["reference"]})

    from comet import download_model, load_from_checkpoint
    import comet

    print("\nDownloading/Loading wmt20-comet-da checkpoint...")
    model_name = "Unbabel/wmt20-comet-da"
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    has_gpu = torch.cuda.is_available()
    batch_size = 16
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"

    print(f"Running inference on {device_name} (batch_size={batch_size})...")
    print(f"Scoring FP32 translations ({len(fp32_comet_input):,} samples)...")
    fp32_pred = model.predict(fp32_comet_input, batch_size=batch_size, gpus=1 if has_gpu else 0)

    print(f"Scoring INT8 translations ({len(int8_comet_input):,} samples)...")
    int8_pred = model.predict(int8_comet_input, batch_size=batch_size, gpus=1 if has_gpu else 0)

    comet_scores = pd.DataFrame({
        "sample_id": divergent_df["sample_id"],
        "fp32_comet": fp32_pred.scores,
        "int8_comet": int8_pred.scores,
    })

    comet_scores["delta_comet"] = comet_scores["fp32_comet"] - comet_scores["int8_comet"]
    divergent_df = divergent_df.merge(comet_scores, on="sample_id")

    mean_chrf_delta = divergent_df["delta_chrf"].mean()
    mean_comet_delta = divergent_df["delta_comet"].mean()

    rho, p_val = stats.spearmanr(divergent_df["delta_chrf"], divergent_df["delta_comet"])
    print("Computing Non-Parametric Bootstrap CI for Spearman ρ (1,000 resamples)...")
    ci_lower, ci_upper = compute_spearman_bootstrap(divergent_df["delta_chrf"], divergent_df["delta_comet"])

    print("\n" + "=" * 75)
    print(" STEP 2: COMET EVALUATION RESULTS")
    print("=" * 75)

    print("[1. Metric Agreement on Divergent Subset]")
    print(f"Mean ΔchrF++         : {mean_chrf_delta:.4f}")
    print(f"Mean ΔCOMET          : {mean_comet_delta:.4f}")
    print(f"Spearman ρ           : {rho:.4f}")
    print(f"95% Bootstrap CI     : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"p-value              : {p_val:.2e}")

    print("\n[2. Win/Loss Comparison Across Thresholds (Sensitivity Analysis)]")
    thresholds = [0.0, 0.01, 0.02, 0.05]
    win_data = []

    for t in thresholds:
        fp32_w = len(divergent_df[divergent_df["delta_comet"] > t])
        int8_w = len(divergent_df[divergent_df["delta_comet"] < -t])
        ties = len(divergent_df[divergent_df["delta_comet"].abs() <= t])
        win_data.append([f"|Δ| > {t}", f"{fp32_w:,}", f"{int8_w:,}", f"{ties:,}"])

    print(tabulate(win_data, headers=["Threshold Condition", "FP32 Wins", "INT8 Wins", "Ties / Indeterminate"], tablefmt="fancy_grid"))

    print("\n[3. COMET Difference by Output Length Groups]")
    divergent_df["length_divergence"] = "Similar Length (|Δ| <= 2 tokens)"
    divergent_df.loc[divergent_df["delta_output_tokens"] > 2, "length_divergence"] = "FP32 Longer (Δ > 2 tokens)"
    divergent_df.loc[divergent_df["delta_output_tokens"] < -2, "length_divergence"] = "INT8 Longer (Δ < -2 tokens)"

    group_means = divergent_df.groupby("length_divergence", observed=False).agg(
        Count=("sample_id", "count"),
        Mean_chrF_Delta=("delta_chrf", "mean"),
        Mean_COMET_Delta=("delta_comet", "mean")
    ).reset_index()

    print(tabulate(group_means, headers=["Length Group", "N", "Mean chrF++ Δ", "Mean COMET Δ"], tablefmt="fancy_grid", showindex=False))
    print("=" * 75)

    run_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_parquet = analysis_dir / f"comet_divergent_scores_{run_timestamp}.parquet"

    divergent_df["fp32_output_words"] = divergent_df["fp32_translation"].apply(lambda x: len(x.split()))
    divergent_df["int8_output_words"] = divergent_df["int8_translation"].apply(lambda x: len(x.split()))

    cols_to_save = [
        "sample_id", "source", "reference", "fp32_translation", "int8_translation",
        "fp32_chrf", "int8_chrf", "delta_chrf",
        "fp32_comet", "int8_comet", "delta_comet",
        "fp32_output_words", "int8_output_words", "delta_output_tokens"
    ]
    divergent_df[cols_to_save].to_parquet(out_parquet, index=False)

    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "experiment_step": "Step 2 - COMET Neural Metric Evaluation",
        "comet_model": model_name,
        "comet_version": comet.__version__,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "device": device_name,
        "batch_size": batch_size,
        "random_seed": 42,
        "total_valid_samples": total_generated,
        "exact_string_matches": n_identical,
        "divergent_scored": n_divergent,
        "mean_chrf_delta_divergent": float(mean_chrf_delta),
        "mean_comet_delta_divergent": float(mean_comet_delta),
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_val),
        "spearman_bootstrap_95_ci": [float(ci_lower), float(ci_upper)],
        "source_parquet": parquet_file.name,
        "fp32_jsonl_source": fp32_file.name,
        "int8_jsonl_source": int8_file.name,
    }

    out_meta = analysis_dir / f"comet_divergent_scores_metadata_{run_timestamp}.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nArtifacts successfully written:")
    print(f"  - Dataset:  {out_parquet.name}")
    print(f"  - Metadata: {out_meta.name}\n")

if __name__ == "__main__":
    main()