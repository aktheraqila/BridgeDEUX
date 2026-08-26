#!/usr/bin/env python3
"""
BridgeDEUX: Level 1 Quantization Divergence Analysis (Clean & ASR)
=================================================================
Evaluates semantic vs. surface divergence distributions and Spearman decoupling.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint

def bootstrap_corr_diff(df_clean, df_asr, n_boot=2000):
    """Bootstraps the 95% CI for the difference between two Spearman correlations."""
    diffs = []
    # Convert to numpy arrays for much faster indexing during bootstrap
    clean_chrf = df_clean["d_chrf"].to_numpy()
    clean_comet = df_clean["d_comet"].to_numpy()
    asr_chrf = df_asr["d_chrf"].to_numpy()
    asr_comet = df_asr["d_comet"].to_numpy()
    
    n_clean = len(clean_chrf)
    n_asr = len(asr_chrf)

    for _ in range(n_boot):
        idx_clean = np.random.choice(n_clean, n_clean, replace=True)
        idx_asr = np.random.choice(n_asr, n_asr, replace=True)
        
        r_clean = spearmanr(clean_chrf[idx_clean], clean_comet[idx_clean])[0]
        r_asr = spearmanr(asr_chrf[idx_asr], asr_comet[idx_asr])[0]
        diffs.append(r_clean - r_asr)
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    return np.mean(diffs), ci_lower, ci_upper

def analyze_condition(
    df: pd.DataFrame,
    condition: str,
    fp32_col: str,
    int8_col: str,
    chrf_f: str,
    chrf_i: str,
    comet_f: str,
    comet_i: str,
):
    print("=" * 80)
    print(f" CONDITION: {condition.upper()}")
    print("=" * 80)

    div_mask = df[fp32_col].str.strip() != df[int8_col].str.strip()
    total = len(df)
    n_div = div_mask.sum()
    lo, hi = proportion_confint(n_div, total, method="wilson")

    print(f"Total Cohort Size    : {total:,}")
    print(
        f"Divergent Outputs    : {n_div:,} ({100 * n_div / total:.2f}%) [95% CI: {100*lo:.2f}% - {100*hi:.2f}%]\n"
    )

    d = df[div_mask].copy()
    d["d_comet"] = d[comet_i] - d[comet_f]
    d["d_chrf"] = d[chrf_i] - d[chrf_f]

    print("--- |ΔCOMET| Distribution on Divergent Pairs ---")
    print(d["d_comet"].abs().describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]))
    print()

    print("--- Threshold Sweep ---")
    for t in [0.01, 0.02, 0.05, 0.10, 0.20]:
        sem = (d["d_comet"].abs() > t).mean()
        print(
            f"|ΔCOMET| > {t:.2f} : {100 * sem:5.2f}% Semantic Shift | {100 * (1 - sem):5.2f}% Surface Invariant"
        )

    corr, pval = spearmanr(d["d_chrf"], d["d_comet"])
    print(
        f"\nSpearman Correlation (ΔchrF++ vs. ΔCOMET on flips): r_s = {corr:.4f} (p = {pval:.4e})\n"
    )
    
    # Return the divergent dataframe so we can compare them later
    return d


def main():
    target_file = Path("analysis/scored_qirg_cohort_complete.parquet")
    assert target_file.exists(), f"Target file missing: {target_file}"

    df = pd.read_parquet(target_file)
    assert len(df) == 13511, f"Expected 13,511 rows, but got {len(df)}. You are loading the wrong cohort."
    print(f"Loaded dataset: {target_file} ({len(df):,} rows)")

    # 1. Clean Condition Sweep (capture the returned dataframe)
    div_clean = analyze_condition(
        df,
        "Clean Input (Gold German)",
        "clean_fp32_translation",
        "clean_int8_translation",
        "chrf_clean_f",
        "chrf_clean_i",
        "comet_clean_f",
        "comet_clean_i",
    )

    # 2. ASR Condition Sweep (capture the returned dataframe)
    div_asr = analyze_condition(
        df,
        "ASR Input (Whisper German)",
        "asr_fp32_translation",
        "asr_int8_translation",
        "chrf_asr_f",
        "chrf_asr_i",
        "comet_asr_f_mt",
        "comet_asr_i_mt",
    )

    # 3. WER Tier Stratification with Wilson CIs
    print("=" * 80)
    print(" CASCADE INTERACTION: ASR WER TIER DIVERGENCE (Wilson 95% CI)")
    print("=" * 80)
    df["asr_divergent"] = (
        df["asr_fp32_translation"].str.strip()
        != df["asr_int8_translation"].str.strip()
    )
    for bin_name, group in df.groupby("wer_bin", observed=True):
        count = len(group)
        divs = group["asr_divergent"].sum()
        lo, hi = proportion_confint(divs, count, method="wilson")
        print(
            f"WER Tier: {str(bin_name):<20} | N={count:5d} | Flip Rate: {100 * divs / count:5.2f}% (95% CI: [{100 * lo:5.2f}%, {100 * hi:5.2f}%])"
        )
        
    # 4. Bootstrap Correlation Difference
    print("\n" + "=" * 80)
    print(" STRUCTURAL DECOUPLING STATISTICAL TEST (Bootstrapped Δr_s)")
    print("=" * 80)
    print("Bootstrapping 2000 iterations to verify the correlation drop...")
    mean_diff, lo, hi = bootstrap_corr_diff(div_clean, div_asr)
    print(f"Correlation Drop (Δr_s) from Clean to ASR: {mean_diff:.4f} [95% CI: {lo:.4f} to {hi:.4f}]")
    if lo > 0:
        print("Result: SIGNIFICANT. The drop in correlation is structurally robust and not a sample size artifact.")
    else:
        print("Result: NOT SIGNIFICANT. The confidence interval crosses zero.")
    print("=" * 80)

if __name__ == "__main__":
    main()