#!/usr/bin/env python3
"""
BridgeDEUX — Matched-Pair Test for Noise-Induced Metric Decoupling
==================================================================
Tests whether the Spearman drop (clean r_s=0.6803 -> ASR r_s=0.4843)
is a genuine decoupling or an artifact of variance attenuation.

The Level 1 bootstrap resampled the two conditions INDEPENDENTLY and
did not control for the fact that std(d_comet) is smaller under ASR
(0.252 vs 0.321). Correlation attenuates mechanically when variance
shrinks, so that test cannot distinguish decoupling from attenuation.

This script restricts to samples divergent in BOTH conditions, so the
same sentences are compared under both inputs, and uses a paired
bootstrap clustered on sample_id.

Run:  python experiments/03b_matched_decoupling_test.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PARQUET = Path("analysis/scored_qirg_cohort_complete.parquet")
N_BOOT = 2000
SEED = 20260818
EXPECTED_ROWS = 13511


# ----------------------------------------------------------------------
# Schema resolution — never guess column names silently
# ----------------------------------------------------------------------
def resolve(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    print(f"\n[FATAL] Could not find a column for: {label}")
    print(f"        Tried: {candidates}")
    print(f"        Available columns:\n        {list(df.columns)}")
    sys.exit(1)


def main():
    rng = np.random.default_rng(SEED)

    if not PARQUET.exists():
        sys.exit(f"[FATAL] Missing {PARQUET}")

    df = pd.read_parquet(PARQUET)
    print(f"Loaded {PARQUET}  ->  {df.shape[0]:,} rows x {df.shape[1]} cols")
    if len(df) != EXPECTED_ROWS:
        print(f"[WARN] Expected {EXPECTED_ROWS:,} rows (post-patch cohort), "
              f"got {len(df):,}. Check you are not on the pre-patch file.")

    ID = resolve(df, ["sample_id", "id", "utt_id", "utterance_id"], "sample id")

    C_FP32_TXT = resolve(df, ["clean_fp32_translation"], "clean FP32 text")
    C_INT8_TXT = resolve(df, ["clean_int8_translation"], "clean INT8 text")
    A_FP32_TXT = resolve(df, ["asr_fp32_translation"], "ASR FP32 text")
    A_INT8_TXT = resolve(df, ["asr_int8_translation"], "ASR INT8 text")

    C_CHRF_F = resolve(df, ["chrf_clean_f"], "clean chrF++ FP32")
    C_CHRF_I = resolve(df, ["chrf_clean_i"], "clean chrF++ INT8")
    C_COM_F  = resolve(df, ["comet_clean_f"], "clean COMET FP32")
    C_COM_I  = resolve(df, ["comet_clean_i"], "clean COMET INT8")

    A_CHRF_F = resolve(df, ["chrf_asr_f", "chrf_noisy_f"], "ASR chrF++ FP32")
    A_CHRF_I = resolve(df, ["chrf_asr_i", "chrf_noisy_i"], "ASR chrF++ INT8")
    A_COM_F  = resolve(df, ["comet_asr_f_mt", "comet_noisy_f"], "ASR COMET FP32")
    A_COM_I  = resolve(df, ["comet_asr_i_mt", "comet_noisy_i"], "ASR COMET INT8")

    # ------------------------------------------------------------------
    # Divergence flags + deltas  (INT8 - FP32, so negative = INT8 worse)
    # ------------------------------------------------------------------
    df["clean_div"] = df[C_FP32_TXT].str.strip() != df[C_INT8_TXT].str.strip()
    df["asr_div"]   = df[A_FP32_TXT].str.strip() != df[A_INT8_TXT].str.strip()

    df["c_dcomet"] = df[C_COM_I]  - df[C_COM_F]
    df["c_dchrf"]  = df[C_CHRF_I] - df[C_CHRF_F]
    df["a_dcomet"] = df[A_COM_I]  - df[A_COM_F]
    df["a_dchrf"]  = df[A_CHRF_I] - df[A_CHRF_F]

    clean = df[df.clean_div].dropna(subset=["c_dchrf", "c_dcomet"])
    asr   = df[df.asr_div].dropna(subset=["a_dchrf", "a_dcomet"])

    print("\n" + "=" * 78)
    print(" 0. UNMATCHED BASELINE (reproduces Level 1)")
    print("=" * 78)
    r_c_all = spearmanr(clean.c_dchrf, clean.c_dcomet)[0]
    r_a_all = spearmanr(asr.a_dchrf, asr.a_dcomet)[0]
    print(f"Clean  n={len(clean):5d}  r_s = {r_c_all:.4f}")
    print(f"ASR    n={len(asr):5d}  r_s = {r_a_all:.4f}")
    print(f"Unmatched drop = {r_c_all - r_a_all:.4f}")

    # ------------------------------------------------------------------
    # 1. Bidirectionality — the signed split (was missing from Level 1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" 1. SIGNED dCOMET SPLIT  (tests the 'zero-sum' claim)")
    print("=" * 78)
    for name, sub, col in [("Clean", clean, "c_dcomet"), ("ASR", asr, "a_dcomet")]:
        v = sub[col]
        pos, neg = (v > 0).sum(), (v < 0).sum()
        print(f"{name:6s} n={len(v):5d} | INT8 better: {pos:5d} ({100*pos/len(v):5.2f}%) "
              f"| FP32 better: {neg:5d} ({100*neg/len(v):5.2f}%)")
        print(f"       signed mean = {v.mean():+.5f} | "
              f"mean|d| = {v.abs().mean():.5f} | std = {v.std():.5f}")
        # binomial check on directional balance
        from scipy.stats import binomtest
        p = binomtest(int(pos), int(pos + neg), 0.5).pvalue
        verdict = "balanced" if p > 0.05 else "IMBALANCED"
        print(f"       binomial p = {p:.4g}  ->  {verdict}")

    # ------------------------------------------------------------------
    # 2. Matched set — same sentences, both conditions
    # ------------------------------------------------------------------
    both = sorted(set(clean[ID]) & set(asr[ID]))
    if len(both) < 200:
        sys.exit(f"[FATAL] Only {len(both)} matched samples — too few to test.")

    c = clean[clean[ID].isin(both)].set_index(ID).sort_index()
    a = asr[asr[ID].isin(both)].set_index(ID).sort_index()
    assert (c.index == a.index).all(), "matched index misalignment"

    print("\n" + "=" * 78)
    print(f" 2. MATCHED SET  (divergent in BOTH conditions, n = {len(both):,})")
    print("=" * 78)

    vr_com  = a.a_dcomet.var() / c.c_dcomet.var()
    vr_chrf = a.a_dchrf.var()  / c.c_dchrf.var()
    print(f"std(dCOMET)  clean {c.c_dcomet.std():.4f} -> ASR {a.a_dcomet.std():.4f}"
          f"   (var ratio {vr_com:.3f})")
    print(f"std(dchrF++) clean {c.c_dchrf.std():.4f} -> ASR {a.a_dchrf.std():.4f}"
          f"   (var ratio {vr_chrf:.3f})")
    print("  -> ratios far from 1.0 mean attenuation is a live confound.")

    r_c = spearmanr(c.c_dchrf, c.c_dcomet)[0]
    r_a = spearmanr(a.a_dchrf, a.a_dcomet)[0]
    print(f"\nMatched clean r_s = {r_c:.4f}")
    print(f"Matched ASR   r_s = {r_a:.4f}")
    print(f"Matched drop      = {r_c - r_a:.4f}")

    # ------------------------------------------------------------------
    # 3. Paired bootstrap, clustered on sample_id
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f" 3. PAIRED BOOTSTRAP  ({N_BOOT} iters, clustered on {ID})")
    print("=" * 78)
    ids = np.array(both)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        cc, aa = c.loc[pick], a.loc[pick]
        diffs[i] = (spearmanr(cc.c_dchrf, cc.c_dcomet)[0]
                    - spearmanr(aa.a_dchrf, aa.a_dcomet)[0])

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"Delta r_s = {diffs.mean():.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n" + "-" * 78)
    if lo > 0:
        print("VERDICT: Drop SURVIVES matched-pair control (CI excludes zero).")
        if not (0.7 < vr_com < 1.4):
            print("         BUT variance ratio is far from 1.0 — attenuation is")
            print("         NOT excluded. Report both the CI and the variance")
            print("         ratios, and state attenuation as a limitation.")
        else:
            print("         Variances are comparable — attenuation is unlikely")
            print("         to explain the drop. This is a defensible finding.")
    else:
        print("VERDICT: Drop DOES NOT SURVIVE. The Level 1 result was driven by")
        print("         population differences between conditions, not decoupling.")
        print("         Do not report metric decoupling as a finding.")
    print("-" * 78)

    out = Path("analysis/matched_decoupling_test.csv")
    pd.DataFrame({"boot_delta_rs": diffs}).to_csv(out, index=False)
    print(f"\nSaved bootstrap draws -> {out}")


if __name__ == "__main__":
    main()