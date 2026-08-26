#!/usr/bin/env python3
"""
Test A — Does ASR input push encoder activations out of their usual range?

ONNX Runtime dynamic quantization computes activation scales at inference
time from the observed tensor range. If ASR-generated German produces wider
activation ranges than clean text, the quantization grid is stretched, each
step is coarser, and quantization error rises. That would explain the
perturbation amplification observed in Step 36.

This measures the encoder activation distribution directly. No new model
export required.
"""
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

FP32_DIR = Path("models/onnx/opus_mt_de_en_opt_extended")
COHORT   = Path("analysis/scored_qirg_cohort_complete.parquet")
N        = 2000
SEED     = 20260818

tok = AutoTokenizer.from_pretrained(FP32_DIR)
m32 = ORTModelForSeq2SeqLM.from_pretrained(FP32_DIR, use_io_binding=False)

#df = pd.read_parquet(COHORT).sample(N, random_state=SEED).reset_index(drop=True)
df = pd.read_parquet(COHORT)
print(f"probing encoder activations on {len(df):,} sentences\n")

rows = []
for col, cond in [("gold_german_source", "clean"), ("whisper_hypothesis", "asr")]:
    for _, r in tqdm(df.iterrows(), total=len(df), desc=cond):
        t = r[col]
        if not isinstance(t, str) or not t.strip():
            continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            h = m32.encoder(**enc).last_hidden_state[0].numpy()
        a = np.abs(h)
        rows.append({
            "sample_id": r.get("sample_id"),
            "cond": cond,
            "absmax": float(a.max()),        # what dynamic scaling keys on
            "p999":   float(np.percentile(a, 99.9)),
            "p99":    float(np.percentile(a, 99)),
            "std":    float(h.std()),
            "n_tok":  int(h.shape[0]),
        })

res = pd.DataFrame(rows)
out = Path("analysis/encoder_activation_range.parquet")
res.to_parquet(out, index=False)

print("\n" + "=" * 70)
print(" ENCODER ACTIVATION RANGE BY INPUT CONDITION")
print("=" * 70)
print(res.groupby("cond")[["absmax", "p999", "p99", "std"]]
         .agg(["mean", "median"]).round(4).to_string())

# paired test on sentences present in both conditions
piv = res.pivot_table(index="sample_id", columns="cond", values="absmax").dropna()
print(f"\npaired on {len(piv):,} sentences")
print(f"mean absmax  clean {piv.clean.mean():.4f} -> asr {piv.asr.mean():.4f} "
      f"({100*(piv.asr.mean()/piv.clean.mean()-1):+.2f}%)")
print(f"ASR larger in {100*(piv.asr > piv.clean).mean():.1f}% of sentences")
u, p = mannwhitneyu(piv.asr, piv.clean, alternative="greater")
print(f"Mann-Whitney (asr > clean): p = {p:.3e}")

print("\n" + "-" * 70)
if p < 0.05 and piv.asr.mean() > piv.clean.mean():
    print("SUPPORTS the mechanism: ASR input widens activation range, so")
    print("dynamic scales are coarser and quantization error rises.")
    print("-> upgrade Step 36 from 'most plausible' to 'supported by direct")
    print("   measurement of encoder activation ranges'.")
else:
    print("DOES NOT SUPPORT the mechanism. Activation ranges are comparable,")
    print("so widened dynamic scales cannot explain the perturbation increase.")
    print("-> the explanation in Step 36 must be withdrawn or revised.")
print("-" * 70)
print(f"\nsaved -> {out}")