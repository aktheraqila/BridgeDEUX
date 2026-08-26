#!/usr/bin/env python3
"""Level 2 analysis: P(flip | margin), position-stratified."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

cond = sys.argv[1] if len(sys.argv) > 1 else "clean"
df = pd.read_parquet(f"analysis/level2_margin_tokens_{cond}.parquet")
print(f"{len(df):,} token positions from {df.sample_id.nunique():,} sentences")
print(f"overall flip rate: {100*df.is_flip.mean():.3f}%\n")

BINS = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
LBL  = ["0-.25", ".25-.5", ".5-1", "1-2", "2-4", "4-8", ">8"]
df["mbin"] = pd.cut(df.margin_fp32, BINS, labels=LBL, include_lowest=True)

print("=" * 72)
print(" P(FLIP | FP32 MARGIN)")
print("=" * 72)
for b in LBL:
    g = df[df.mbin == b]
    if len(g) == 0: continue
    lo, hi = proportion_confint(g.is_flip.sum(), len(g), method="wilson")
    print(f"{b:>8} | N={len(g):8,} | flip {100*g.is_flip.mean():6.2f}% "
          f"[{100*lo:5.2f}, {100*hi:5.2f}]")

print("\n" + "=" * 72)
print(" POSITION-STRATIFIED (rules out the position confound)")
print("=" * 72)
df["pband"] = pd.cut(df.rel_position, [0, .25, .5, .75, 1.0],
                     labels=["first25", "25-50", "50-75", "last25"],
                     include_lowest=True)
piv = df.pivot_table(index="mbin", columns="pband", values="is_flip",
                     aggfunc="mean", observed=True) * 100
print(piv.round(2).to_string())
print("\n-> if rows are flat across columns, margin (not position) drives flips")

print("\n" + "=" * 72)
print(" LOGISTIC MODEL (SE clustered on sentence)")
print("=" * 72)
s = df.sample(min(200_000, len(df)), random_state=0)
m = smf.logit("is_flip ~ margin_fp32 + rel_position", data=s).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": s.sample_id})
print(m.summary().tables[1])

print("\nperturbation vs margin at flip / non-flip:")
print(df.groupby("is_flip")[["margin_fp32", "mean_abs_pert"]].mean().round(4))