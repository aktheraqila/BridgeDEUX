#!/usr/bin/env python3
"""
Level 2 comparison: is the flip mechanism invariant across input conditions?

Two competing explanations for the WER gradient (30.55% -> 45.15%):
  H1 (mechanism invariant): the P(flip|margin) curve is the SAME, but ASR
      input shifts the margin DISTRIBUTION toward zero. More positions land
      in the danger zone; the danger zone itself is unchanged.
  H2 (mechanism changes): ASR input makes flips more likely even at equal
      margin, i.e. the curve itself moves.
"""
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import mannwhitneyu

BINS = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
LBL  = ["0-.25", ".25-.5", ".5-1", "1-2", "2-4", "4-8", ">8"]

d = {}
for c in ["clean", "asr"]:
    p = Path(f"analysis/level2_margin_tokens_{c}.parquet")
    if not p.exists():
        raise SystemExit(f"missing {p} — run the probe for --condition {c}")
    x = pd.read_parquet(p)
    x["mbin"] = pd.cut(x.margin_fp32, BINS, labels=LBL, include_lowest=True)
    d[c] = x
    print(f"{c:5s}: {len(x):7,} positions | {x.sample_id.nunique():,} sentences "
          f"| overall flip {100*x.is_flip.mean():.3f}%")

# ---------------------------------------------------------------- H1 part A
print("\n" + "=" * 78)
print(" A. MARGIN DISTRIBUTION — does ASR input compress margins?")
print("=" * 78)
for c in ["clean", "asr"]:
    m = d[c].margin_fp32
    print(f"{c:5s} | mean {m.mean():6.3f} | median {m.median():6.3f} | "
          f"P10 {m.quantile(.10):6.3f} | P25 {m.quantile(.25):6.3f} | "
          f"%<0.5 {100*(m<0.5).mean():5.2f}%")

u, p = mannwhitneyu(d["clean"].margin_fp32, d["asr"].margin_fp32,
                    alternative="greater")
print(f"\nMann-Whitney (clean margins > asr margins): p = {p:.3e}")
shift = 100*(d['asr'].margin_fp32 < 0.5).mean() - 100*(d['clean'].margin_fp32 < 0.5).mean()
print(f"Change in share of positions below margin 0.5: {shift:+.2f} pp")

# ---------------------------------------------------------------- H1 part B
print("\n" + "=" * 78)
print(" B. P(FLIP | MARGIN) — is the curve itself unchanged?")
print("=" * 78)
print(f"{'bin':>8} | {'clean':>22} | {'asr':>22} | overlap?")
print("-" * 78)
for b in LBL:
    gc, ga = d["clean"][d["clean"].mbin == b], d["asr"][d["asr"].mbin == b]
    if len(gc) < 30 or len(ga) < 30:
        continue
    lc, hc = proportion_confint(gc.is_flip.sum(), len(gc), method="wilson")
    la, ha = proportion_confint(ga.is_flip.sum(), len(ga), method="wilson")
    ov = "YES" if (lc <= ha and la <= hc) else "NO  <-- curve differs"
    print(f"{b:>8} | {100*gc.is_flip.mean():5.2f}% [{100*lc:5.2f},{100*hc:5.2f}] "
          f"N={len(gc):6,} | {100*ga.is_flip.mean():5.2f}% [{100*la:5.2f},{100*ha:5.2f}] "
          f"N={len(ga):6,} | {ov}")

# ---------------------------------------------------------------- decomposition
print("\n" + "=" * 78)
print(" C. DECOMPOSITION — how much of the flip-rate rise is distribution shift?")
print("=" * 78)
cw = d["clean"].mbin.value_counts(normalize=True)
aw = d["asr"].mbin.value_counts(normalize=True)
rc = d["clean"].groupby("mbin", observed=True).is_flip.mean()
ra = d["asr"].groupby("mbin", observed=True).is_flip.mean()

obs_c, obs_a = d["clean"].is_flip.mean(), d["asr"].is_flip.mean()
# ASR margin distribution, but CLEAN per-bin flip rates
counter = sum(aw.get(b, 0) * rc.get(b, 0) for b in LBL)

print(f"observed clean flip rate       : {100*obs_c:.3f}%")
print(f"observed ASR   flip rate       : {100*obs_a:.3f}%")
print(f"counterfactual (ASR margins,   ")
print(f"  clean per-bin flip rates)    : {100*counter:.3f}%")
if obs_a != obs_c:
    expl = 100 * (counter - obs_c) / (obs_a - obs_c)
    print(f"\n-> margin-distribution shift explains {expl:.1f}% of the increase")
    print(f"-> remaining {100-expl:.1f}% would be curve change")