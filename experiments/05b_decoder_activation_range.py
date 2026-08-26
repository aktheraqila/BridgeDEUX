#!/usr/bin/env python3
"""
Test A2 — Decoder-side activation ranges under clean vs ASR input.

Step 35 measured perturbation in DECODER logits, but 05_ measured ENCODER
activations. This checks the tensors that actually feed the quantized
decoder MatMuls.

Verdict requires BOTH significance AND an effect large enough to plausibly
explain the observed 5-12% perturbation increase. A p-value alone is not
evidence at n=13,511.

Resumable: every sentence is flushed to JSONL immediately, so a power cut
costs at most the sentence in flight. Re-run the same command to resume.

  python experiments/05b_decoder_activation_range.py
  python experiments/05b_decoder_activation_range.py --limit 2000
"""
import argparse, json
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

FP32_DIR = Path("models/onnx/opus_mt_de_en_opt_extended")
COHORT   = Path("analysis/scored_qirg_cohort_complete.parquet")
CKPT     = Path("analysis/decoder_activation_ckpt.jsonl")
OUT      = Path("analysis/decoder_activation_range.parquet")
MAX_NEW  = 150

# Effect must exceed this to be a candidate explanation.
# Step 35 perturbation increase was ~5% (non-flip) to ~12% (at flips).
MIN_EFFECT_PCT = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(FP32_DIR)
    m32 = ORTModelForSeq2SeqLM.from_pretrained(FP32_DIR, use_io_binding=False)

    df = pd.read_parquet(COHORT)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)

    # ---- resume ----
    done = set()
    if CKPT.exists():
        prev = pd.read_json(CKPT, lines=True)
        done = set(zip(prev.sample_id.astype(str), prev.cond))
        print(f"[resume] {len(done):,} (sentence, condition) pairs already done")

    total = len(df) * 2
    print(f"probing decoder activations: {len(df):,} sentences x 2 conditions "
          f"= {total:,} passes ({len(done):,} complete)\n")

    fh = CKPT.open("a", encoding="utf-8")
    try:
        for col, cond in [("gold_german_source", "clean"),
                          ("whisper_hypothesis", "asr")]:
            for _, r in tqdm(df.iterrows(), total=len(df), desc=cond):
                sid = str(r.get("sample_id"))
                if (sid, cond) in done:
                    continue
                t = r[col]
                if not isinstance(t, str) or not t.strip():
                    continue

                enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    gen = m32.generate(**enc, max_new_tokens=MAX_NEW,
                                       num_beams=1, do_sample=False)
                    seq = gen[0].tolist()
                    if len(seq) < 3:
                        continue
                    out = m32(**enc,
                              decoder_input_ids=torch.tensor([seq[:-1]]),
                              output_hidden_states=True)

                lg = out.logits[0].numpy()
                rec = {"sample_id": sid, "cond": cond,
                       "n_steps": int(lg.shape[0]),
                       "logit_absmax": float(np.abs(lg).max()),
                       "logit_p999": float(np.percentile(np.abs(lg), 99.9)),
                       "logit_std": float(lg.std())}

                hs_all = getattr(out, "decoder_hidden_states", None)
                if hs_all is not None:
                    hs = hs_all[-1][0].numpy()
                    a = np.abs(hs)
                    rec.update({"hid_absmax": float(a.max()),
                                "hid_p999": float(np.percentile(a, 99.9)),
                                "hid_p99": float(np.percentile(a, 99)),
                                "hid_std": float(hs.std())})

                fh.write(json.dumps(rec) + "\n")
                fh.flush()          # survives a hard power cut
    finally:
        fh.close()

    analyze()


def analyze():
    if not CKPT.exists():
        raise SystemExit("no checkpoint to analyse")
    res = pd.read_json(CKPT, lines=True)
    res.to_parquet(OUT, index=False)

    metrics = [c for c in ["hid_absmax", "hid_p999", "hid_p99", "hid_std",
                           "logit_absmax", "logit_p999", "logit_std", "n_steps"]
               if c in res.columns]

    print("\n" + "=" * 74)
    print(" DECODER ACTIVATION STATISTICS BY INPUT CONDITION")
    print("=" * 74)
    print(res.groupby("cond")[metrics].agg(["mean", "median"]).round(4).to_string())

    print("\n" + "=" * 74)
    print(" PAIRED COMPARISON  (effect size first, p-value second)")
    print("=" * 74)
    print(f"{'metric':<14} {'clean':>10} {'asr':>10} {'delta%':>9} "
          f"{'asr>clean':>10} {'p':>12}   verdict")
    print("-" * 74)

    candidates = []
    for m in metrics:
        piv = res.pivot_table(index="sample_id", columns="cond", values=m).dropna()
        if len(piv) < 100 or "clean" not in piv or "asr" not in piv:
            continue
        c, a = piv["clean"].mean(), piv["asr"].mean()
        pct = 100 * (a / c - 1) if c else float("nan")
        frac = 100 * (piv["asr"] > piv["clean"]).mean()
        p = mannwhitneyu(piv["asr"], piv["clean"], alternative="two-sided")[1]

        if abs(pct) >= MIN_EFFECT_PCT and p < 0.05:
            v = "CANDIDATE"; candidates.append((m, pct))
        elif p < 0.05:
            v = "sig but trivial"
        else:
            v = "no effect"
        print(f"{m:<14} {c:>10.4f} {a:>10.4f} {pct:>+8.2f}% "
              f"{frac:>9.1f}% {p:>12.2e}   {v}")

    print("-" * 74)
    print(f"paired on {len(piv):,} sentences")
    print(f"threshold: |delta| >= {MIN_EFFECT_PCT}% AND p < 0.05")
    print(f"(Step 35 perturbation increase was ~5% non-flip, ~12% at flips)\n")

    if candidates:
        print("CANDIDATE EXPLANATION(S) FOUND:")
        for m, pct in sorted(candidates, key=lambda x: -abs(x[1])):
            print(f"  {m}: {pct:+.2f}%")
        print("\n-> a shift this size could contribute to the perturbation")
        print("   increase. Report it, but sentence-level correlation does")
        print("   not establish causation.")
    else:
        print("NO CANDIDATE EXPLANATION.")
        print("Decoder statistics are effectively unchanged between conditions.")
        print("With the encoder result (+0.04%), the cause of the perturbation")
        print("increase remains unidentified.")
        print("-> report as a tested and rejected hypothesis in Step 36.")

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()