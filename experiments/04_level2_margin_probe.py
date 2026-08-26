#!/usr/bin/env python3
"""
BridgeDEUX Level 2 — P(rank flip | logit gap) via ONNX teacher forcing
======================================================================
Both FP32 and INT8 are forced onto the SAME (FP32-generated) prefix at
every step. 

*CHECKPOINTED*: Writes to JSONL incrementally to survive power failures.
"""

import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

FP32_DIR = Path("models/onnx/opus_mt_de_en_opt_extended")
INT8_DIR = Path("models/onnx/opus_mt_de_en_opt_extended_int8")
COHORT   = Path("analysis/scored_qirg_cohort_complete.parquet")
MAX_NEW  = 150

def load(d):
    if not d.exists():
        sys.exit(f"[FATAL] missing {d}")
    return ORTModelForSeq2SeqLM.from_pretrained(d, use_io_binding=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=13511)
    ap.add_argument("--condition", choices=["clean", "asr"], default="clean")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(COHORT)
    print(f"cohort: {len(df):,} rows")

    src_col = "gold_german_source" if args.condition == "clean" else "whisper_hypothesis"
    for cand in ([src_col] if src_col in df.columns else []) or \
                ["source_text", "clean_source", "asr_source", "hypothesis"]:
        if cand in df.columns:
            src_col = cand
            break
    else:
        sys.exit(f"[FATAL] no source column found. Have: {list(df.columns)}")
    print(f"source column: {src_col}")

    if "wer_bin" in df.columns and args.limit < len(df):
        sub = (df.groupby("wer_bin", observed=True, group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), max(1, args.limit // df.wer_bin.nunique())),
                                           random_state=args.seed)))
    else:
        sub = df.sample(min(args.limit, len(df)), random_state=args.seed)
    sub = sub.reset_index(drop=True)
    print(f"probing {len(sub):,} sentences ({args.condition})\n")

    # --- CHECKPOINTING LOGIC ---
    ckpt = Path(f"analysis/level2_ckpt_{args.condition}.jsonl")
    done = set()
    if ckpt.exists():
        try:
            prev = pd.read_json(ckpt, lines=True)
            if not prev.empty and "sample_id" in prev.columns:
                done = set(prev.sample_id.unique())
            print(f"[resume] {len(done):,} sentences already done, skipping them")
        except ValueError:
            print("[warn] Checkpoint file exists but is empty or corrupted. Starting fresh.")
            
    fh = ckpt.open("a", encoding="utf-8")
    # ---------------------------

    tok = AutoTokenizer.from_pretrained(FP32_DIR)
    m32, m8 = load(FP32_DIR), load(INT8_DIR)

    for _, r in tqdm(sub.iterrows(), total=len(sub)):
        sid = r.get("sample_id", _)
        if sid in done:
            continue
            
        text = r[src_col]
        if not isinstance(text, str) or not text.strip():
            continue

        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            gen = m32.generate(**enc, max_new_tokens=MAX_NEW,
                               num_beams=1, do_sample=False)
        seq = gen[0].tolist()
        if len(seq) < 3:
            continue

        dec_in = torch.tensor([seq[:-1]])
        with torch.no_grad():
            lg32 = m32(**enc, decoder_input_ids=dec_in).logits[0]
            lg8  = m8(**enc,  decoder_input_ids=dec_in).logits[0]

        top2_32 = torch.topk(lg32, 2, dim=-1)
        margin  = (top2_32.values[:, 0] - top2_32.values[:, 1]).numpy()
        arg32   = top2_32.indices[:, 0].numpy()
        arg8    = lg8.argmax(-1).numpy()

        top2_8   = torch.topk(lg8, 2, dim=-1)
        margin_8 = (top2_8.values[:, 0] - top2_8.values[:, 1]).numpy()
        pert     = (lg8 - lg32).abs().mean(-1).numpy()

        T = len(margin)
        batch = []
        for t in range(T):
            batch.append({
                "sample_id": sid,
                "position": t,
                "rel_position": t / max(T - 1, 1),
                "seq_len": T,
                "margin_fp32": float(margin[t]),
                "margin_int8": float(margin_8[t]),
                "mean_abs_pert": float(pert[t]),
                "is_flip": int(arg32[t] != arg8[t]),
                "wer_bin": r.get("wer_bin", None),
            })
            
        # Write to disk immediately
        for rec in batch:
            fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush() 

    fh.close()
    
    # Compile final parquet from the checkpoint
    final = pd.read_json(ckpt, lines=True)
    out = Path(f"analysis/level2_margin_tokens_{args.condition}.parquet")
    final.to_parquet(out, index=False)
    print(f"\nlogged {len(final):,} token positions from "
          f"{final.sample_id.nunique():,} sentences -> {out}")

if __name__ == "__main__":
    main()