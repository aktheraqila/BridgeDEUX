#!/usr/bin/env python3
"""
BridgeDEUX: Step 3 - Causal EOS Margin Probe (Strict Edition)
=============================================================
Investigates the 319 severe truncations. Finds the Longest Common Prefix (LCP).
Separates "Pure EOS Truncations" from "Structural Rewrites".
For pure EOS cases, it probes the exact decision boundary to extract:
- EOS Logit vs Best Non-EOS Logit
- Exact EOS Margin
- Sign flips (proving the boundary was crossed)
- The raw argmax token
"""

import sys
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def find_latest_file(pattern: str) -> Path:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return Path(files[0])

def get_logit_stats(logits: np.ndarray, eos_id: int):
    """Extracts precise logit telemetry for the step."""
    eos_logit = float(logits[eos_id])
    
    masked = np.copy(logits)
    masked[eos_id] = -np.inf
    best_non_eos_id = int(np.argmax(masked))
    best_non_eos_logit = float(masked[best_non_eos_id])
    
    margin = eos_logit - best_non_eos_logit
    argmax_id = int(np.argmax(logits))
    
    return eos_logit, best_non_eos_logit, margin, argmax_id

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    
    try:
        parquet_file = find_latest_file(str(analysis_dir / "comet_divergent_scores_*.parquet"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_file)
    trunc_df = df[df["delta_output_tokens"].abs() > 2].copy()
    
    print("=" * 75)
    print(" STEP 3: CAUSAL EOS MARGIN PROBE")
    print("=" * 75)
    print(f"Loaded {len(trunc_df)} severe divergence cases.")
    
    fp32_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended"
    int8_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended_int8"
    
    print("Loading ONNX computation graphs and Tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(fp32_dir)
    fp32_model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, provider="CPUExecutionProvider")
    int8_model = ORTModelForSeq2SeqLM.from_pretrained(int8_dir, provider="CPUExecutionProvider")
    
    eos_token_id = tokenizer.eos_token_id
    decoder_start_token_id = getattr(fp32_model.config, "decoder_start_token_id", tokenizer.pad_token_id)
    
    results = []
    structural_rewrites = 0
    pure_eos_truncations = 0
    
    print("\nRunning Shared-Prefix Logit Extraction...")
    for idx, row in tqdm(trunc_df.iterrows(), total=len(trunc_df), desc="Probing Models"):
        source = row["source"]
        f_text = row["fp32_translation"]
        i_text = row["int8_translation"]
        
        # Tokenize both outputs (without special tokens)
        f_toks = tokenizer(f_text, add_special_tokens=False).input_ids
        i_toks = tokenizer(i_text, add_special_tokens=False).input_ids
        
        # Find the Longest Common Prefix (LCP)
        min_len = min(len(f_toks), len(i_toks))
        lcp_len = 0
        for i in range(min_len):
            if f_toks[i] == i_toks[i]:
                lcp_len += 1
            else:
                break
                
        # Classify divergence type
        is_pure_eos = False
        truncated_model = None
        
        if lcp_len == len(f_toks) and len(i_toks) > len(f_toks):
            is_pure_eos = True
            truncated_model = "FP32"
        elif lcp_len == len(i_toks) and len(f_toks) > len(i_toks):
            is_pure_eos = True
            truncated_model = "INT8"
            
        if not is_pure_eos:
            structural_rewrites += 1
            continue
            
        pure_eos_truncations += 1
        shared_prefix = f_toks[:lcp_len]
        
        # Prepare inputs
        inputs = tokenizer(source, return_tensors="pt")
        prefix_tensor = torch.tensor([shared_prefix], dtype=torch.long)
        
        if lcp_len == 0:
            decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
        else:
            decoder_input_ids = torch.cat([torch.tensor([[decoder_start_token_id]]), prefix_tensor], dim=1)
        
        # Forward Pass
        with torch.no_grad():
            fp32_out = fp32_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=decoder_input_ids)
            int8_out = int8_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=decoder_input_ids)
            
        f_logits = fp32_out.logits[0, -1, :].numpy()
        i_logits = int8_out.logits[0, -1, :].numpy()
        
        # Extract telemetry
        f_eos_log, f_best_non_eos_log, f_margin, f_argmax = get_logit_stats(f_logits, eos_token_id)
        i_eos_log, i_best_non_eos_log, i_margin, i_argmax = get_logit_stats(i_logits, eos_token_id)
        
        # Did the margin cross zero? (i.e. did the winning token flip between EOS and non-EOS?)
        margin_sign_flip = (f_margin > 0 and i_margin < 0) or (f_margin < 0 and i_margin > 0)
        
        results.append({
            "sample_id": row["sample_id"],
            "truncated_model": truncated_model,
            "prefix_len": lcp_len,
            "fp32_eos_logit": f_eos_log,
            "int8_eos_logit": i_eos_log,
            "fp32_best_non_eos_logit": f_best_non_eos_log,
            "int8_best_non_eos_logit": i_best_non_eos_log,
            "fp32_eos_margin": f_margin,
            "int8_eos_margin": i_margin,
            "margin_diff_abs": abs(f_margin - i_margin),
            "fp32_argmax_id": f_argmax,
            "int8_argmax_id": i_argmax,
            "margin_sign_flip": margin_sign_flip
        })

    print("\n" + "=" * 75)
    print(" RESULTS: CAUSAL EOS MARGIN PROBE")
    print("=" * 75)
    print(f"Total Severe Divergences  : {len(trunc_df)}")
    print(f"Structural Rewrites       : {structural_rewrites} (Diverged before truncation)")
    print(f"Pure EOS Truncations      : {pure_eos_truncations} (Exact shared prefix)")
    
    if pure_eos_truncations == 0:
        print("\nCONCLUSION: 0 cases of pure EOS divergence found.")
        print("The hypothesis is rejected. Truncations are downstream effects of earlier structural rewrites.")
        sys.exit(0)
        
    res_df = pd.DataFrame(results)
    out_path = analysis_dir / "eos_causal_margins.csv"
    res_df.to_csv(out_path, index=False)
    
    print("\n[Analysis of Pure EOS Boundaries]")
    flips = res_df[res_df["margin_sign_flip"] == True]
    print(f"Boundaries with explicit EOS sign flip : {len(flips)} / {pure_eos_truncations}")
    print(f"Average absolute logit perturbation  : {res_df['margin_diff_abs'].mean():.4f} logits")
    
    for model in ["FP32", "INT8"]:
        subset = res_df[res_df["truncated_model"] == model]
        if len(subset) > 0:
            print(f"\nWhen {model} Truncates (N={len(subset)}):")
            print(f"  Mean FP32 EOS Margin : {subset['fp32_eos_margin'].mean():.4f}")
            print(f"  Mean INT8 EOS Margin : {subset['int8_eos_margin'].mean():.4f}")

    print("\nData saved to:", out_path.name)
    print("=" * 75)

if __name__ == "__main__":
    main()