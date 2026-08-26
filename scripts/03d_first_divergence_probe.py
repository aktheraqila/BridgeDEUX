#!/usr/bin/env python3
"""
BridgeDEUX: Step 3D - First-Divergence Margin Probe
===================================================
Analyzes the 318 structural rewrites. Pauses the decoder at the exact 
Longest Common Prefix (LCP) where FP32 and INT8 first disagreed.
Extracts the Top-1 vs Top-2 logit margin to prove that quantization 
flips occur precisely at boundaries of high uncertainty (near-ties).
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

def get_top2_margin(logits: np.ndarray):
    """Returns Top-1 logit, Top-2 logit, the margin between them, and the Argmax ID."""
    top2_indices = np.argsort(logits)[-2:][::-1] # Get top 2 indices, descending
    top1_id, top2_id = top2_indices[0], top2_indices[1]
    
    top1_logit = float(logits[top1_id])
    top2_logit = float(logits[top2_id])
    margin = top1_logit - top2_logit
    
    return top1_logit, top2_logit, margin, int(top1_id)

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    
    try:
        parquet_file = find_latest_file(str(analysis_dir / "comet_divergent_scores_*.parquet"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_file)
    # Target the severe divergences
    trunc_df = df[df["delta_output_tokens"].abs() > 2].copy()
    
    fp32_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended"
    int8_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended_int8"
    
    print("=" * 75)
    print(" STEP 3D: FIRST-DIVERGENCE MARGIN PROBE")
    print("=" * 75)
    
    print("Loading ONNX computation graphs and Tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(fp32_dir)
    fp32_model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, provider="CPUExecutionProvider")
    int8_model = ORTModelForSeq2SeqLM.from_pretrained(int8_dir, provider="CPUExecutionProvider")
    
    decoder_start_token_id = getattr(fp32_model.config, "decoder_start_token_id", tokenizer.pad_token_id)
    
    results = []
    
    print("\nProbing the exact moment of structural divergence...")
    for idx, row in tqdm(trunc_df.iterrows(), total=len(trunc_df), desc="Extracting Margins"):
        source = row["source"]
        f_text = row["fp32_translation"]
        i_text = row["int8_translation"]
        
        f_toks = tokenizer(f_text, add_special_tokens=False).input_ids
        i_toks = tokenizer(i_text, add_special_tokens=False).input_ids
        
        # Find LCP (Longest Common Prefix)
        min_len = min(len(f_toks), len(i_toks))
        lcp_len = 0
        for i in range(min_len):
            if f_toks[i] == i_toks[i]:
                lcp_len += 1
            else:
                break
                
        # Filter OUT the pure EOS cases (we already know there is only 1)
        if lcp_len == len(f_toks) or lcp_len == len(i_toks):
            continue 
            
        shared_prefix = f_toks[:lcp_len]
        
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
        
        f_top1, f_top2, f_margin, f_argmax = get_top2_margin(f_logits)
        i_top1, i_top2, i_margin, i_argmax = get_top2_margin(i_logits)
        
        results.append({
            "sample_id": row["sample_id"],
            "prefix_len": lcp_len,
            "fp32_top1_logit": f_top1,
            "fp32_margin": f_margin,
            "int8_top1_logit": i_top1,
            "int8_margin": i_margin,
            "fp32_argmax_id": f_argmax,
            "int8_argmax_id": i_argmax,
            "margin_diff_abs": abs(f_margin - i_margin)
        })

    res_df = pd.DataFrame(results)
    out_path = analysis_dir / "first_divergence_margins.csv"
    res_df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 75)
    print(" RESULTS: FIRST-DIVERGENCE MARGIN PROBE")
    print("=" * 75)
    print(f"Total Structural Rewrites Analyzed : {len(res_df)}")
    
    print(f"\n[Margin Telemetry at the Moment of Divergence]")
    print(f"Mean FP32 Top-1/Top-2 Margin : {res_df['fp32_margin'].mean():.4f} logits")
    print(f"Mean INT8 Top-1/Top-2 Margin : {res_df['int8_margin'].mean():.4f} logits")
    print(f"Mean Absolute Quantization Noise (Logit Shift) : {res_df['margin_diff_abs'].mean():.4f} logits")
    
    highly_uncertain = res_df[res_df['fp32_margin'] < 2.0]
    print(f"\nDivergences triggered under high uncertainty (Margin < 2.0) : {len(highly_uncertain)} / {len(res_df)}")
    
    print("\nData saved to:", out_path.name)
    print("=" * 75)

if __name__ == "__main__":
    main()