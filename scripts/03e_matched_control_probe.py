#!/usr/bin/env python3
"""
BridgeDEUX: Step 3E - Matched Control Margin Probe (Fixed)
==========================================================
Compares the margins of the 318 divergent cases against 318 
position-matched healthy (identical) translations. Bypasses 
column-name issues by loading raw inference files directly.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")

def get_top2_margin(logits: np.ndarray):
    top2_indices = np.argsort(logits)[-2:][::-1]
    return float(logits[top2_indices[0]] - logits[top2_indices[1]])

def main():
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis"
    
    div_file = analysis_dir / "first_divergence_margins.csv"
    if not div_file.exists():
        print("Error: Run 03d_first_divergence_probe.py first.")
        sys.exit(1)
    div_df = pd.read_csv(div_file)
    
    # 1. LOAD RAW INFERENCE FILES DIRECTLY
    print("Loading raw inference outputs directly...")
    try:
        fp32_raw_path = list((repo_root / "results").rglob("*opus_mt_de_en_opt_extended_covost2_*/*_results.parquet"))[0]
        int8_raw_path = list((repo_root / "results").rglob("*opus_mt_de_en_opt_extended_int8_covost2_*/*_results.parquet"))[0]
    except IndexError:
        print("FATAL: Could not locate the raw results parquet files in the /results/ directory.")
        sys.exit(1)
        
    df_f = pd.read_parquet(fp32_raw_path)
    df_i = pd.read_parquet(int8_raw_path)
    
    # Detect the prediction column name dynamically (translation, prediction, etc.)
    pred_col = "translation" if "translation" in df_f.columns else ("prediction" if "prediction" in df_f.columns else df_f.columns[1])
    source_col = "source" if "source" in df_f.columns else df_f.columns[0]
    
    # Build the healthy dataframe (where outputs are perfectly identical)
    is_healthy = df_f[pred_col] == df_i[pred_col]
    healthy_df = pd.DataFrame({
        "source": df_f.loc[is_healthy, source_col].values,
        "fp32_translation": df_f.loc[is_healthy, pred_col].values
    })
    
    print("=" * 75)
    print(" STEP 3E: MATCHED CONTROL MARGIN PROBE")
    print("=" * 75)
    print(f"Loaded {len(div_df)} Divergent cases.")
    print(f"Loaded {len(healthy_df)} Healthy Control cases directly from raw inferences.")
    
    if len(healthy_df) == 0:
        print("FATAL: 0 healthy cases found. The arrays might be misaligned.")
        sys.exit(1)
        
    # Match N size and Position Distribution
    N = len(div_df)
    control_sample = healthy_df.sample(n=N, random_state=42).reset_index(drop=True)
    
    divergent_positions = div_df["prefix_len"].tolist()
    random.seed(42)
    random.shuffle(divergent_positions)
    
    fp32_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended"
    tokenizer = MarianTokenizer.from_pretrained(fp32_dir)
    fp32_model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, provider="CPUExecutionProvider")
    decoder_start_token_id = getattr(fp32_model.config, "decoder_start_token_id", tokenizer.pad_token_id)
    
    control_margins = []
    
    print("\nProbing Matched Control Positions (FP32)...")
    for idx, row in tqdm(control_sample.iterrows(), total=N, desc="Extracting Control Margins"):
        source = row["source"]
        f_text = row["fp32_translation"]
        f_toks = tokenizer(f_text, add_special_tokens=False).input_ids
        
        target_len = divergent_positions[idx]
        safe_len = min(target_len, max(0, len(f_toks) - 1))
        shared_prefix = f_toks[:safe_len]
        
        inputs = tokenizer(source, return_tensors="pt")
        prefix_tensor = torch.tensor([shared_prefix], dtype=torch.long)
        
        if safe_len == 0:
            decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
        else:
            decoder_input_ids = torch.cat([torch.tensor([[decoder_start_token_id]]), prefix_tensor], dim=1)
            
        with torch.no_grad():
            fp32_out = fp32_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=decoder_input_ids)
            
        logits = fp32_out.logits[0, -1, :].numpy()
        margin = get_top2_margin(logits)
        control_margins.append(margin)

    div_mean = div_df['fp32_margin'].mean()
    ctrl_mean = np.mean(control_margins)
    
    div_under_2 = (div_df['fp32_margin'] < 2.0).mean() * 100
    ctrl_under_2 = (np.array(control_margins) < 2.0).mean() * 100
    
    print("\n" + "=" * 75)
    print(" RESULTS: DIVERGENT VS MATCHED CONTROL")
    print("=" * 75)
    print(f"[Divergent Group]  Mean FP32 Margin : {div_mean:.4f} logits")
    print(f"[Control Group]    Mean FP32 Margin : {ctrl_mean:.4f} logits")
    print("-" * 75)
    print(f"[Divergent Group]  Cases < 2.0 Margin : {div_under_2:.1f}%")
    print(f"[Control Group]    Cases < 2.0 Margin : {ctrl_under_2:.1f}%")
    print("=" * 75)
    
    if ctrl_mean > (div_mean + 1.0) or (ctrl_under_2 < div_under_2 - 20):
        print("\nCONCLUSION: HYPOTHESIS SUPPORTED.")
        print("Healthy translations maintain significantly higher margins (or fewer tight margins).")
        print("This establishes the NMT-specific empirical link between low-margin decisions and structural rewrites.")
    else:
        print("\nCONCLUSION: NULL RESULT.")
        print("Control margins are similar to divergent margins.")
        print("This implies low margin alone does not guarantee a structural rewrite.")

if __name__ == "__main__":
    main()