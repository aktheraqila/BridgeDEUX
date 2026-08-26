#!/usr/bin/env python3
"""
BridgeDEUX: Step 3 - Exploratory Sequence Likelihood Measurement
================================================================
Performs a Teacher-Forcing scoring pass on the generated hypotheses to 
calculate the INT8 model's generated-sequence self-likelihood (L_hyp). 
Tests continuous correlation against actual translation degradation (delta_chrf).
Maintains all 13,511 samples, isolating empty outputs as a specific category.
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformers import MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import sacrebleu
from scipy import stats
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Int8LikelihoodProbe")


def find_latest_file(pattern: str) -> Path:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])


def compute_sentence_chrf(hypothesis: str, reference: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    return float(sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score)


def load_jsonl_bak_records(file_path: Path) -> dict:
    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records[data["sample_id"]] = data
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def shift_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Verified shifting logic matching Hugging Face internal implementation."""
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    analysis_dir = repo_root / "analysis"
    model_dir = repo_root / "models" / "onnx" / "opus_mt_de_en_opt_extended_int8"
    
    try:
        fp32_file = find_latest_file(str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_covost2_de_en_test" / "*.jsonl.bak"))
        int8_file = find_latest_file(str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "*.jsonl.bak"))
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    fp32_data = load_jsonl_bak_records(fp32_file)
    int8_data = load_jsonl_bak_records(int8_file)
    
    # Deterministic alphabetical ordering of sample IDs
    common_ids = sorted(set(fp32_data.keys()).intersection(int8_data.keys()))
    
    logger.info(f"Aligned {len(common_ids):,} common samples.")

    logging.getLogger("optimum.onnxruntime.modeling_seq2seq").setLevel(logging.ERROR)
    logger.info("Loading ONNX INT8 MarianMT Model...")
    
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
    
    vocab_size = model.config.vocab_size
    pad_token_id = model.config.pad_token_id
    decoder_start_token_id = model.config.decoder_start_token_id

    rows = []
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    logger.info("Executing Exploratory Scoring Pass (Est. Time: ~35-45 mins)...")
    
    with torch.no_grad():
        for idx, sid in enumerate(common_ids):
            if idx > 0 and idx % 1000 == 0:
                logger.info(f"Processed {idx:,}/{len(common_ids):,} samples.")
                
            f_rec = fp32_data[sid]
            i_rec = int8_data[sid]

            src = i_rec.get("source_text", "")
            ref = i_rec.get("reference_translation", "")
            f_hyp = f_rec.get("translation", "")
            i_hyp = i_rec.get("translation", "")

            f_chrf = compute_sentence_chrf(f_hyp, ref)
            i_chrf = compute_sentence_chrf(i_hyp, ref)
            delta_chrf = f_chrf - i_chrf
            
            # Explicit tracking of generation validity
            generation_status = "Generated"
            
            if not i_hyp.strip():
                seq_loss = np.nan
                generation_status = "Empty Output"
            else:
                inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=512)
                labels = tokenizer(text_target=i_hyp, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
                
                labels[labels == pad_token_id] = -100
                decoder_input_ids = shift_right(labels, pad_token_id, decoder_start_token_id)
                
                outputs = model(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids
                )
                
                loss = loss_fct(outputs.logits.view(-1, vocab_size), labels.view(-1))
                seq_loss = loss.item()

            f_out_toks = max(f_rec.get("output_tokens", 1), 1)
            i_out_toks = max(i_rec.get("output_tokens", 1), 1)

            rows.append({
                "sample_id": sid,
                "fp32_chrf": f_chrf,
                "int8_chrf": i_chrf,
                "delta_chrf": delta_chrf,
                "seq_loss_hyp": seq_loss,
                "generation_status": generation_status,
                "delta_output_tokens": f_out_toks - i_out_toks,
                "fp32_ms_per_token": f_rec.get("total_time_ms", 0) / f_out_toks,
                "int8_ms_per_token": i_rec.get("total_time_ms", 0) / i_out_toks
            })

    df = pd.DataFrame(rows)

    logger.info("Computing Continuous Correlations...")
    valid_df = df[df["generation_status"] == "Generated"].copy()
    
    pearson_r, p_p = stats.pearsonr(valid_df["seq_loss_hyp"], valid_df["delta_chrf"])
    spearman_rho, p_s = stats.spearmanr(valid_df["seq_loss_hyp"], valid_df["delta_chrf"])

    # Categorize degeneration direction
    valid_df["flip_direction"] = "Identical / Minor (±2 tokens)"
    valid_df.loc[valid_df["delta_output_tokens"] > 2, "flip_direction"] = "INT8 Truncated (FP32 longer)"
    valid_df.loc[valid_df["delta_output_tokens"] < -2, "flip_direction"] = "FP32 Truncated (INT8 longer)"

    hygiene_stats = valid_df.groupby("flip_direction", observed=False).agg(
        Count=("sample_id", "count"),
        Mean_Delta_chrF=("delta_chrf", "mean"),
        Mean_Seq_Loss=("seq_loss_hyp", "mean"),
        Std_Seq_Loss=("seq_loss_hyp", "std")
    ).reset_index()

    print("\n" + "="*80)
    print(" STEP 3: REFERENCE-FREE SEQUENCE LIKELIHOOD REPORT")
    print("="*80)
    print(f"Total Samples Processed: {len(df):,}")
    print(f"Empty Translations (Excluded from continuous math): {len(df) - len(valid_df):,}")
    print("-" * 80)
    
    corr_results = [
        ["Hypothesis Seq-Loss (L_hyp) vs Δ chrF++", round(pearson_r, 4), f"{p_p:.2e}", round(spearman_rho, 4), f"{p_s:.2e}"]
    ]
    print("Does INT8 Self-Likelihood correlate with actual quality degradation (Delta chrF++)?")
    print("Note: Positive delta means FP32 is better (INT8 lost quality).")
    print(tabulate(corr_results, headers=["Signal", "Pearson (r)", "p-val", "Spearman (rho)", "p-val"], tablefmt="fancy_grid"))
    
    print("\n[Self-Likelihood by Bidirectional Degeneration Category]")
    print(tabulate(hygiene_stats, headers="keys", tablefmt="fancy_grid", showindex=False))
    print("="*80 + "\n")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = analysis_dir / f"int8_seq_likelihood_exploratory_{run_timestamp}.parquet"
    df.to_parquet(out_file, index=False)
    logger.info(f"Saved artifacts to {out_file.name}")


if __name__ == "__main__":
    main()