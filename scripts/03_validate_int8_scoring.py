#!/usr/bin/env python3
"""
BridgeDEUX: Step 3 (Validation) - INT8 ONNX Scoring Sanity Check
================================================================
Verifies that the Optimum ONNX INT8 model successfully accepts 
teacher-forced inputs and returns mathematically valid logits for 
Sequence Log-Probability calculation.
"""

import os
import sys
import glob
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OnnxValidator")

def find_latest_file(pattern: str) -> Path:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])

def load_jsonl_subset(file_path: Path, limit: int = None) -> list:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return records

def main():
    parser = argparse.ArgumentParser(description="Validate ONNX INT8 Seq-Logprob Extraction")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to process")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    model_dir = repo_root / "models" / "onnx" / "opus_mt_de_en_opt_extended_int8"

    # 1. Check Model Directory
    logger.info("CHECK 1: Model Loads")
    if not model_dir.exists():
        logger.error(f"FAIL: Cannot find INT8 model directory at {model_dir}")
        sys.exit(1)

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
        logger.info("PASS: Tokenizer and ORTModelForSeq2SeqLM loaded successfully.")
    except Exception as e:
        logger.error(f"FAIL: Model loading crashed: {e}")
        sys.exit(1)

    # Load a small subset of the data
    try:
        int8_file = find_latest_file(str(results_dir / "marianmt-onnx_opus_mt_de_en_opt_extended_int8_covost2_de_en_test" / "*.jsonl.bak"))
        test_samples = load_jsonl_subset(int8_file, limit=args.limit)
        logger.info(f"Loaded {len(test_samples)} samples for validation.")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    vocab_size = model.config.vocab_size

    for idx, rec in enumerate(test_samples):
        logger.info(f"--- Validating Sample {idx + 1} ---")
        src = rec.get("source_text", "")
        hyp = rec.get("translation", "")

        if not hyp.strip():
            logger.warning("Empty translation. Skipping to next sample.")
            continue

        inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=512)
        labels = tokenizer(text_target=hyp, return_tensors="pt", truncation=True, max_length=512)

        # 2. Check decoder_input_ids acceptance
        logger.info("CHECK 2: decoder_input_ids are accepted")
        try:
            # MarianMT standard: shift labels right and prepend pad_token_id
            decoder_input_ids = torch.full_like(labels["input_ids"], tokenizer.pad_token_id)
            decoder_input_ids[:, 1:] = labels["input_ids"][:, :-1]
            decoder_input_ids[:, 0] = model.config.decoder_start_token_id
            logger.info("PASS: decoder_input_ids successfully constructed.")
        except Exception as e:
            logger.error(f"FAIL: Could not construct decoder_input_ids: {e}")
            continue

        # 3 & 4. Check logits extraction and dimensions
        logger.info("CHECK 3 & 4: Logits returned & vocabulary dimension matches")
        try:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            
            expected_shape = (1, decoder_input_ids.shape[1], vocab_size)
            if logits.shape == expected_shape:
                logger.info(f"PASS: Logits returned with expected shape {logits.shape}")
            else:
                logger.error(f"FAIL: Logit shape mismatch. Expected {expected_shape}, got {logits.shape}")
                continue
        except Exception as e:
            logger.error(f"FAIL: Forward pass crashed: {e}")
            continue

        # 5 & 6. Check finite loss and translatability
        logger.info("CHECK 5 & 6: Loss is finite and translation scored successfully")
        try:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            # Flatten logits and labels for CrossEntropy
            loss = loss_fct(logits.view(-1, vocab_size), labels["input_ids"].view(-1))
            
            if torch.isfinite(loss):
                logger.info(f"PASS: Loss is finite. Seq-Loss: {loss.item():.4f}")
            else:
                logger.error("FAIL: Loss is NaN or Inf.")
        except Exception as e:
            logger.error(f"FAIL: Loss calculation crashed: {e}")
            continue
        
        logger.info(f"Sample {idx + 1} fully validated.\n")

if __name__ == "__main__":
    main()