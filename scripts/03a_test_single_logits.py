#!/usr/bin/env python3
"""
BridgeDEUX: Single-Sentence Logits Diagnostic Test
==================================================
Technical validation to verify that raw decoder logits can be extracted
from the MarianMT ONNX computational graph with valid shapes and finite values.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer

def main():
    repo_root = Path(__file__).resolve().parent.parent
    fp32_dir = repo_root / "models/onnx/opus_mt_de_en_opt_extended"
    
    print("=" * 65)
    print(" 1. LOADING FP32 ONNX MODEL AND TOKENIZER")
    print("=" * 65)
    print(f"Path: {fp32_dir}")
    
    if not fp32_dir.exists():
        print(f"ERROR: Model directory does not exist: {fp32_dir}")
        sys.exit(1)
        
    tokenizer = MarianTokenizer.from_pretrained(fp32_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, provider="CPUExecutionProvider")
    
    print("\n" + "=" * 65)
    print(" 2. PREPARING DIAGNOSTIC INPUTS")
    print("=" * 65)
    source_text = "Das ist ein Test."
    print(f"Source German : '{source_text}'")
    
    # Tokenize the encoder input
    inputs = tokenizer(source_text, return_tensors="pt")
    
    # Resolve decoder start token dynamically from config or tokenizer
    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id
        
    decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
    eos_token_id = tokenizer.eos_token_id
    
    print(f"Encoder input shape       : {inputs.input_ids.shape}")
    print(f"Decoder start token ID    : {decoder_start_token_id}")
    print(f"EOS Token ID              : {eos_token_id}")
    print(f"Tokenizer Vocabulary Size : {tokenizer.vocab_size}")
    
    print("\n" + "=" * 65)
    print(" 3. EXECUTING FORWARD PASS")
    print("=" * 65)
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
    ms_elapsed = (time.perf_counter() - start_time) * 1000
    print(f"Forward pass completed in : {ms_elapsed:.2f} ms")
    
    # Validate logits existence and shape
    if not hasattr(outputs, "logits") or outputs.logits is None:
        print("FATAL: outputs.logits is missing or None.")
        sys.exit(1)
        
    logits = outputs.logits
    print(f"Logits Tensor Shape       : {list(logits.shape)} [batch, seq_step, vocab_size]")
    
    # Check for NaN / Inf values
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"Contains NaN values       : {has_nan}")
    print(f"Contains Inf values       : {has_inf}")
    
    if has_nan or has_inf:
        print("FATAL: Logits contain NaN or Inf values.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" 4. NUMERICAL LOGIT EXTRACTION")
    print("=" * 65)
    step_logits = logits[0, -1, :].numpy()
    eos_score = step_logits[eos_token_id]
    
    # Find the top non-EOS token
    masked_logits = np.copy(step_logits)
    masked_logits[eos_token_id] = -np.inf
    
    best_non_eos_id = int(np.argmax(masked_logits))
    best_non_eos_score = float(masked_logits[best_non_eos_id])
    best_non_eos_word = tokenizer.decode([best_non_eos_id])
    
    eos_margin = float(eos_score - best_non_eos_score)
    
    print(f"Top non-EOS Token : ID {best_non_eos_id} ('{best_non_eos_word}') -> Logit: {best_non_eos_score:.4f}")
    print(f"EOS Token         : ID {eos_token_id} ('{tokenizer.decode([eos_token_id])}') -> Logit: {eos_score:.4f}")
    print("-" * 65)
    print(f"Calculated Margin : {eos_margin:.4f}")
    
    # Final compatibility check
    vocab_matches = (logits.shape[-1] == tokenizer.vocab_size)
    print(f"Vocab Size Match  : {vocab_matches} ({logits.shape[-1]} == {tokenizer.vocab_size})")
    
    if vocab_matches and not (has_nan or has_inf):
        print("\nSTATUS: Logit telemetry verified. Interface is mathematically valid.")
    else:
        print("\nSTATUS: Interface check failed.")

if __name__ == "__main__":
    main()