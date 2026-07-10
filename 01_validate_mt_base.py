"""
Module 1: PyTorch Baseline Validation

Purpose:
--------
Validate that a Hugging Face MarianMT model:
1. Downloads correctly
2. Loads correctly
3. Produces valid translations
4. Measures inference stages separately

This module DOES NOT:
- Export ONNX
- Compute BLEU/ROUGE
- Benchmark Android
- Measure RAM
- Quantize models
"""

import time

import torch
from transformers import MarianMTModel, MarianTokenizer

from utils.experiment_logger import ExperimentLogger


def validate_translation_model(model_name: str, german_sentences: list[str]):
    """
    Validate a translation model.

    Parameters
    ----------
    model_name : str
        Hugging Face model name.

    german_sentences : list[str]
        German sentences for validation.

    Returns
    -------
    dict
        Translation results and timing information.
    """

    print("=" * 60)
    print(f"Validating Model: {model_name}")
    print("=" * 60)

    # --------------------------------------------------
    # Load Tokenizer
    # --------------------------------------------------

    print("\n[1/4] Loading tokenizer...")

    start = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    tokenizer_load_time = time.perf_counter() - start

    print(f"Done ({tokenizer_load_time:.3f} sec)")

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    print("\n[2/4] Loading model...")

    start = time.perf_counter()
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    model_load_time = time.perf_counter() - start

    print(f"Done ({model_load_time:.3f} sec)")

    # --------------------------------------------------
    # Warm-up Run
    # --------------------------------------------------

    print("\n[3/4] Running warm-up inference...")

    warmup_inputs = tokenizer(
        ["Hallo."],
        return_tensors="pt"
    )

    with torch.no_grad():
        _ = model.generate(**warmup_inputs)

    print("Warm-up complete.")

    # --------------------------------------------------
    # Actual Validation
    # --------------------------------------------------

    print("\n[4/4] Running validation...")

    # Tokenization
    start = time.perf_counter()

    inputs = tokenizer(
        german_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    tokenization_time = time.perf_counter() - start

    # Translation
    start = time.perf_counter()

    with torch.no_grad():
        generated_tokens = model.generate(**inputs)

    generation_time = time.perf_counter() - start

    # Decoding
    start = time.perf_counter()

    english_sentences = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    decoding_time = time.perf_counter() - start

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------

    print("\nTranslation Results")
    print("-" * 60)

    for de, en in zip(german_sentences, english_sentences):
        print(f"DE : {de}")
        print(f"EN : {en}")
        print()

    total_time = (
        tokenization_time
        + generation_time
        + decoding_time
    )

    print("-" * 60)
    print("Timing")
    print("-" * 60)

    print(f"Tokenizer Load : {tokenizer_load_time:.4f} sec")
    print(f"Model Load     : {model_load_time:.4f} sec")
    print(f"Tokenization   : {tokenization_time:.4f} sec")
    print(f"Generation     : {generation_time:.4f} sec")
    print(f"Decoding       : {decoding_time:.4f} sec")
    print(f"Total          : {total_time:.4f} sec")

    # --------------------------------------------------
    # Return
    # --------------------------------------------------

    return {
        "model_name": model_name,
        "translations": english_sentences,
        "timings": {
            "tokenizer_load": tokenizer_load_time,
            "model_load": model_load_time,
            "tokenization": tokenization_time,
            "generation": generation_time,
            "decoding": decoding_time,
            "total": total_time,
        }
    }


if __name__ == "__main__":

    MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"

    TEST_SENTENCES = [
        "Guten Morgen. Wie geht es Ihnen?",
        "Ich brauche eine Wegbeschreibung zum Bahnhof.",
        "Die Batterietemperatur ist zu hoch.",
    ]

    results = validate_translation_model(
        MODEL_NAME,
        TEST_SENTENCES
    )

    results["experiment_id"] = "MT-001"
    results["direction"] = "de->en"

    logger = ExperimentLogger()
    logger.log(results)

