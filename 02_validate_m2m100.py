"""
Module 2: PyTorch Baseline Validation (M2M100)

Purpose
-------
Validate that Facebook M2M100:

1. Downloads correctly
2. Loads correctly
3. Produces valid translations
4. Measures inference stages separately

This module DOES NOT:
- Export ONNX
- Compute BLEU / ROUGE
- Benchmark Android
- Measure RAM
- Quantize models
"""

import time

import torch
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)
from utils.experiment_logger import ExperimentLogger


def validate_translation_model(
    model_name: str,
    source_language: str,
    target_language: str,
    input_sentences: list[str],
):
    """
    Validate M2M100 translation model.
    """

    print("=" * 60)
    print(f"Validating Model: {model_name}")
    print("=" * 60)

    # --------------------------------------------------
    # Load Tokenizer
    # --------------------------------------------------

    print("\n[1/4] Loading tokenizer...")

    start = time.perf_counter()

    tokenizer = M2M100Tokenizer.from_pretrained(model_name)

    tokenizer.src_lang = source_language

    tokenizer_load_time = time.perf_counter() - start

    print(f"Done ({tokenizer_load_time:.3f} sec)")

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    print("\n[2/4] Loading model...")

    start = time.perf_counter()

    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    model.eval()

    model_load_time = time.perf_counter() - start

    print(f"Done ({model_load_time:.3f} sec)")

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print("\n[3/4] Running warm-up inference...")

    warmup_inputs = tokenizer(
        ["Hallo."],
        return_tensors="pt",
    )

    with torch.no_grad():

        _ = model.generate(
            **warmup_inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_language),
        )

    print("Warm-up complete.")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print("\n[4/4] Running validation...")

    # Tokenization

    start = time.perf_counter()

    inputs = tokenizer(
        input_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    tokenization_time = time.perf_counter() - start

    # Generation

    start = time.perf_counter()

    with torch.no_grad():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_language),
        )

    generation_time = time.perf_counter() - start

    # Decoding

    start = time.perf_counter()

    output_sentences = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    decoding_time = time.perf_counter() - start

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\nTranslation Results")
    print("-" * 60)

    for source, target in zip(
        input_sentences,
        output_sentences,
    ):

        print(f"IN : {source}")
        print(f"OUT: {target}")
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

    return {
        "model_name": model_name,
        "translations": output_sentences,
        "timings": {
            "tokenizer_load": tokenizer_load_time,
            "model_load": model_load_time,
            "tokenization": tokenization_time,
            "generation": generation_time,
            "decoding": decoding_time,
            "total": total_time,
        },
    }


if __name__ == "__main__":

    MODEL_NAME = "facebook/m2m100_418M"

    SOURCE_LANGUAGE = "de"

    TARGET_LANGUAGE = "en"

    TEST_SENTENCES = [
        "Guten Morgen. Wie geht es Ihnen?",
        "Ich brauche eine Wegbeschreibung zum Bahnhof.",
        "Die Batterietemperatur ist zu hoch.",
    ]

    results = validate_translation_model(
        MODEL_NAME,
        SOURCE_LANGUAGE,
        TARGET_LANGUAGE,
        TEST_SENTENCES,
    )

    # ------------------------------------------
    # Experiment Metadata
    # ------------------------------------------

    results["experiment_id"] = "MT-002"
    results["direction"] = "de->en"
    results["framework"] = "PyTorch"
    results["device"] = "Desktop"
    results["notes"] = "Baseline M2M100 validation"

    # ------------------------------------------
    # Save Results
    # ------------------------------------------

    logger = ExperimentLogger()
    logger.log(results)
