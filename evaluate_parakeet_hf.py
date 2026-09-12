"""
Primary HF/PyTorch evaluation of NVIDIA Parakeet TDT v3 on MSLT German test.

Evaluation protocol
--------------------
Model:
    nvidia/parakeet-tdt-0.6b-v3

Runtime:
    Hugging Face Transformers / PyTorch

Dataset:
    MSLT German test
    2,275 utterances

Reference:
    t1_reference

Metrics:
    Corpus-level WER and CER

Normalization:
    Same normalization used by the primary W0/W1/W2 evaluation:
        - remove <...> tags
        - remove Unicode punctuation
        - normalize whitespace
        - lowercase

Features:
    - --limit N from terminal
    - resumable JSONL predictions
    - saves each prediction immediately
    - reports inference time
    - final corpus-level WER/CER
    - handles interrupted runs

IMPORTANT:
    This script performs genuine Parakeet inference through
    Hugging Face Transformers.

    It does NOT use Parakeet.cpp/GGUF predictions.
"""


# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

from jiwer import wer, cer

from transformers import (
    AutoProcessor,
    AutoModelForTDT,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

TEST_PARQUET = (
    "datasets/cache/mslt/de_en/test/"
    "mslt_de_asr_test.parquet"
)

OUTPUT_JSONL = (
    "results/parakeet_hf_mslt_test_predictions.jsonl"
)

DEVICE = "cpu"

MAX_NEW_TOKENS = 256


# ============================================================
# MSLT NORMALIZATION
# ============================================================

def normalize_mslt_t1(text):
    """
    Normalize MSLT T1 reference/hypothesis.

    Protocol:
        1. Remove XML-style annotation tags such as <SPN/>
        2. Remove punctuation using Unicode categories
        3. Normalize whitespace
        4. Lowercase
    """

    text = str(text)

    # --------------------------------------------------------
    # Remove tags such as:
    # <SPN/>
    # <NON/>
    # <UNIN/>
    # <LM>...</LM>
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Remove Unicode punctuation
    # --------------------------------------------------------

    text = "".join(
        ch
        for ch in text
        if not unicodedata.category(ch).startswith("P")
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    text = text.lower()

    return text


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NVIDIA Parakeet TDT v3 through "
            "Hugging Face/PyTorch on MSLT German test."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of samples from the beginning "
            "of the MSLT test set to evaluate. "
            "If omitted, evaluate the complete test set."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_JSONL,
        help=(
            "Path to JSONL prediction output."
        ),
    )

    return parser.parse_args()


# ============================================================
# LOAD EXISTING PREDICTIONS
# ============================================================

def load_completed_predictions(path):
    """
    Load predictions that already exist.

    This allows the evaluator to resume after interruption.
    """

    completed = {}

    if not path.exists():
        return completed

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

                sample_id = str(
                    row["sample_id"]
                )

                completed[sample_id] = row

            except Exception:
                # Ignore malformed/incomplete lines.
                continue

    return completed


# ============================================================
# SAVE ONE PREDICTION
# ============================================================

def append_prediction(path, row):
    """
    Append one result immediately.

    The file is flushed and closed after every prediction,
    making the evaluation resumable.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

def calculate_metrics(predictions):
    """
    Calculate corpus-level WER and CER.

    References and hypotheses are normalized using
    normalize_mslt_t1().
    """

    references = []
    hypotheses = []

    for row in predictions:

        reference = normalize_mslt_t1(
            row["reference"]
        )

        hypothesis = normalize_mslt_t1(
            row["prediction"]
        )

        references.append(reference)
        hypotheses.append(hypothesis)

    corpus_wer = wer(
        references,
        hypotheses,
    )

    corpus_cer = cer(
        references,
        hypotheses,
    )

    exact = sum(
        ref == hyp
        for ref, hyp in zip(
            references,
            hypotheses,
        )
    )

    return {
        "wer": corpus_wer,
        "cer": corpus_cer,
        "exact": exact,
        "total": len(predictions),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    output_path = Path(
        args.output
    )

    print()
    print("=" * 70)
    print(
        "Parakeet TDT v3 — "
        "Hugging Face / PyTorch MSLT Evaluation"
    )
    print("=" * 70)

    # ========================================================
    # DEVICE
    # ========================================================

    print()
    print(f"Device: {DEVICE}")
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("Loading MSLT test set...")

    df = pd.read_parquet(
        TEST_PARQUET
    )

    total_dataset_samples = len(df)

    # --------------------------------------------------------
    # Apply terminal --limit
    # --------------------------------------------------------

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        df = df.head(
            args.limit
        )

    print(
        f"Total samples in MSLT test: "
        f"{total_dataset_samples}"
    )

    print(
        f"Samples selected for this run: "
        f"{len(df)}"
    )

    # ========================================================
    # LOAD EXISTING RESULTS
    # ========================================================

    completed = load_completed_predictions(
        output_path
    )

    print(
        f"Existing predictions found: "
        f"{len(completed)}"
    )

    # --------------------------------------------------------
    # Only process samples selected by this invocation
    # --------------------------------------------------------

    selected_ids = {
        str(row["id"])
        for _, row in df.iterrows()
    }

    remaining = [
        row
        for _, row in df.iterrows()
        if str(row["id"]) not in completed
    ]

    print(
        f"Remaining samples to process: "
        f"{len(remaining)}"
    )

    # ========================================================
    # IF NOTHING REMAINS
    # ========================================================

    if len(remaining) == 0:

        print()
        print(
            "All selected samples already have "
            "saved predictions."
        )

    # ========================================================
    # LOAD PROCESSOR
    # ========================================================

    print()
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading Parakeet model...")
    print(
        "This may take a while on CPU."
    )

    model = AutoModelForTDT.from_pretrained(
        MODEL_ID,
        dtype="auto",
    )

    model.to(
        DEVICE
    )

    model.eval()

    print()
    print("Model loaded.")

    print(
        "Model device:",
        next(model.parameters()).device,
    )

    # ========================================================
    # SAMPLE RATE
    # ========================================================

    sample_rate = (
        processor
        .feature_extractor
        .sampling_rate
    )

    print(
        f"Expected sample rate: "
        f"{sample_rate} Hz"
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = list(
        completed.values()
    )

    evaluation_start = time.time()

    number_remaining = len(
        remaining
    )

    for position, row in enumerate(
        remaining,
        start=1,
    ):

        sample_id = str(
            row["id"]
        )

        audio_path = str(
            row["audio_path"]
        )

        reference = str(
            row["t1_reference"]
        )

        print()
        print("-" * 70)

        print(
            f"[{position}/{number_remaining}] "
            f"Sample {sample_id}"
        )

        print(
            f"Audio: {audio_path}"
        )

        # ====================================================
        # LOAD AUDIO
        # ====================================================

        try:

            audio, sr = sf.read(
                audio_path
            )

            # ------------------------------------------------
            # Convert stereo -> mono if necessary
            # ------------------------------------------------

            if len(audio.shape) > 1:

                audio = audio.mean(
                    axis=1
                )

            # ------------------------------------------------
            # Verify sample rate
            # ------------------------------------------------

            if sr != sample_rate:

                raise ValueError(
                    f"Unexpected sample rate "
                    f"{sr}; expected "
                    f"{sample_rate}"
                )

            # =================================================
            # PROCESS AUDIO
            # =================================================

            inputs = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
            )

            # ------------------------------------------------
            # Move tensors to CPU/device
            # ------------------------------------------------

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            # =================================================
            # INFERENCE
            # =================================================

            inference_start = time.time()

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

            inference_seconds = (
                time.time()
                - inference_start
            )

            # =================================================
            # DECODE
            # =================================================

            prediction = processor.decode(
                outputs.sequences[0],
                skip_special_tokens=True,
            )

            prediction = str(
                prediction
            ).strip()

            # =================================================
            # DISPLAY
            # =================================================

            print(
                f"Reference:  {reference}"
            )

            print(
                f"Prediction: {prediction}"
            )

            print(
                f"Inference:  "
                f"{inference_seconds:.2f} sec"
            )

            # =================================================
            # NORMALIZED VALUES
            # =================================================

            normalized_reference = (
                normalize_mslt_t1(
                    reference
                )
            )

            normalized_prediction = (
                normalize_mslt_t1(
                    prediction
                )
            )

            # =================================================
            # SAVE RESULT
            # =================================================

            result = {
                "sample_id": sample_id,
                "audio_path": audio_path,
                "reference": reference,
                "prediction": prediction,
                "normalized_reference":
                    normalized_reference,
                "normalized_prediction":
                    normalized_prediction,
                "inference_seconds":
                    inference_seconds,
                "model": MODEL_ID,
                "runtime":
                    "huggingface_pytorch",
            }

            append_prediction(
                output_path,
                result,
            )

            predictions.append(
                result
            )

        except Exception as e:

            # =================================================
            # ERROR HANDLING
            # =================================================

            print()
            print(
                f"ERROR on sample "
                f"{sample_id}: {e}"
            )

            error_result = {
                "sample_id": sample_id,
                "audio_path": audio_path,
                "reference": reference,
                "prediction": "",
                "error": str(e),
                "model": MODEL_ID,
                "runtime":
                    "huggingface_pytorch",
            }

            append_prediction(
                output_path,
                error_result,
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    successful = [
        row
        for row in predictions
        if "error" not in row
    ]

    failed = [
        row
        for row in predictions
        if "error" in row
    ]

    if len(successful) > 0:

        metrics = calculate_metrics(
            successful
        )

    else:

        metrics = {
            "wer": None,
            "cer": None,
            "exact": 0,
            "total": 0,
        }

    total_time = (
        time.time()
        - evaluation_start
    )

    # ========================================================
    # PRINT FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Selected samples: "
        f"{len(df)}"
    )

    print(
        f"Successful predictions: "
        f"{len(successful)}"
    )

    print(
        f"Failed predictions: "
        f"{len(failed)}"
    )

    if metrics["wer"] is not None:

        print(
            f"Exact matches: "
            f"{metrics['exact']}"
        )

        print(
            f"WER: "
            f"{metrics['wer'] * 100:.2f}%"
        )

        print(
            f"CER: "
            f"{metrics['cer'] * 100:.2f}%"
        )

    print(
        f"Evaluation time: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        f"Predictions saved to:"
    )

    print(
        output_path
    )

    # ========================================================
    # PROTOCOL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION PROTOCOL")
    print("=" * 70)

    print(
        "Model:      "
        "nvidia/parakeet-tdt-0.6b-v3"
    )

    print(
        "Runtime:    "
        "Hugging Face / PyTorch"
    )

    print(
        "Dataset:    "
        "MSLT German test"
    )

    print(
        "Reference:  "
        "t1_reference"
    )

    print(
        "Metrics:    "
        "Corpus-level WER / CER"
    )

    print(
        "Normalize:  "
        "normalize_mslt_t1"
    )

    print(
        "Teacher:    "
        "Direct HF Parakeet inference"
    )

    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()