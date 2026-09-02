#!/usr/bin/env python3
"""
BridgeDEUX: MSLT-100 Desktop End-to-End Evaluation
===================================================

Uses the SAME 100 MSLT samples and evaluates:

1. Whisper.cpp base
   German audio -> Whisper.cpp -> Marian FP32/INT8

2. W1 student
   German audio -> W1 -> Marian FP32/INT8

3. Clean reference condition
   Gold German -> Marian FP32/INT8

Metrics:
    - Whisper corpus-level WER/CER
    - W1 corpus-level WER/CER
    - sentence-level WER/CER
    - sentence-level chrF++
    - corpus-level chrF++
    - FP32/INT8 behavioral divergence
    - chrF++ DiD
    - COMET: clean + Whisper + W1
    - COMET DiD
    - 10,000-iteration paired bootstrap

IMPORTANT:
    Whisper.cpp is NOT rerun when the existing 100-sample result
    parquet is valid.

WER/CER FIX:
    Both Whisper.cpp and W1 are now evaluated using:
        - the same text normalization
        - the same JiWER implementation
        - corpus-level WER/CER for headline reporting

    Existing asr_wer/asr_cer values are retained only as legacy
    columns and are NOT used for the headline Whisper WER/CER.
"""

from __future__ import annotations

import logging
import random
import re
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import soundfile as sf
import torch

from comet import download_model, load_from_checkpoint
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from tqdm import tqdm
from transformers import (
    MarianTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from models.asr.whisper_cpp import WhisperCppASR


warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 16
BOOTSTRAP_ITERATIONS = 10_000

FP32_DIR_NAME = "opus_mt_de_en_opt_extended"
INT8_DIR_NAME = "opus_mt_de_en_opt_extended_int8"

COMET_MODEL_NAME = "Unbabel/wmt20-comet-da"

W1_MODEL_DIR_NAME = "w1_merged"
W1_NUM_BEAMS = 5

SUBSET_PATH = Path(
    "datasets/cache/mslt_benchmark_subset_100.parquet"
)

AUDIO_DIR = Path(
    "analysis/mobile_benchmark_audio_100/mslt"
)

EXISTING_WHISPER_RESULTS = Path(
    "analysis/mslt_100_desktop/mslt_100_desktop_results.parquet"
)

OUTPUT_DIR = Path(
    "analysis/mslt_100_desktop"
)

# Keep the existing Whisper cache/result filename unchanged.
WHISPER_OUTPUT_PARQUET = (
    OUTPUT_DIR / "mslt_100_desktop_results.parquet"
)

WHISPER_OUTPUT_CSV = (
    OUTPUT_DIR / "mslt_100_desktop_results.csv"
)

# W1 gets its own result files in the same directory.
OUTPUT_PARQUET = (
    OUTPUT_DIR / "mslt_100_desktop_w1_results.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR / "mslt_100_desktop_w1_results.csv"
)


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

def normalize_asr_text(text: str) -> str:
    """
    Common ASR normalization used for BOTH Whisper.cpp and W1.

    Removes XML/annotation tags, removes Unicode punctuation,
    collapses whitespace, and lowercases.
    """

    text = re.sub(
        r"<[^>]*>",
        " ",
        str(text),
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.category(c).startswith("P")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


# ---------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------

def _edit_distance(
    ref_tokens: list[str],
    hyp_tokens: list[str],
) -> int:

    prev = list(
        range(
            len(hyp_tokens) + 1
        )
    )

    for i, ref_token in enumerate(
        ref_tokens,
        start=1,
    ):

        cur = [i]

        for j, hyp_token in enumerate(
            hyp_tokens,
            start=1,
        ):

            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1]
                    + (ref_token != hyp_token),
                )
            )

        prev = cur

    return prev[-1]


def wer(
    ref: str,
    hyp: str,
) -> float:

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    if not ref_tokens:
        return (
            0.0
            if not hyp_tokens
            else 1.0
        )

    return (
        _edit_distance(
            ref_tokens,
            hyp_tokens,
        )
        / len(ref_tokens)
    )


def cer(
    ref: str,
    hyp: str,
) -> float:

    ref_chars = list(ref)
    hyp_chars = list(hyp)

    if not ref_chars:
        return (
            0.0
            if not hyp_chars
            else 1.0
        )

    return (
        _edit_distance(
            ref_chars,
            hyp_chars,
        )
        / len(ref_chars)
    )


def corpus_asr_metrics(
    references: list[str],
    hypotheses: list[str],
) -> tuple[float, float]:
    """
    Calculates corpus-level WER and CER using JiWER.

    This function is used identically for Whisper.cpp and W1.
    """

    refs = [
        normalize_asr_text(x)
        for x in references
    ]

    hyps = [
        normalize_asr_text(x)
        for x in hypotheses
    ]

    return (
        jiwer_wer(
            refs,
            hyps,
        ),
        jiwer_cer(
            refs,
            hyps,
        ),
    )


# ---------------------------------------------------------------------
# chrF++
# ---------------------------------------------------------------------

def corpus_chrfpp(
    hyps: list[str],
    refs: list[str],
) -> float:

    return sacrebleu.corpus_chrf(
        hyps,
        [refs],
        word_order=2,
    ).score


def sentence_chrfpp(
    hyp: str,
    ref: str,
) -> float:

    return sacrebleu.sentence_chrf(
        hyp,
        [ref],
        word_order=2,
    ).score


# ---------------------------------------------------------------------
# W1
# ---------------------------------------------------------------------

def load_w1_model(
    repo_root: Path,
):

    model_dir = (
        repo_root
        / "experiments"
        / "checkpoints"
        / W1_MODEL_DIR_NAME
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"W1 checkpoint not found: {model_dir}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nLoading W1 student model...")
    print(f"Model  : {model_dir}")
    print(f"Device : {device}")
    print(f"Beams  : {W1_NUM_BEAMS}")

    processor = (
        WhisperProcessor.from_pretrained(
            model_dir,
            language="german",
            task="transcribe",
        )
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(model_dir)
    )

    model.to(device)
    model.eval()

    return (
        processor,
        model,
        device,
    )


@torch.no_grad()
def transcribe_w1(
    processor,
    model,
    device: str,
    audio_path: str,
) -> str:

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16 kHz audio, "
            f"got {sample_rate} Hz: "
            f"{audio_path}"
        )

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    generated_ids = model.generate(
        input_features=inputs.input_features.to(
            device
        ),
        language="de",
        task="transcribe",
        num_beams=W1_NUM_BEAMS,
        do_sample=False,
        temperature=0.0,
        return_timestamps=False,
    )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    repo_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    subset_path = (
        repo_root / SUBSET_PATH
    )

    audio_dir = (
        repo_root / AUDIO_DIR
    )

    existing_whisper_path = (
        repo_root
        / EXISTING_WHISPER_RESULTS
    )

    output_dir = (
        repo_root / OUTPUT_DIR
    )

    output_parquet = (
        repo_root / OUTPUT_PARQUET
    )

    output_csv = (
        repo_root / OUTPUT_CSV
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(SEED)
    random.seed(SEED)

    # ---------------------------------------------------------------
    # Validate subset
    # ---------------------------------------------------------------

    if not subset_path.exists():
        raise FileNotFoundError(
            f"MSLT subset not found: "
            f"{subset_path}"
        )

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"MSLT audio directory not found: "
            f"{audio_dir}"
        )

    df = pd.read_parquet(
        subset_path
    ).copy()

    required = {
        "id",
        "source_text",
        "target_text",
        "client_id",
        "file_name",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Subset is missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 100:
        raise ValueError(
            f"Expected exactly 100 MSLT "
            f"samples, found {len(df)}"
        )

    if df["id"].nunique() != 100:
        raise ValueError(
            "MSLT subset contains duplicate IDs"
        )

    # ---------------------------------------------------------------
    # Construct common cohort
    # ---------------------------------------------------------------

    audio_paths = []

    for _, row in df.iterrows():

        audio_path = (
            audio_dir
            / Path(
                row["file_name"]
            ).name
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing audio for sample "
                f"{row['id']}: {audio_path}"
            )

        audio_paths.append(
            audio_path
        )

    df["sample_id"] = (
        df["id"].astype(str)
    )

    df["gold_german_source"] = (
        df["source_text"]
        .fillna("")
        .astype(str)
    )

    df["gold_english_reference"] = (
        df["target_text"]
        .fillna("")
        .astype(str)
    )

    df["audio_path"] = [
        str(p)
        for p in audio_paths
    ]

    print("=" * 80)
    print(
        " MSLT-100 DESKTOP — WHISPER + W1"
    )
    print("=" * 80)
    print(
        f"Samples : {len(df)}"
    )
    print(
        f"Audio   : {audio_dir}"
    )
    print(
        f"Seed    : {SEED}"
    )
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Recover existing Whisper.cpp results
    # ---------------------------------------------------------------

    whisper_recovered = False

    if existing_whisper_path.exists():

        print(
            "\n[1/5] Loading existing "
            "Whisper.cpp MSLT-100 results..."
        )

        old = pd.read_parquet(
            existing_whisper_path
        )

        required_whisper = {
            "sample_id",
            "whisper_hypothesis",
            "asr_inference_time_ms",
            "clean_fp32_translation",
            "clean_int8_translation",
            "asr_fp32_translation",
            "asr_int8_translation",
        }

        missing_whisper = (
            required_whisper
            - set(old.columns)
        )

        if not missing_whisper:

            old_ids = set(
                old["sample_id"]
                .astype(str)
            )

            current_ids = set(
                df["sample_id"]
                .astype(str)
            )

            if (
                len(old) == 100
                and old_ids == current_ids
            ):

                whisper_cols = [
                    c
                    for c in old.columns
                    if c not in {
                        "id",
                        "source_text",
                        "target_text",
                        "client_id",
                        "file_name",
                        "sample_id",
                        "gold_german_source",
                        "gold_english_reference",
                        "audio_path",
                    }
                ]

                old_merge = old[
                    ["sample_id"]
                    + whisper_cols
                ].copy()

                df = df.merge(
                    old_merge,
                    on="sample_id",
                    how="left",
                    validate="one_to_one",
                )

                whisper_recovered = True

                print(
                    "Existing Whisper.cpp results "
                    "verified: 100/100 samples."
                )

    # ---------------------------------------------------------------
    # Fallback: run Whisper.cpp
    # ---------------------------------------------------------------

    if not whisper_recovered:

        print(
            "\n[1/5] Existing Whisper.cpp "
            "results not usable."
        )

        print(
            "Running Whisper.cpp..."
        )

        asr = WhisperCppASR(
            model_size="base",
            n_threads=4,
        )

        asr.load()

        hypotheses = []
        asr_times_ms = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Whisper.cpp",
        ):

            audio, sample_rate = sf.read(
                row["audio_path"],
                dtype="float32",
            )

            if audio.ndim > 1:
                audio = np.mean(
                    audio,
                    axis=1,
                )

            if sample_rate != 16000:
                raise ValueError(
                    f"Expected 16 kHz audio, "
                    f"got {sample_rate} Hz"
                )

            result = asr.transcribe(
                audio,
                language="de",
            )

            hypotheses.append(
                result.transcription
            )

            asr_times_ms.append(
                float(
                    result.generation_time_ms
                )
            )

        df["whisper_hypothesis"] = (
            hypotheses
        )

        df["asr_inference_time_ms"] = (
            asr_times_ms
        )

    # ---------------------------------------------------------------
    # WER/CER FIX
    # ---------------------------------------------------------------
    # Calculate BOTH Whisper and W1 using exactly the same method.
    #
    # This is deliberately done BEFORE the W1 block so that Whisper's
    # metrics are recalculated even when its old parquet is recovered.
    # ---------------------------------------------------------------

    print(
        "\n[ASR METRICS] Recalculating "
        "corpus-level WER/CER..."
    )

    whisper_corpus_wer, whisper_corpus_cer = (
        corpus_asr_metrics(
            df["gold_german_source"].tolist(),
            df["whisper_hypothesis"].tolist(),
        )
    )

    # Preserve the old sentence-level fields if they existed,
    # but create explicitly named sentence-level metrics so they
    # cannot be confused with corpus-level WER/CER.
    whisper_refs_normalized = [
        normalize_asr_text(x)
        for x in df["gold_german_source"]
    ]

    whisper_preds_normalized = [
        normalize_asr_text(x)
        for x in df["whisper_hypothesis"]
    ]

    df["whisper_sentence_wer"] = [
        jiwer_wer(
            [r],
            [h],
        )
        for r, h in zip(
            whisper_refs_normalized,
            whisper_preds_normalized,
        )
    ]

    df["whisper_sentence_cer"] = [
        jiwer_cer(
            [r],
            [h],
        )
        for r, h in zip(
            whisper_refs_normalized,
            whisper_preds_normalized,
        )
    ]

    # These are the CORPUS metrics and replace the old ambiguous
    # asr_wer/asr_cer values in the final artifact.
    df["asr_wer"] = whisper_corpus_wer
    df["asr_cer"] = whisper_corpus_cer

    # ---------------------------------------------------------------
    # Save Whisper-only results separately
    # ---------------------------------------------------------------
    whisper_output_parquet = repo_root / WHISPER_OUTPUT_PARQUET
    whisper_output_csv = repo_root / WHISPER_OUTPUT_CSV

    whisper_result_columns = [
        "sample_id",
        "gold_german_source",
        "gold_english_reference",
        "audio_path",
        "whisper_hypothesis",
        "asr_inference_time_ms",
        "asr_wer",
        "asr_cer",
        "clean_fp32_translation",
        "clean_int8_translation",
        "asr_fp32_translation",
        "asr_int8_translation",
    ]

    whisper_result_columns = [
        c for c in whisper_result_columns if c in df.columns
    ]

    df[whisper_result_columns].to_parquet(
        whisper_output_parquet,
        index=False,
    )

    df[whisper_result_columns].to_csv(
        whisper_output_csv,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Whisper-only parquet       : {whisper_output_parquet}"
    )
    print(
        f"Whisper-only CSV           : {whisper_output_csv}"
    )

    # ---------------------------------------------------------------
    # 2. W1 student ASR
    # ---------------------------------------------------------------

    print(
        "\n[2/5] Running W1 student ASR "
        "on the same 100 samples..."
    )

    (
        w1_processor,
        w1_model,
        w1_device,
    ) = load_w1_model(
        repo_root
    )

    w1_hypotheses = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="W1",
    ):

        w1_hypotheses.append(
            transcribe_w1(
                w1_processor,
                w1_model,
                w1_device,
                row["audio_path"],
            )
        )

    df["w1_hypothesis"] = (
        w1_hypotheses
    )

    w1_corpus_wer, w1_corpus_cer = (
        corpus_asr_metrics(
            df["gold_german_source"].tolist(),
            df["w1_hypothesis"].tolist(),
        )
    )

    w1_refs_normalized = [
        normalize_asr_text(x)
        for x in df["gold_german_source"]
    ]

    w1_preds_normalized = [
        normalize_asr_text(x)
        for x in df["w1_hypothesis"]
    ]

    df["w1_asr_wer"] = [
        jiwer_wer(
            [r],
            [h],
        )
        for r, h in zip(
            w1_refs_normalized,
            w1_preds_normalized,
        )
    ]

    df["w1_asr_cer"] = [
        jiwer_cer(
            [r],
            [h],
        )
        for r, h in zip(
            w1_refs_normalized,
            w1_preds_normalized,
        )
    ]

    print(
        f"Whisper WER: "
        f"{whisper_corpus_wer:.4f} "
        f"({whisper_corpus_wer * 100:.2f}%)"
    )

    print(
        f"Whisper CER: "
        f"{whisper_corpus_cer:.4f} "
        f"({whisper_corpus_cer * 100:.2f}%)"
    )

    print(
        f"W1 WER: "
        f"{w1_corpus_wer:.4f} "
        f"({w1_corpus_wer * 100:.2f}%)"
    )

    print(
        f"W1 CER: "
        f"{w1_corpus_cer:.4f} "
        f"({w1_corpus_cer * 100:.2f}%)"
    )

    # ---------------------------------------------------------------
    # 3. Marian FP32 + INT8
    # ---------------------------------------------------------------

    print(
        "\n[3/5] Loading Marian FP32 and INT8..."
    )

    fp32_dir = (
        repo_root
        / "models"
        / "onnx"
        / FP32_DIR_NAME
    )

    int8_dir = (
        repo_root
        / "models"
        / "onnx"
        / INT8_DIR_NAME
    )

    if not fp32_dir.exists():
        raise FileNotFoundError(
            f"FP32 model not found: {fp32_dir}"
        )

    if not int8_dir.exists():
        raise FileNotFoundError(
            f"INT8 model not found: {int8_dir}"
        )

    tokenizer = (
        MarianTokenizer.from_pretrained(
            fp32_dir
        )
    )

    fp32_model = (
        ORTModelForSeq2SeqLM.from_pretrained(
            fp32_dir,
            provider="CPUExecutionProvider",
        )
    )

    int8_model = (
        ORTModelForSeq2SeqLM.from_pretrained(
            int8_dir,
            provider="CPUExecutionProvider",
        )
    )

    clean_fp32 = []
    clean_int8 = []

    asr_fp32 = []
    asr_int8 = []

    w1_asr_fp32 = []
    w1_asr_int8 = []

    for start in tqdm(
        range(
            0,
            len(df),
            BATCH_SIZE,
        ),
        desc="Marian",
    ):

        batch = df.iloc[
            start : start + BATCH_SIZE
        ]

        clean_text = [
            x if str(x).strip() else "."
            for x in batch[
                "gold_german_source"
            ].tolist()
        ]

        whisper_text = [
            x if str(x).strip() else "."
            for x in batch[
                "whisper_hypothesis"
            ].tolist()
        ]

        w1_text = [
            x if str(x).strip() else "."
            for x in batch[
                "w1_hypothesis"
            ].tolist()
        ]

        clean_inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        whisper_inputs = tokenizer(
            whisper_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        w1_inputs = tokenizer(
            w1_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        clean_f = fp32_model.generate(
            **clean_inputs,
            max_length=150,
        )

        clean_i = int8_model.generate(
            **clean_inputs,
            max_length=150,
        )

        whisper_f = fp32_model.generate(
            **whisper_inputs,
            max_length=150,
        )

        whisper_i = int8_model.generate(
            **whisper_inputs,
            max_length=150,
        )

        w1_f = fp32_model.generate(
            **w1_inputs,
            max_length=150,
        )

        w1_i = int8_model.generate(
            **w1_inputs,
            max_length=150,
        )

        clean_fp32.extend(
            tokenizer.batch_decode(
                clean_f,
                skip_special_tokens=True,
            )
        )

        clean_int8.extend(
            tokenizer.batch_decode(
                clean_i,
                skip_special_tokens=True,
            )
        )

        asr_fp32.extend(
            tokenizer.batch_decode(
                whisper_f,
                skip_special_tokens=True,
            )
        )

        asr_int8.extend(
            tokenizer.batch_decode(
                whisper_i,
                skip_special_tokens=True,
            )
        )

        w1_asr_fp32.extend(
            tokenizer.batch_decode(
                w1_f,
                skip_special_tokens=True,
            )
        )

        w1_asr_int8.extend(
            tokenizer.batch_decode(
                w1_i,
                skip_special_tokens=True,
            )
        )

    df["clean_fp32_translation"] = (
        clean_fp32
    )

    df["clean_int8_translation"] = (
        clean_int8
    )

    df["asr_fp32_translation"] = (
        asr_fp32
    )

    df["asr_int8_translation"] = (
        asr_int8
    )

    df["w1_asr_fp32_translation"] = (
        w1_asr_fp32
    )

    df["w1_asr_int8_translation"] = (
        w1_asr_int8
    )

    # ---------------------------------------------------------------
    # 4. chrF++
    # ---------------------------------------------------------------

    print(
        "\n[4/5] Computing chrF++ and DiD..."
    )

    refs = (
        df["gold_english_reference"]
        .tolist()
    )

    df["chrf_clean_f"] = [
        sentence_chrfpp(
            h,
            r,
        )
        for h, r in zip(
            df["clean_fp32_translation"],
            refs,
        )
    ]

    df["chrf_clean_i"] = [
        sentence_chrfpp(
            h,
            r,
        )
        for h, r in zip(
            df["clean_int8_translation"],
            refs,
        )
    ]

    df["chrf_asr_f"] = [
        sentence_chrfpp(
            h,
            r,
        )
        for h, r in zip(
            df["asr_fp32_translation"],
            refs,
        )
    ]

    df["chrf_asr_i"] = [
        sentence_chrfpp(
            h,
            r,
        )
        for h, r in zip(
            df["asr_int8_translation"],
            refs,
        )
    ]

    df["chrf_w1_asr_f"] = [
        sentence_chrfpp(
            h,
            r,
        )
        for h, r in zip(
            df["w1_asr_fp32_translation"],
            refs,
        )
    ]

    df["chrf_w1_asr_i"] = [
        sentence_chrfpp(
            h,
            r,
        )
        for h, r in zip(
            df["w1_asr_int8_translation"],
            refs,
        )
    ]

    # Clean quantization effect.
    df["delta_clean"] = (
        df["chrf_clean_i"]
        - df["chrf_clean_f"]
    )

    # Whisper quantization effect.
    df["delta_asr"] = (
        df["chrf_asr_i"]
        - df["chrf_asr_f"]
    )

    # Whisper DiD.
    df["did"] = (
        df["delta_asr"]
        - df["delta_clean"]
    )

    # W1 quantization effect.
    df["delta_w1_asr"] = (
        df["chrf_w1_asr_i"]
        - df["chrf_w1_asr_f"]
    )

    # W1 DiD.
    df["did_w1"] = (
        df["delta_w1_asr"]
        - df["delta_clean"]
    )

    # ---------------------------------------------------------------
    # 5. COMET
    # ---------------------------------------------------------------

    print(
        "\n[5/5] Running COMET..."
    )

    comet_path = download_model(
        COMET_MODEL_NAME
    )

    comet_model = load_from_checkpoint(
        comet_path
    )

    comet_model.eval()

    gold_src = (
        df["gold_german_source"]
        .tolist()
    )

    whisper_src = (
        df["whisper_hypothesis"]
        .tolist()
    )

    w1_src = (
        df["w1_hypothesis"]
        .tolist()
    )

    passes = {

        "comet_clean_f": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                gold_src,
                df["clean_fp32_translation"],
                refs,
            )
        ],

        "comet_clean_i": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                gold_src,
                df["clean_int8_translation"],
                refs,
            )
        ],

        "comet_asr_f_e2e": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                gold_src,
                df["asr_fp32_translation"],
                refs,
            )
        ],

        "comet_asr_i_e2e": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                gold_src,
                df["asr_int8_translation"],
                refs,
            )
        ],

        "comet_asr_f_mt": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                whisper_src,
                df["asr_fp32_translation"],
                refs,
            )
        ],

        "comet_asr_i_mt": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                whisper_src,
                df["asr_int8_translation"],
                refs,
            )
        ],

        "comet_w1_asr_f_e2e": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                gold_src,
                df["w1_asr_fp32_translation"],
                refs,
            )
        ],

        "comet_w1_asr_i_e2e": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                gold_src,
                df["w1_asr_int8_translation"],
                refs,
            )
        ],

        "comet_w1_asr_f_mt": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                w1_src,
                df["w1_asr_fp32_translation"],
                refs,
            )
        ],

        "comet_w1_asr_i_mt": [
            {
                "src": s,
                "mt": m,
                "ref": r,
            }
            for s, m, r in zip(
                w1_src,
                df["w1_asr_int8_translation"],
                refs,
            )
        ],
    }

    for col, data in passes.items():

        out = comet_model.predict(
            data,
            batch_size=BATCH_SIZE,
            gpus=0,
        )

        df[col] = out.scores

    # COMET quantization effects.

    df["comet_delta_clean"] = (
        df["comet_clean_i"]
        - df["comet_clean_f"]
    )

    df["comet_delta_asr"] = (
        df["comet_asr_i_e2e"]
        - df["comet_asr_f_e2e"]
    )

    df["comet_did"] = (
        df["comet_delta_asr"]
        - df["comet_delta_clean"]
    )

    df["comet_delta_w1_asr"] = (
        df["comet_w1_asr_i_e2e"]
        - df["comet_w1_asr_f_e2e"]
    )

    df["comet_did_w1"] = (
        df["comet_delta_w1_asr"]
        - df["comet_delta_clean"]
    )

    # ---------------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------------

    print(
        "\nRunning 10,000-iteration "
        "paired bootstrap..."
    )

    rng = np.random.default_rng(
        SEED
    )

    did_array = (
        df["did"].to_numpy()
    )

    did_w1_array = (
        df["did_w1"].to_numpy()
    )

    comet_did_array = (
        df["comet_did"].to_numpy()
    )

    comet_did_w1_array = (
        df["comet_did_w1"].to_numpy()
    )

    boot_chrf = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    boot_chrf_w1 = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    boot_comet = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    boot_comet_w1 = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    for b in range(
        BOOTSTRAP_ITERATIONS
    ):

        idx = rng.integers(
            0,
            len(df),
            size=len(df),
        )

        boot_chrf[b] = (
            did_array[idx].mean()
        )

        boot_chrf_w1[b] = (
            did_w1_array[idx].mean()
        )

        boot_comet[b] = (
            comet_did_array[idx].mean()
        )

        boot_comet_w1[b] = (
            comet_did_w1_array[idx].mean()
        )

    chrf_ci = np.percentile(
        boot_chrf,
        [2.5, 97.5],
    )

    chrf_w1_ci = np.percentile(
        boot_chrf_w1,
        [2.5, 97.5],
    )

    comet_ci = np.percentile(
        boot_comet,
        [2.5, 97.5],
    )

    comet_w1_ci = np.percentile(
        boot_comet_w1,
        [2.5, 97.5],
    )

    # ---------------------------------------------------------------
    # Corpus-level chrF++
    # ---------------------------------------------------------------

    corp_clean_f = corpus_chrfpp(
        df[
            "clean_fp32_translation"
        ].tolist(),
        refs,
    )

    corp_clean_i = corpus_chrfpp(
        df[
            "clean_int8_translation"
        ].tolist(),
        refs,
    )

    corp_asr_f = corpus_chrfpp(
        df[
            "asr_fp32_translation"
        ].tolist(),
        refs,
    )

    corp_asr_i = corpus_chrfpp(
        df[
            "asr_int8_translation"
        ].tolist(),
        refs,
    )

    corp_w1_f = corpus_chrfpp(
        df[
            "w1_asr_fp32_translation"
        ].tolist(),
        refs,
    )

    corp_w1_i = corpus_chrfpp(
        df[
            "w1_asr_int8_translation"
        ].tolist(),
        refs,
    )

    # ---------------------------------------------------------------
    # Divergence
    # ---------------------------------------------------------------

    clean_div = (
        df["clean_fp32_translation"]
        != df["clean_int8_translation"]
    ).mean() * 100

    whisper_div = (
        df["asr_fp32_translation"]
        != df["asr_int8_translation"]
    ).mean() * 100

    w1_div = (
        df["w1_asr_fp32_translation"]
        != df["w1_asr_int8_translation"]
    ).mean() * 100

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        " FINAL MSLT-100 DESKTOP RESULTS"
    )
    print("=" * 80)

    print(
        f"Samples                    : "
        f"{len(df)}"
    )

    # IMPORTANT:
    # These are now corpus-level metrics for BOTH ASR systems.
    print(
        f"Whisper WER                : "
        f"{whisper_corpus_wer:.4f} "
        f"({whisper_corpus_wer * 100:.2f}%)"
    )

    print(
        f"Whisper CER                : "
        f"{whisper_corpus_cer:.4f} "
        f"({whisper_corpus_cer * 100:.2f}%)"
    )

    print(
        f"W1 WER                     : "
        f"{w1_corpus_wer:.4f} "
        f"({w1_corpus_wer * 100:.2f}%)"
    )

    print(
        f"W1 CER                     : "
        f"{w1_corpus_cer:.4f} "
        f"({w1_corpus_cer * 100:.2f}%)"
    )

    print()
    print("Corpus chrF++:")

    print(
        f"  Clean -> FP32            : "
        f"{corp_clean_f:.3f}"
    )

    print(
        f"  Clean -> INT8            : "
        f"{corp_clean_i:.3f}"
    )

    print(
        f"  Whisper ASR -> FP32      : "
        f"{corp_asr_f:.3f}"
    )

    print(
        f"  Whisper ASR -> INT8      : "
        f"{corp_asr_i:.3f}"
    )

    print(
        f"  W1 ASR -> FP32           : "
        f"{corp_w1_f:.3f}"
    )

    print(
        f"  W1 ASR -> INT8           : "
        f"{corp_w1_i:.3f}"
    )

    print(
        f"  Clean Δ(INT8-FP32)       : "
        f"{corp_clean_i - corp_clean_f:+.3f}"
    )

    print(
        f"  Whisper Δ(INT8-FP32)     : "
        f"{corp_asr_i - corp_asr_f:+.3f}"
    )

    print(
        f"  W1 Δ(INT8-FP32)          : "
        f"{corp_w1_i - corp_w1_f:+.3f}"
    )

    print(
        f"  Whisper Corpus DiD        : "
        f"{df['did'].mean():+.3f}"
    )

    print(
        f"  W1 Corpus DiD             : "
        f"{df['did_w1'].mean():+.3f}"
    )

    print()
    print(
        "Behavioral divergence:"
    )

    print(
        f"  Clean FP32 vs INT8       : "
        f"{clean_div:.2f}%"
    )

    print(
        f"  Whisper FP32 vs INT8     : "
        f"{whisper_div:.2f}%"
    )

    print(
        f"  W1 FP32 vs INT8          : "
        f"{w1_div:.2f}%"
    )

    print()
    print(
        "chrF++ DiD bootstrap:"
    )

    print(
        f"  Whisper Mean DiD          : "
        f"{df['did'].mean():+.4f}"
    )

    print(
        f"  Whisper 95% CI            : "
        f"[{chrf_ci[0]:+.4f}, "
        f"{chrf_ci[1]:+.4f}]"
    )

    print(
        f"  W1 Mean DiD               : "
        f"{df['did_w1'].mean():+.4f}"
    )

    print(
        f"  W1 95% CI                 : "
        f"[{chrf_w1_ci[0]:+.4f}, "
        f"{chrf_w1_ci[1]:+.4f}]"
    )

    print()
    print("COMET:")

    print(
        f"  Clean FP32 mean           : "
        f"{df['comet_clean_f'].mean():.4f}"
    )

    print(
        f"  Clean INT8 mean           : "
        f"{df['comet_clean_i'].mean():.4f}"
    )

    print(
        f"  Whisper ASR FP32 E2E      : "
        f"{df['comet_asr_f_e2e'].mean():.4f}"
    )

    print(
        f"  Whisper ASR INT8 E2E      : "
        f"{df['comet_asr_i_e2e'].mean():.4f}"
    )

    print(
        f"  Whisper COMET DiD         : "
        f"{df['comet_did'].mean():+.4f}"
    )

    print(
        f"  Whisper 95% CI            : "
        f"[{comet_ci[0]:+.4f}, "
        f"{comet_ci[1]:+.4f}]"
    )

    print(
        f"  W1 ASR FP32 E2E           : "
        f"{df['comet_w1_asr_f_e2e'].mean():.4f}"
    )

    print(
        f"  W1 ASR INT8 E2E           : "
        f"{df['comet_w1_asr_i_e2e'].mean():+.4f}"
    )

    print(
        f"  W1 COMET DiD              : "
        f"{df['comet_did_w1'].mean():+.4f}"
    )

    print(
        f"  W1 95% CI                 : "
        f"[{comet_w1_ci[0]:+.4f}, "
        f"{comet_w1_ci[1]:+.4f}]"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    # Save W1 results separately. Do not include Whisper ASR columns.
    w1_result_columns = [
        "sample_id",
        "gold_german_source",
        "gold_english_reference",
        "audio_path",
        "w1_hypothesis",
        "w1_asr_wer",
        "w1_asr_cer",
        "clean_fp32_translation",
        "clean_int8_translation",
        "w1_asr_fp32_translation",
        "w1_asr_int8_translation",
        "chrf_clean_f",
        "chrf_clean_i",
        "chrf_w1_asr_f",
        "chrf_w1_asr_i",
        "delta_clean",
        "delta_w1_asr",
        "did_w1",
        "comet_clean_f",
        "comet_clean_i",
        "comet_w1_asr_f_e2e",
        "comet_w1_asr_i_e2e",
        "comet_w1_asr_f_mt",
        "comet_w1_asr_i_mt",
        "comet_delta_clean",
        "comet_delta_w1_asr",
        "comet_did_w1",
    ]

    w1_result_columns = [
        c for c in w1_result_columns if c in df.columns
    ]

    df[w1_result_columns].to_parquet(
        output_parquet,
        index=False,
    )

    df[w1_result_columns].to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"W1 Parquet                 : "
        f"{output_parquet}"
    )

    print(
        f"W1 CSV                     : "
        f"{output_csv}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()