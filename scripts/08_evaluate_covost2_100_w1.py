#!/usr/bin/env python3
"""
BridgeDEUX: CoVoST2-100 Desktop W1 Evaluation
==============================================

Exact 100 CoVoST2 samples used by the mobile benchmark.

Cohort:
    Mobile benchmark CSV
        -> exact 100 sample IDs

Audio:
    analysis/mobile_benchmark_audio/covost2/

References:
    Existing CoVoST2 TEST provider
        -> German source
        -> English target

Evaluation:
    WAV
      -> W1 student ASR
      -> German hypothesis
      -> Marian FP32
      -> Marian INT8

Also evaluates:
    Clean German
      -> Marian FP32
      -> Marian INT8

Metrics:
    - W1 corpus WER
    - W1 corpus CER
    - sentence WER/CER
    - corpus chrF++
    - sentence chrF++
    - FP32/INT8 behavioral divergence
    - Difference-in-Differences
    - 10,000-iteration paired bootstrap
    - COMET
    - COMET DiD
    - exact ID integrity

IMPORTANT:
    The mobile CSV is used ONLY for the exact 100 sample IDs.
    It is NOT used as the source of German or English references.
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
from jiwer import wer as jiwer_wer
from jiwer import cer as jiwer_cer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from tqdm import tqdm
from transformers import (
    MarianTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from datasets.providers.covost_provider import CoVoSTProvider


warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# =========================================================================
# CONFIGURATION
# =========================================================================

SEED = 42

BATCH_SIZE = 16
BOOTSTRAP_ITERATIONS = 10_000

W1_MODEL_DIR_NAME = "w1_merged"
W1_BEAMS = 5

FP32_DIR_NAME = "opus_mt_de_en_opt_extended"
INT8_DIR_NAME = "opus_mt_de_en_opt_extended_int8"

COMET_MODEL_NAME = "Unbabel/wmt20-comet-da"

MOBILE_CSV_DIR = (
    Path("analysis")
    / "Mobile Benchmark Results"
    / "covost2"
)

AUDIO_DIR = (
    Path("analysis")
    / "mobile_benchmark_audio_100"
    / "covost2"
)

OUTPUT_DIR = (
    Path("analysis")
    / "covost2_100_w1_desktop"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "covost2_100_w1_desktop_results.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "covost2_100_w1_desktop_results.csv"
)


# =========================================================================
# DETERMINISM
# =========================================================================

def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# =========================================================================
# TEXT NORMALIZATION
# =========================================================================

def normalize_asr_text(text: str) -> str:
    """
    Normalization used for WER/CER.

    - removes XML/HTML-style tags
    - removes punctuation
    - collapses whitespace
    - lowercases
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


# =========================================================================
# WER / CER
# =========================================================================

def corpus_wer_cer(
    references: list[str],
    hypotheses: list[str],
) -> tuple[float, float]:

    refs = [
        normalize_asr_text(x)
        for x in references
    ]

    hyps = [
        normalize_asr_text(x)
        for x in hypotheses
    ]

    return (
        float(
            jiwer_wer(
                refs,
                hyps,
            )
        ),
        float(
            jiwer_cer(
                refs,
                hyps,
            )
        ),
    )


def sentence_wer(
    reference: str,
    hypothesis: str,
) -> float:

    ref = normalize_asr_text(
        reference
    )

    hyp = normalize_asr_text(
        hypothesis
    )

    if not ref:
        return 0.0 if not hyp else 1.0

    return float(
        jiwer_wer(
            [ref],
            [hyp],
        )
    )


def sentence_cer(
    reference: str,
    hypothesis: str,
) -> float:

    ref = normalize_asr_text(
        reference
    )

    hyp = normalize_asr_text(
        hypothesis
    )

    if not ref:
        return 0.0 if not hyp else 1.0

    return float(
        jiwer_cer(
            [ref],
            [hyp],
        )
    )


# =========================================================================
# chrF++
# =========================================================================

def sentence_chrfpp(
    hypothesis: str,
    reference: str,
) -> float:

    return float(
        sacrebleu.sentence_chrf(
            hypothesis,
            [reference],
            word_order=2,
        ).score
    )


def corpus_chrfpp(
    hypotheses: list[str],
    references: list[str],
) -> float:

    return float(
        sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            word_order=2,
        ).score
    )


# =========================================================================
# MOBILE BENCHMARK — EXACT 100 IDS
# =========================================================================

def find_mobile_csv(
    repo_root: Path,
) -> Path:

    csv_dir = (
        repo_root
        / MOBILE_CSV_DIR
    )

    if not csv_dir.exists():
        raise FileNotFoundError(
            f"Mobile benchmark directory not found:\n"
            f"{csv_dir}"
        )

    candidates = sorted(
        csv_dir.glob("*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in:\n"
            f"{csv_dir}"
        )

    honor_candidates = [
        p
        for p in candidates
        if "Honor600" in p.name
    ]

    if len(honor_candidates) == 1:
        return honor_candidates[0]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify the CoVoST2 mobile "
        "benchmark CSV.\n\n"
        "Candidates:\n"
        + "\n".join(
            f"  {p.name}"
            for p in candidates
        )
    )


def load_exact_mobile_ids(
    repo_root: Path,
) -> list[str]:

    csv_path = find_mobile_csv(
        repo_root
    )

    print(
        "\nMobile benchmark CSV:"
    )

    print(
        f"  {csv_path}"
    )

    mobile_df = pd.read_csv(
        csv_path
    )

    if "sample_id" not in mobile_df.columns:
        raise ValueError(
            "Mobile benchmark CSV does not contain "
            "'sample_id'."
        )

    ids = (
        mobile_df[
            "sample_id"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(ids) != 100:
        raise ValueError(
            f"Expected exactly 100 mobile benchmark "
            f"rows, found {len(ids)}."
        )

    if len(set(ids)) != 100:
        raise ValueError(
            "Mobile benchmark contains duplicate "
            "sample IDs."
        )

    print(
        f"  Exact mobile IDs loaded: {len(ids)}"
    )

    return ids


# =========================================================================
# AUDIO VALIDATION
# =========================================================================

def validate_audio(
    repo_root: Path,
    expected_ids: list[str],
) -> dict[str, Path]:

    audio_dir = (
        repo_root
        / AUDIO_DIR
    )

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"CoVoST2 mobile audio directory not found:\n"
            f"{audio_dir}"
        )

    expected_set = set(
        expected_ids
    )

    audio_files = list(
        audio_dir.glob("*.wav")
    )

    actual_map = {
        p.stem.strip(): p
        for p in audio_files
    }

    actual_set = set(
        actual_map
    )

    missing = (
        expected_set
        - actual_set
    )

    extra = (
        actual_set
        - expected_set
    )

    print(
        "\nEXACT AUDIO-ID VALIDATION"
    )

    print(
        f"  Expected IDs : {len(expected_set)}"
    )

    print(
        f"  Audio IDs    : {len(actual_set)}"
    )

    print(
        f"  Missing IDs  : {len(missing)}"
    )

    print(
        f"  Extra IDs    : {len(extra)}"
    )

    print(
        f"  Exact match  : "
        f"{expected_set == actual_set}"
    )

    if missing:
        print(
            "\nMissing audio IDs:"
        )

        for sample_id in sorted(
            missing
        ):
            print(
                f"  {sample_id}"
            )

        raise ValueError(
            "The audio folder does not contain "
            "all 100 required sample IDs."
        )

    if extra:
        print(
            "\nUnexpected audio IDs:"
        )

        for sample_id in sorted(
            extra
        ):
            print(
                f"  {sample_id}"
            )

        raise ValueError(
            "The audio folder contains sample IDs "
            "outside the exact mobile cohort."
        )

    return {
        sample_id: actual_map[sample_id]
        for sample_id in expected_ids
    }


# =========================================================================
# COVOST2 TEST REFERENCES
# =========================================================================

def load_covost2_references(
    expected_ids: list[str],
) -> dict[str, dict[str, str]]:
    """
    Load German and English references from the existing
    CoVoST2 TEST provider.

    No training dataset is used.
    """

    expected_set = set(
        expected_ids
    )

    print(
        "\nLoading CoVoST2 TEST references..."
    )

    provider = CoVoSTProvider(
        split="test",
        include_audio=False,
    )

    references: dict[
        str,
        dict[str, str],
    ] = {}

    for sample in provider:

        sample_id = str(
            sample.id
        ).strip()

        if sample_id not in expected_set:
            continue

        if sample_id in references:
            raise ValueError(
                f"Duplicate CoVoST2 reference ID: "
                f"{sample_id}"
            )

        references[sample_id] = {
            "gold_german_source": str(
                sample.source_text
            ),
            "gold_english_reference": str(
                sample.target_text
            ),
        }

        if len(references) == 100:
            break

    missing = (
        expected_set
        - set(references)
    )

    print(
        f"  References found: "
        f"{len(references)}/100"
    )

    if missing:
        print(
            "\nMissing reference IDs:"
        )

        for sample_id in sorted(
            missing
        ):
            print(
                f"  {sample_id}"
            )

        raise ValueError(
            "Could not find CoVoST2 TEST references "
            "for all 100 required IDs."
        )

    return references


# =========================================================================
# BUILD EXACT COHORT
# =========================================================================

def build_cohort(
    repo_root: Path,
) -> pd.DataFrame:

    expected_ids = (
        load_exact_mobile_ids(
            repo_root
        )
    )

    audio_map = validate_audio(
        repo_root,
        expected_ids,
    )

    references = (
        load_covost2_references(
            expected_ids
        )
    )

    rows = []

    for sample_id in expected_ids:

        rows.append(
            {
                "sample_id": sample_id,

                "gold_german_source":
                    references[
                        sample_id
                    ][
                        "gold_german_source"
                    ],

                "gold_english_reference":
                    references[
                        sample_id
                    ][
                        "gold_english_reference"
                    ],

                "audio_path":
                    str(
                        audio_map[
                            sample_id
                        ]
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    if len(df) != 100:
        raise ValueError(
            f"Final cohort contains {len(df)} "
            f"samples, expected 100."
        )

    if (
        df["sample_id"]
        .nunique()
        != 100
    ):
        raise ValueError(
            "Final cohort contains duplicate IDs."
        )

    if (
        set(df["sample_id"])
        != set(expected_ids)
    ):
        raise ValueError(
            "Final cohort IDs do not exactly match "
            "the mobile benchmark IDs."
        )

    print(
        "\nFINAL COVOST2-100 COHORT VERIFIED"
    )

    print(
        "  Samples        : 100"
    )

    print(
        "  IDs             : exact mobile benchmark IDs"
    )

    print(
        "  Audio           : "
        "analysis/mobile_benchmark_audio/covost2"
    )

    print(
        "  References      : CoVoST2 TEST"
    )

    print(
        "  Training data   : NOT USED"
    )

    return df


# =========================================================================
# W1 MODEL
# =========================================================================

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
            f"W1 checkpoint not found:\n"
            f"{model_dir}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nLoading W1 student model..."
    )

    print(
        f"Model  : {model_dir}"
    )

    print(
        f"Device : {device}"
    )

    print(
        f"Beams  : {W1_BEAMS}"
    )

    processor = (
        WhisperProcessor.from_pretrained(
            model_dir,
            language="german",
            task="transcribe",
        )
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            model_dir
        )
    )

    model.to(device)
    model.eval()

    return (
        processor,
        model,
        device,
    )


@torch.no_grad()
def w1_transcribe(
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
            f"got {sample_rate} Hz:\n"
            f"{audio_path}"
        )

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    generated_ids = model.generate(
        input_features=(
            inputs.input_features.to(
                device
            )
        ),
        language="de",
        task="transcribe",
        num_beams=W1_BEAMS,
        do_sample=False,
        temperature=0.0,
        return_timestamps=False,
    )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


# =========================================================================
# MARIAN
# =========================================================================

def load_marian_models(
    repo_root: Path,
):

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
            f"Marian FP32 model not found:\n"
            f"{fp32_dir}"
        )

    if not int8_dir.exists():
        raise FileNotFoundError(
            f"Marian INT8 model not found:\n"
            f"{int8_dir}"
        )

    print(
        "\nLoading Marian FP32 and INT8..."
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

    return (
        tokenizer,
        fp32_model,
        int8_model,
    )


def translate_batch(
    tokenizer,
    model,
    texts: list[str],
) -> list[str]:

    safe_texts = [
        (
            str(text)
            if str(text).strip()
            else "."
        )
        for text in texts
    ]

    inputs = tokenizer(
        safe_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    outputs = model.generate(
        **inputs,
        max_length=150,
    )

    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )


# =========================================================================
# BOOTSTRAP
# =========================================================================

def bootstrap_ci(
    values: np.ndarray,
) -> tuple[float, float]:

    rng = np.random.default_rng(
        SEED
    )

    n = len(values)

    boot_means = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    for i in range(
        BOOTSTRAP_ITERATIONS
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means[i] = (
            values[indices].mean()
        )

    ci = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return (
        float(ci[0]),
        float(ci[1]),
    )


# =========================================================================
# COMET
# =========================================================================

def run_comet_pass(
    model,
    source: list[str],
    hypotheses: list[str],
    references: list[str],
) -> list[float]:

    data = [
        {
            "src": src,
            "mt": hyp,
            "ref": ref,
        }
        for src, hyp, ref in zip(
            source,
            hypotheses,
            references,
        )
    ]

    result = model.predict(
        data,
        batch_size=BATCH_SIZE,
        gpus=0,
    )

    return [
        float(x)
        for x in result.scores
    ]


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:

    repo_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    set_seed()

    output_dir = (
        repo_root
        / OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_parquet = (
        repo_root
        / OUTPUT_PARQUET
    )

    output_csv = (
        repo_root
        / OUTPUT_CSV
    )

    print("=" * 80)
    print(
        " COVOST2-100 DESKTOP — W1 -> MARIAN FP32/INT8"
    )
    print("=" * 80)

    # =====================================================================
    # 1. EXACT COHORT
    # =====================================================================

    print(
        "\n[1/6] Building exact 100-sample CoVoST2 cohort..."
    )

    df = build_cohort(
        repo_root
    )

    # =====================================================================
    # 2. W1 ASR
    # =====================================================================

    print(
        "\n[2/6] Running W1 student ASR..."
    )

    (
        processor,
        w1_model,
        w1_device,
    ) = load_w1_model(
        repo_root
    )

    w1_hypotheses = []
    w1_times_ms = []

    w1_start = time.perf_counter()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="W1",
    ):

        start = time.perf_counter()

        hypothesis = w1_transcribe(
            processor,
            w1_model,
            w1_device,
            row["audio_path"],
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        w1_hypotheses.append(
            hypothesis
        )

        w1_times_ms.append(
            elapsed_ms
        )

    df["w1_hypothesis"] = (
        w1_hypotheses
    )

    df["w1_inference_time_ms"] = (
        w1_times_ms
    )

    w1_corpus_wer, w1_corpus_cer = (
        corpus_wer_cer(
            df[
                "gold_german_source"
            ].tolist(),
            df[
                "w1_hypothesis"
            ].tolist(),
        )
    )

    df["w1_asr_wer"] = [
        sentence_wer(
            ref,
            hyp,
        )
        for ref, hyp in zip(
            df[
                "gold_german_source"
            ],
            df[
                "w1_hypothesis"
            ],
        )
    ]

    df["w1_asr_cer"] = [
        sentence_cer(
            ref,
            hyp,
        )
        for ref, hyp in zip(
            df[
                "gold_german_source"
            ],
            df[
                "w1_hypothesis"
            ],
        )
    ]

    print()
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

    print(
        f"W1 mean sentence WER: "
        f"{df['w1_asr_wer'].mean():.4f}"
    )

    print(
        f"W1 mean sentence CER: "
        f"{df['w1_asr_cer'].mean():.4f}"
    )

    # =====================================================================
    # 3. MARIAN
    # =====================================================================

    print(
        "\n[3/6] Loading Marian FP32 and INT8..."
    )

    (
        tokenizer,
        fp32_model,
        int8_model,
    ) = load_marian_models(
        repo_root
    )

    clean_fp32 = []
    clean_int8 = []

    w1_fp32 = []
    w1_int8 = []

    print(
        "Running clean-input and W1-ASR-input translation..."
    )

    for start in tqdm(
        range(
            0,
            len(df),
            BATCH_SIZE,
        ),
        desc="Marian",
    ):

        batch = df.iloc[
            start :
            start + BATCH_SIZE
        ]

        clean_texts = (
            batch[
                "gold_german_source"
            ]
            .tolist()
        )

        w1_texts = (
            batch[
                "w1_hypothesis"
            ]
            .tolist()
        )

        # -------------------------------------------------------------
        # Clean German -> FP32
        # -------------------------------------------------------------

        clean_fp32.extend(
            translate_batch(
                tokenizer,
                fp32_model,
                clean_texts,
            )
        )

        # -------------------------------------------------------------
        # Clean German -> INT8
        # -------------------------------------------------------------

        clean_int8.extend(
            translate_batch(
                tokenizer,
                int8_model,
                clean_texts,
            )
        )

        # -------------------------------------------------------------
        # W1 ASR -> FP32
        # -------------------------------------------------------------

        w1_fp32.extend(
            translate_batch(
                tokenizer,
                fp32_model,
                w1_texts,
            )
        )

        # -------------------------------------------------------------
        # W1 ASR -> INT8
        # -------------------------------------------------------------

        w1_int8.extend(
            translate_batch(
                tokenizer,
                int8_model,
                w1_texts,
            )
        )

    df[
        "clean_fp32_translation"
    ] = clean_fp32

    df[
        "clean_int8_translation"
    ] = clean_int8

    df[
        "w1_asr_fp32_translation"
    ] = w1_fp32

    df[
        "w1_asr_int8_translation"
    ] = w1_int8

    # =====================================================================
    # 4. chrF++ / DiD
    # =====================================================================

    print(
        "\n[4/6] Computing chrF++ and DiD..."
    )

    refs = (
        df[
            "gold_english_reference"
        ].tolist()
    )

    # -------------------------------------------------------------
    # Sentence-level chrF++
    # -------------------------------------------------------------

    df["chrf_clean_f"] = [
        sentence_chrfpp(
            hyp,
            ref,
        )
        for hyp, ref in zip(
            df[
                "clean_fp32_translation"
            ],
            refs,
        )
    ]

    df["chrf_clean_i"] = [
        sentence_chrfpp(
            hyp,
            ref,
        )
        for hyp, ref in zip(
            df[
                "clean_int8_translation"
            ],
            refs,
        )
    ]

    df["chrf_w1_asr_f"] = [
        sentence_chrfpp(
            hyp,
            ref,
        )
        for hyp, ref in zip(
            df[
                "w1_asr_fp32_translation"
            ],
            refs,
        )
    ]

    df["chrf_w1_asr_i"] = [
        sentence_chrfpp(
            hyp,
            ref,
        )
        for hyp, ref in zip(
            df[
                "w1_asr_int8_translation"
            ],
            refs,
        )
    ]

    # -------------------------------------------------------------
    # INT8 - FP32 deltas
    # -------------------------------------------------------------

    df["delta_clean"] = (
        df["chrf_clean_i"]
        -
        df["chrf_clean_f"]
    )

    df["delta_w1_asr"] = (
        df["chrf_w1_asr_i"]
        -
        df["chrf_w1_asr_f"]
    )

    # -------------------------------------------------------------
    # Difference-in-Differences
    # -------------------------------------------------------------

    df["did_w1"] = (
        df["delta_w1_asr"]
        -
        df["delta_clean"]
    )

    # -------------------------------------------------------------
    # Corpus-level chrF++
    # -------------------------------------------------------------

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

    clean_delta_corpus = (
        corp_clean_i
        -
        corp_clean_f
    )

    w1_delta_corpus = (
        corp_w1_i
        -
        corp_w1_f
    )

    corpus_did = (
        w1_delta_corpus
        -
        clean_delta_corpus
    )

    # -------------------------------------------------------------
    # Behavioral divergence
    # -------------------------------------------------------------

    clean_div = (
        df[
            "clean_fp32_translation"
        ]
        !=
        df[
            "clean_int8_translation"
        ]
    ).mean() * 100

    w1_div = (
        df[
            "w1_asr_fp32_translation"
        ]
        !=
        df[
            "w1_asr_int8_translation"
        ]
    ).mean() * 100

    # =====================================================================
    # 5. COMET
    # =====================================================================

    print(
        "\n[5/6] Running COMET..."
    )

    print(
        f"Loading '{COMET_MODEL_NAME}'..."
    )

    comet_path = download_model(
        COMET_MODEL_NAME
    )

    comet_model = (
        load_from_checkpoint(
            comet_path
        )
    )

    comet_model.eval()

    gold_src = (
        df[
            "gold_german_source"
        ].tolist()
    )

    w1_src = (
        df[
            "w1_hypothesis"
        ].tolist()
    )

    # -------------------------------------------------------------
    # Clean FP32
    # -------------------------------------------------------------

    df["comet_clean_f"] = (
        run_comet_pass(
            comet_model,
            gold_src,
            df[
                "clean_fp32_translation"
            ].tolist(),
            refs,
        )
    )

    # -------------------------------------------------------------
    # Clean INT8
    # -------------------------------------------------------------

    df["comet_clean_i"] = (
        run_comet_pass(
            comet_model,
            gold_src,
            df[
                "clean_int8_translation"
            ].tolist(),
            refs,
        )
    )

    # -------------------------------------------------------------
    # W1 ASR -> FP32 E2E
    # -------------------------------------------------------------

    df["comet_w1_asr_f_e2e"] = (
        run_comet_pass(
            comet_model,
            gold_src,
            df[
                "w1_asr_fp32_translation"
            ].tolist(),
            refs,
        )
    )

    # -------------------------------------------------------------
    # W1 ASR -> INT8 E2E
    # -------------------------------------------------------------

    df["comet_w1_asr_i_e2e"] = (
        run_comet_pass(
            comet_model,
            gold_src,
            df[
                "w1_asr_int8_translation"
            ].tolist(),
            refs,
        )
    )

    # -------------------------------------------------------------
    # W1 ASR -> FP32 MT-only
    # -------------------------------------------------------------

    df["comet_w1_asr_f_mt"] = (
        run_comet_pass(
            comet_model,
            w1_src,
            df[
                "w1_asr_fp32_translation"
            ].tolist(),
            refs,
        )
    )

    # -------------------------------------------------------------
    # W1 ASR -> INT8 MT-only
    # -------------------------------------------------------------

    df["comet_w1_asr_i_mt"] = (
        run_comet_pass(
            comet_model,
            w1_src,
            df[
                "w1_asr_int8_translation"
            ].tolist(),
            refs,
        )
    )

    # -------------------------------------------------------------
    # COMET DiD
    # -------------------------------------------------------------

    df["comet_delta_clean"] = (
        df["comet_clean_i"]
        -
        df["comet_clean_f"]
    )

    df["comet_delta_w1_asr"] = (
        df[
            "comet_w1_asr_i_e2e"
        ]
        -
        df[
            "comet_w1_asr_f_e2e"
        ]
    )

    df["comet_did_w1"] = (
        df["comet_delta_w1_asr"]
        -
        df["comet_delta_clean"]
    )

    # =====================================================================
    # 6. BOOTSTRAP + FINAL OUTPUT
    # =====================================================================

    print(
        "\n[6/6] Running 10,000-iteration paired bootstrap..."
    )

    chrf_ci = bootstrap_ci(
        df[
            "did_w1"
        ].to_numpy(
            dtype=np.float64
        )
    )

    comet_ci = bootstrap_ci(
        df[
            "comet_did_w1"
        ].to_numpy(
            dtype=np.float64
        )
    )

    # =====================================================================
    # FINAL RESULTS
    # =====================================================================

    print()
    print("=" * 80)
    print(
        " FINAL COVOST2-100 W1 DESKTOP RESULTS"
    )
    print("=" * 80)

    print(
        f"Samples                    : {len(df)}"
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
    print(
        "Corpus chrF++:"
    )

    print(
        f"  Clean -> FP32            : "
        f"{corp_clean_f:.3f}"
    )

    print(
        f"  Clean -> INT8            : "
        f"{corp_clean_i:.3f}"
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
        f"{clean_delta_corpus:+.3f}"
    )

    print(
        f"  W1 Δ(INT8-FP32)          : "
        f"{w1_delta_corpus:+.3f}"
    )

    print(
        f"  W1 Corpus DiD            : "
        f"{corpus_did:+.3f}"
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
        f"  W1 ASR FP32 vs INT8      : "
        f"{w1_div:.2f}%"
    )

    print()
    print(
        "chrF++ DiD bootstrap:"
    )

    print(
        f"  W1 Mean DiD              : "
        f"{df['did_w1'].mean():+.4f}"
    )

    print(
        f"  W1 95% CI                : "
        f"[{chrf_ci[0]:+.4f}, "
        f"{chrf_ci[1]:+.4f}]"
    )

    print()
    print(
        "COMET:"
    )

    print(
        f"  Clean FP32 mean          : "
        f"{df['comet_clean_f'].mean():.4f}"
    )

    print(
        f"  Clean INT8 mean          : "
        f"{df['comet_clean_i'].mean():.4f}"
    )

    print(
        f"  W1 ASR FP32 E2E mean     : "
        f"{df['comet_w1_asr_f_e2e'].mean():.4f}"
    )

    print(
        f"  W1 ASR INT8 E2E mean     : "
        f"{df['comet_w1_asr_i_e2e'].mean():.4f}"
    )

    print(
        f"  W1 COMET DiD             : "
        f"{df['comet_did_w1'].mean():+.4f}"
    )

    print(
        f"  W1 95% CI                : "
        f"[{comet_ci[0]:+.4f}, "
        f"{comet_ci[1]:+.4f}]"
    )

    print()
    print(
        "W1 ASR timing:"
    )

    print(
        f"  Mean inference           : "
        f"{df['w1_inference_time_ms'].mean():.2f} ms"
    )

    print(
        f"  Total inference         : "
        f"{df['w1_inference_time_ms'].sum() / 1000:.2f} s"
    )

    print()
    print(
        "ID integrity:"
    )

    print(
        f"  Unique sample IDs        : "
        f"{df['sample_id'].nunique()}"
    )

    print(
        "  Exact 100-ID cohort      : True"
    )

    # =====================================================================
    # SAVE
    # =====================================================================

    df.to_parquet(
        output_parquet,
        index=False,
    )

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Parquet                    : "
        f"{output_parquet}"
    )

    print(
        f"CSV                        : "
        f"{output_csv}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()