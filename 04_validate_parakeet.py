"""
Module 4: Parakeet.cpp ASR Validation

Purpose:
--------
Validate that the Parakeet.cpp backend:

1. Finds the expected executable and GGUF model.
2. Accepts a known German speech sample.
3. Produces a German transcription.
4. Measures end-to-end cold-start transcription time.

This module DOES NOT:
- Benchmark MSLT.
- Train or fine-tune Parakeet.
- Measure aggregate ASR performance.
- Measure RAM/CPU.
- Perform translation.
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf
from jiwer import wer

from models.asr.parakeet_cpp import ParakeetCppASR


# --------------------------------------------------
# Fixed validation artifact
# --------------------------------------------------

AUDIO_PATH = (
    Path("analysis")
    / "mobile_benchmark_audio_100"
    / "common_voice_de_17325303.wav"
)

REFERENCE = "Insbesondere keine Topflappen."


# --------------------------------------------------
# Parakeet.cpp configuration
# --------------------------------------------------

MODEL_PATH = (
    Path("models")
    / "parakeet"
    / "tdt-0.6b-v3-f16.gguf"
)

EXECUTABLE_PATH = (
    Path("models")
    / "parakeet"
    / "parakeet-cli.exe"
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_parakeet() -> None:

    model = ParakeetCppASR(
        model_path=MODEL_PATH,
        executable_path=EXECUTABLE_PATH,
    )

    print("=" * 60)
    print(f"Validating Model: {model.model_name}")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Verify backend paths
    # --------------------------------------------------

    print("\n[1/3] Verifying backend paths...")

    if not EXECUTABLE_PATH.exists():
        raise FileNotFoundError(
            f"Parakeet executable not found: {EXECUTABLE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Parakeet GGUF model not found: {MODEL_PATH}"
        )

    print("Done. Executable and GGUF artifact located.")

    # --------------------------------------------------
    # 2. Load deterministic validation audio
    # --------------------------------------------------

    print("\n[2/3] Loading deterministic validation audio...")

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Validation audio not found: {AUDIO_PATH}"
        )

    audio, sample_rate = sf.read(
        str(AUDIO_PATH),
        dtype="float32",
    )

    # Convert stereo → mono if necessary.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # This validator expects the Parakeet wrapper's
    # 16-kHz input contract.
    if sample_rate != 16000:
        raise ValueError(
            f"Validation audio must be 16 kHz, "
            f"but received {sample_rate} Hz."
        )

    print(
        f"Done. Prepared {len(audio)} samples "
        f"at {sample_rate} Hz."
    )

    # --------------------------------------------------
    # 3. Execute transcription
    # --------------------------------------------------

    print("\n[3/3] Executing CLI subprocess transcription...")
    print(
        "Note: Latency includes subprocess startup, "
        "model initialization, disk I/O, and decoding."
    )

    model.load()

    start = time.perf_counter()

    result = model.transcribe(audio)

    validation_time_ms = (
        time.perf_counter() - start
    ) * 1000

    hypothesis = result.transcription.strip()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    sample_wer = wer(
        REFERENCE.lower(),
        hypothesis.lower(),
    )

    print("\n--- Results ---")
    print(f"Audio:       {AUDIO_PATH}")
    print(f"Reference:   {REFERENCE}")
    print(f"Hypothesis:  {hypothesis}")
    print(f"WER:         {sample_wer:.4f}")
    print(
        f"CLI latency: {result.generation_time_ms:.2f} ms"
    )
    print(
        f"Validation:  {validation_time_ms:.2f} ms"
    )
    print("=" * 60)

    model.unload()


if __name__ == "__main__":
    validate_parakeet()