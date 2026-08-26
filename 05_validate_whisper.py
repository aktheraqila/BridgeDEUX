"""
Module 5: Whisper.cpp ASR Validation

Purpose:
--------
Validate that the Whisper.cpp backend:
1. Finds the expected model via ProjectConfig.
2. Loads correctly into memory.
3. Accepts a known German speech sample via AudioProcessor.
4. Produces German transcription.
5. Measures in-process inference time.

This module DOES NOT:
- Benchmark MSLT.
- Fine-tune Whisper.
- Measure aggregate ASR performance.
- Perform translation.
"""

from __future__ import annotations

import time
from pathlib import Path

from jiwer import wer

from bridge.config import ProjectConfig
from bridge.audio import AudioProcessor
from models.asr.whisper_cpp import WhisperCppASR


def validate_whisper() -> None:
    # --------------------------------------------------
    # Fixed validation artifact
    # --------------------------------------------------
    audio_path = Path("analysis/mobile_benchmark_audio_100/common_voice_de_17325303.wav")
    reference = "Insbesondere keine Topflappen."
    model_size = "base"
    n_threads = 4

    # --------------------------------------------------
    # 1. Verify model path via ProjectConfig
    # --------------------------------------------------
    print("\n[1/3] Verifying backend paths...")
    
    expected_path = ProjectConfig.MODEL_DIR / "whisper" / f"ggml-{model_size}.bin"
    if not expected_path.exists():
        raise FileNotFoundError(f"Whisper model not found: {expected_path}")

    print("Done. Whisper model artifact located.")

    # Instantiate wrapper only after config and paths are confirmed valid
    model = WhisperCppASR(model_size=model_size, n_threads=n_threads)

    print("=" * 60)
    print(f"Validating Model: {model.model_name}")
    print("=" * 60)

    # --------------------------------------------------
    # 2. Load deterministic validation audio via AudioProcessor
    # --------------------------------------------------
    print("\n[2/3] Extracting isolated validation audio...")

    if not audio_path.exists():
        raise FileNotFoundError(f"Validation audio not found: {audio_path}")

    raw_array, sample_rate = AudioProcessor.decode_to_pcm(audio_path.read_bytes())
    pcm_16k = AudioProcessor.resample_to_16k(raw_array, sample_rate)
    
    print(f"Done. Prepared {len(pcm_16k)} samples at 16kHz.")

    # --------------------------------------------------
    # 3. Load and transcribe
    # --------------------------------------------------
    print("\n[3/3] Loading Whisper.cpp and transcribing...")
    print("Note: Latency represents isolated in-process memory loading and execution.")

    model.load()

    result = model.transcribe(pcm_16k, language="de")
    hypothesis = result.transcription.strip()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    sample_wer = wer(reference.lower(), hypothesis.lower())

    print("\n--- Results ---")
    print(f"Audio:        {audio_path}")
    print(f"Reference:    {reference}")
    print(f"Hypothesis:   {hypothesis}")
    print(f"WER:          {sample_wer:.4f}")
    print(f"Model load:   {model.load_time_ms:.2f} ms")
    print(f"Inference:    {result.generation_time_ms:.2f} ms")
    print("=" * 60)

    model.unload()

if __name__ == "__main__":
    ProjectConfig.initialize()
    validate_whisper()