# models/asr/whisper_cpp.py
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from pywhispercpp.model import Model

from models.asr.base_asr import BaseASR
from models.asr.result import ASRResult
from bridge.config import ProjectConfig

class WhisperCppASR(BaseASR):
    """
    Concrete CPU runtime wrapper for pywhispercpp.
    Exposes explicit resource initialization and inference telemetry metrics.
    """

    def __init__(self, model_size: str = "base", n_threads: int = 4) -> None:
        self._model_size = model_size
        self._n_threads = n_threads # Use threads for CPU parallelization, not audio splitting
        
        self._whisper_dir = ProjectConfig.MODEL_DIR / "whisper"
        self._engine: Model | None = None
        self._load_time_ms: float | None = None

    @property
    def model_name(self) -> str:
        return f"Whisper.cpp ({self._model_size})"

    @property
    def load_time_ms(self) -> float | None:
        return self._load_time_ms

    def load(self) -> None:
        if self._engine is not None:
            return

        if not self._whisper_dir.exists():
            raise FileNotFoundError(f"Whisper directory does not exist: {self._whisper_dir}")

        model_path = self._whisper_dir / f"ggml-{self._model_size}.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weight file not found: {model_path}")

        start_time = time.perf_counter()
        
        # Pass n_threads here for backend CPU parallelization
        self._engine = Model(
            model=self._model_size,
            models_dir=str(self._whisper_dir),
            n_threads=self._n_threads,
            redirect_whispercpp_logs_to=None
        )
        
        self._load_time_ms = (time.perf_counter() - start_time) * 1000

    def unload(self) -> None:
        self._engine = None
        self._load_time_ms = None

    def transcribe(self, audio: np.ndarray, language: str = "de") -> ASRResult:
        if self._engine is None:
            raise RuntimeError("ASR engine execution attempted before executing load().")

        start_time = time.perf_counter()

        # Remove n_processors to prevent destructive audio chunking
        # Force the engine to decode using the German language model
        segments = self._engine.transcribe(
            media=audio,
            language=language 
        )

        full_text = " ".join([seg.text for seg in segments]).strip()
        generation_time_ms = (time.perf_counter() - start_time) * 1000

        return ASRResult(
            transcription=full_text,
            generation_time_ms=generation_time_ms
        )