# models/asr/vosk.py
from __future__ import annotations

import time
import json
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
from pathlib import Path

from bridge.logger import BridgeLogger
from models.asr.base_asr import BaseASR
from models.asr.result import ASRResult

# Suppress verbose Vosk C++ backend logs to keep the console clean
SetLogLevel(-1)

class VoskASR(BaseASR):
    """
    Concrete ASR engine implementation for Vosk.
    """
    def __init__(self, model_path: Path | str):
        self._model_path = str(model_path)
        self._display_name = "Vosk"
        self._logger = BridgeLogger.get_logger(self.__class__.__name__)
        
        self._model: Model | None = None
        self._sample_rate = 16000.0 

    @property
    def model_name(self) -> str:
        return self._display_name

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        self._logger.info("Loading %s model from %s...", self._display_name, self._model_path)
        try:
            self._model = Model(self._model_path)
            self._logger.info("%s initialized successfully.", self._display_name)
        except Exception as e:
            self._logger.error("Failed to load Vosk engine: %s", e)
            raise

    def unload(self) -> None:
        if not self.is_loaded():
            return
            
        self._logger.info("Unloading %s from memory...", self._display_name)
        self._model = None

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        """
        Transcribe a 16kHz float32 audio array into text.
        Converts the array to 16-bit PCM bytes and streams it to the engine.
        """
        if not self.is_loaded():
            raise RuntimeError("Inference requested before engine initialization.")
            
        start_time = time.perf_counter()
        
        try:
            # Convert float32 numpy array to 16-bit PCM bytes
            pcm_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            rec = KaldiRecognizer(self._model, self._sample_rate)
            
            # Stream in chunks (4000 bytes) for C++ buffer stability
            chunk_size = 4000
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                rec.AcceptWaveform(chunk)
            
            result_json = json.loads(rec.FinalResult())
            transcription = result_json.get("text", "")
            
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            
            return ASRResult(
                transcription=transcription,
                generation_time_ms=generation_time_ms
            )

        except Exception as e:
            self._logger.error("Vosk transcription pipeline failed: %s", e)
            raise