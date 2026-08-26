from __future__ import annotations

import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from models.asr.base_asr import BaseASR
from models.asr.result import ASRResult


class ParakeetCppASR(BaseASR):

    def __init__(
        self,
        model_path: Path,
        executable_path: Path,
    ) -> None:
        self._model_path = Path(model_path)
        self._executable_path = Path(executable_path)
        self._is_loaded = False

        self.last_metrics: dict[str, float] = {
            "cli_execution_time_ms": 0.0,
        }

    @property
    def model_name(self) -> str:
        return "Parakeet.cpp (TDT 0.6B v3 F16)"

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Parakeet model not found: {self._model_path}"
            )

        if not self._executable_path.exists():
            raise FileNotFoundError(
                f"Parakeet executable not found: {self._executable_path}"
            )

        self._is_loaded = True

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        if not self._is_loaded:
            self.load()

        audio = np.asarray(audio, dtype=np.float32)

        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp:
            wav_path = Path(temp.name)

        try:
            pcm = np.clip(audio, -1.0, 1.0)
            pcm_int16 = (pcm * 32767).astype(np.int16)

            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm_int16.tobytes())

            command = [
                str(self._executable_path),
                "transcribe",
                "--model",
                str(self._model_path),
                "--input",
                str(wav_path),
                "--decoder",
                "tdt",
                "--lang",
                "de",
                "--json",
            ]

            start = time.perf_counter()

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            cli_execution_time_ms = (
                time.perf_counter() - start
            ) * 1000

            self.last_metrics = {
                "cli_execution_time_ms": cli_execution_time_ms,
            }

            if process.returncode != 0:
                raise RuntimeError(
                    f"Parakeet CLI failed "
                    f"(exit code {process.returncode}).\n"
                    f"stdout:\n{process.stdout}\n"
                    f"stderr:\n{process.stderr}"
                )

            output = json.loads(process.stdout)
            transcription = output["text"].strip()

            return ASRResult(
                transcription=transcription,
                generation_time_ms=cli_execution_time_ms,
            )

        finally:
            wav_path.unlink(missing_ok=True)

    def unload(self) -> None:
        self._is_loaded = False