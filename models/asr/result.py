# models/asr/result.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ASRResult:
    transcription: str
    generation_time_ms: float
    # We can add audio_duration_ms here later if the model calculates it