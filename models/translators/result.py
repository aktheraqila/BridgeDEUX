"""
BridgeDEUX Core Framework
Translator Inference Result Definition
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """
    Immutable structured data container for a single translation event.
    Self-describing to support reliable downstream dataset integration and 
    statistical profiling for thesis metrics.
    """
    model_name: str
    model_version: str
    source_text: str
    translation: str
    input_tokens: int
    output_tokens: int
    tokenization_time_ms: float
    generation_time_ms: float
    decoding_time_ms: float
    total_time_ms: float