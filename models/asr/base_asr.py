# models/asr/base_asr.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from models.asr.result import ASRResult

class BaseASR(ABC):
    """
    Abstract contract for all Automatic Speech Recognition models.
    Defines the public API for the benchmark runner without enforcing
    any model-specific logic.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> ASRResult:
        """
        Transcribe a 16kHz float32 audio array into text.
        Must return an ASRResult containing the text and execution time.
        """
        pass