"""
BridgeDEUX Core Framework
Abstract Base Translator Interface (Frozen v1.0)
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from models.translators.result import TranslationResult


class BaseTranslator(ABC):
    """
    Abstract contract for all Machine Translation models.
    Defines the public API for the benchmark runner without enforcing
    any model-specific Hugging Face logic.
    """

    # ---------------------------------------------------------
    # Identity & Metadata Primitives
    # ---------------------------------------------------------
    @abstractmethod
    def model_name(self) -> str:
        """Returns the human-readable identifier of the model."""
        pass
        
    @abstractmethod
    def model_version(self) -> str:
        """Returns the exact model checkpoint or version identifier."""
        pass

    @abstractmethod
    def device(self) -> str:
        """Returns the hardware device the model is loaded on (e.g., 'cuda', 'cpu')."""
        pass

    # ---------------------------------------------------------
    # Lifecycle Management Primitives
    # ---------------------------------------------------------
    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the model engine is currently initialized in memory."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Loads the model weights and tokenizer configurations into hardware memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Clears the model and tokenizer from memory to prevent Out-Of-Memory (OOM) 
        errors when switching models during sequential benchmarking operations.
        """
        pass

    # ---------------------------------------------------------
    # Execution & Inference Primitives
    # ---------------------------------------------------------
    @abstractmethod
    def translate(self, text: str) -> TranslationResult:
        """
        Translates a single string.
        Must return a TranslationResult dataclass containing the prediction
        and granular execution metrics.
        """
        pass

    @abstractmethod
    def translate_batch(self, texts: list[str]) -> list[TranslationResult]:
        """
        Translates a batch of strings.
        Must return a list of TranslationResult dataclasses.
        """
        pass