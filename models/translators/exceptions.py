"""
BridgeDEUX Core Framework
Custom Exception Hierarchy for Translator Engines
"""

from __future__ import annotations


class TranslatorError(Exception):
    """
    Base exception for the translator subsystem.
    Allows the benchmark runner to catch all translator-related errors uniformly.
    """
    pass


class ModelLoadError(TranslatorError):
    """
    Raised when the neural network tensor weights or structure fail to allocate
    in memory or transfer to the designated hardware target (CPU/CUDA).
    """
    pass


class TokenizerLoadError(TranslatorError):
    """
    Raised when the subword tokenizer resources, vocabulary files, or config
    fail to initialize from the local path or remote repository hub.
    """
    pass


class TranslationError(TranslatorError):
    """
    Raised when an operational exception occurs during active inference execution,
    such as empty inputs, internal model matrix crashes, or string decoding failures.
    """
    pass


class ModelNotLoadedError(TranslatorError):
    """
    Defensive programming guard exception. Raised if a translation request is 
    dispatched before the engine has successfully executed its `load()` lifecycle method.
    """
    pass