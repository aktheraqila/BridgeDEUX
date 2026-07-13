"""
BridgeDEUX Core Framework
Evaluation Exceptions
"""

from __future__ import annotations


class EvaluationError(Exception):
    """
    Base exception for all evaluation-related errors 
    within the framework.
    """
    pass


class EvaluationDataError(EvaluationError):
    """
    Raised when benchmark Parquet data is malformed, 
    corrupted, or fails structural schema validation.
    """
    pass

class BenchmarkError(Exception):
    """Base exception for all benchmark-related failures."""
    pass

class CheckpointError(BenchmarkError):
    """Raised when filesystem/WAL operations fail (e.g., Disk Full)."""
    pass


class TranslationError(Exception):
    """Raised when the model fails to translate a specific sample."""
    pass

class CircuitBreakerError(BenchmarkError):
    """Circuit breaker triggered due to excessive consecutive failures."""
    pass