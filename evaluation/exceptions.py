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