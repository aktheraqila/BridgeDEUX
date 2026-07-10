"""
BridgeDEUX Core Framework
Translation Metric Strategies
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import sacrebleu

from evaluation.exceptions import EvaluationError


# ---------------------------------------------------------
# Core Abstraction
# ---------------------------------------------------------

class EvaluationMetric(ABC):
    """
    Abstract base interface for all translation and 
    speech evaluation metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the human-readable string identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:
        """
        Executes the specific metric scoring algorithm.
        """
        raise NotImplementedError


# ---------------------------------------------------------
# Intermediate Library Abstraction
# ---------------------------------------------------------

class SacreBleuMetric(EvaluationMetric, ABC):
    """
    Intermediate abstract tier encapsulating shared validation,
    exception boundaries, and type casting for SacreBLEU metrics.
    """

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:

        if not predictions:
            return 0.0

        if len(predictions) != len(references):
            raise EvaluationError(
                f"Data length mismatch: {len(predictions)} predictions "
                f"vs {len(references)} references."
            )

        try:

            score = self._execute_calculation(
                predictions=predictions,
                references=references,
            )

        # SacreBLEU does not expose a unified exception hierarchy.
        # We catch a broad Exception here to capture internal library 
        # crashes, but immediately wrap it in our domain-specific 
        # EvaluationError to maintain strict engine boundaries.
        except Exception as e:

            raise EvaluationError(
                f"SacreBLEU execution failed for {self.name}: {e}"
            ) from e

        return float(score)

    @abstractmethod
    def _execute_calculation(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:
        """
        Internal hook for concrete library execution.
        """
        raise NotImplementedError


# ---------------------------------------------------------
# Concrete SacreBLEU Implementations
# ---------------------------------------------------------

class BleuMetric(SacreBleuMetric):
    """
    Concrete implementation of the SacreBLEU corpus BLEU calculation.
    """

    @property
    def name(self) -> str:
        return "BLEU"

    def _execute_calculation(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:

        result = sacrebleu.corpus_bleu(
            predictions,
            references,
        )

        return result.score


class ChrfMetric(SacreBleuMetric):
    """
    Concrete implementation of the SacreBLEU chrF++ calculation.
    """

    @property
    def name(self) -> str:
        return "chrF++"

    def _execute_calculation(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:

        # word_order=2 upgrades standard chrF to chrF++
        result = sacrebleu.corpus_chrf(
            predictions,
            references,
            word_order=2,
        )

        return result.score