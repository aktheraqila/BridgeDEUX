"""
BridgeDEUX Core Framework
Translation Metric Strategies
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import sacrebleu

from bridge.logger import BridgeLogger
from evaluation.exceptions import EvaluationError
from comet import download_model, load_from_checkpoint

# ---------------------------------------------------------
# Core Abstraction
# ---------------------------------------------------------

class EvaluationMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        sources: list[str] | None = None
    ) -> float:
        raise NotImplementedError

# ---------------------------------------------------------
# Intermediate Library Abstraction
# ---------------------------------------------------------

class SacreBleuMetric(EvaluationMetric, ABC):
    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        sources: list[str] | None = None
    ) -> float:
        if not predictions:
            return 0.0

        if len(predictions) != len(references):
            raise EvaluationError("Prediction/reference length mismatch.")

        try:
            score = self._execute_calculation(predictions, references)
        except Exception as e:
            raise EvaluationError(f"SacreBLEU execution failed for {self.name}: {e}") from e

        return float(score)

    @abstractmethod
    def _execute_calculation(
        self, predictions: list[str], references: list[list[str]]
    ) -> float:
        raise NotImplementedError

# ---------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------

class BleuMetric(SacreBleuMetric):
    @property
    def name(self) -> str:
        return "BLEU"

    def _execute_calculation(self, predictions, references):
        return sacrebleu.corpus_bleu(predictions, references).score

class ChrfMetric(SacreBleuMetric):
    @property
    def name(self) -> str:
        return "chrF++"

    def _execute_calculation(self, predictions, references):
        return sacrebleu.corpus_chrf(predictions, references, word_order=2).score

class CometMetric(EvaluationMetric):
    DEFAULT_BATCH_SIZE = 8

    def __init__(
        self, 
        model_name: str = "Unbabel/wmt20-comet-da",  # <-- CHANGE THIS LINE
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._logger = BridgeLogger.get_logger(self.__class__.__name__)
        
        self._logger.info("Loading COMET model: %s (Batch Size: %d)", self._model_name, self._batch_size)

        try:
            model_path = download_model(model_name)
            self._model = load_from_checkpoint(model_path)
            self._logger.info("COMET model loaded successfully.")
        except Exception as e:
            raise EvaluationError(f"Failed to initialize COMET model '{model_name}': {e}") from e

    @property
    def name(self) -> str:
        return "COMET"
    
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def compute(
        self, 
        predictions: list[str], 
        references: list[list[str]], 
        sources: list[str] | None = None
    ) -> float:
        
        if not predictions:
            return 0.0
            
        if sources is None:
            raise EvaluationError("COMET metric requires 'sources' to be provided.")

        if len(predictions) != len(references):
            raise EvaluationError("Prediction/reference length mismatch.")
            
        if len(predictions) != len(sources):
            raise EvaluationError("Prediction/source length mismatch.")

        # Validate reference structure
        for refs in references:
            if len(refs) != 1:
                raise EvaluationError("COMET implementation currently supports exactly one reference per sample.")

        # Construct data batch
        data = [
            {"src": src, "mt": mt, "ref": refs[0]}
            for src, mt, refs in zip(sources, predictions, references)
        ]

        try:
            # CPU inference
            model_output = self._model.predict(
                data, 
                batch_size=self.batch_size, 
                gpus=0
            )
            return float(model_output.system_score)
        except Exception as e:
            raise EvaluationError(f"COMET prediction failed: {e}") from e