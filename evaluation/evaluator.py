"""
BridgeDEUX Core Framework
Robust Evaluation Engine
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from evaluation.exceptions import EvaluationDataError
from evaluation.result import ModelEvaluation


class EvaluationEngine:
    """
    Orchestrates benchmark discovery, schema validation,
    and downstream metric calculation.
    """

    def __init__(
        self,
        benchmark_dir: Path | None = None,
    ) -> None:

        if benchmark_dir is None:
            benchmark_dir = ProjectConfig.BENCHMARK_DIR

        self._benchmark_dir = benchmark_dir
        
        self._logger = BridgeLogger.get_logger(
            self.__class__.__name__
        )

    def discover(self) -> list[Path]:
        """
        Locates all valid evaluation target files.
        """

        targets = sorted(
            self._benchmark_dir.glob("*_results.parquet")
        )

        self._logger.info(
            "Discovered %d target dataset(s).",
            len(targets),
        )

        return targets

    def load_and_validate(
        self,
        target_file: Path,
    ) -> pd.DataFrame:
        """
        Loads dataset and enforces strict structural integrity.
        Raises EvaluationDataError on failure.
        """

        self._logger.info(
            "Ingesting dataset: %s",
            target_file.name,
        )

        try:

            df = pd.read_parquet(
                target_file
            )

        except Exception as e:

            raise EvaluationDataError(
                f"Failed to parse Parquet file '{target_file.name}': {e}"
            ) from e

        # ---------------------------------------------------------
        # Strict Content & Schema Enforcement
        # ---------------------------------------------------------

        if df.empty:
            
            raise EvaluationDataError(
                f"{target_file.name} contains no benchmark records."
            )

        required_schema = {
            "model_name",
            "model_version",
            "source_text",
            "translation",
            "reference_translation",
            "input_tokens",
            "output_tokens",
            "tokenization_time_ms",
            "generation_time_ms",
            "decoding_time_ms",
            "total_time_ms",
        }

        if not required_schema.issubset(df.columns):

            missing = required_schema - set(df.columns)

            raise EvaluationDataError(
                f"Schema violation in {target_file.name}. "
                f"Missing fields: {missing}"
            )

        if df["model_name"].nunique() != 1:
            
            raise EvaluationDataError(
                f"{target_file.name} contains multiple model names."
            )

        if df["model_version"].nunique() != 1:
            
            raise EvaluationDataError(
                f"{target_file.name} contains multiple model versions."
            )

        return df

    def evaluate_file(
        self,
        target_file: Path,
    ) -> ModelEvaluation:
        """
        Loads the data, validates it, and generates the
        ModelEvaluation DTO.
        """

        df = self.load_and_validate(
            target_file
        )

        total_samples = len(df)

        model_name = str(df["model_name"].iloc[0])
        model_version = str(df["model_version"].iloc[0])

        valid_df = df[
            df["translation"].notna()
            & df["reference_translation"].notna()
        ]

        failed_samples = total_samples - len(valid_df)

        mean_latency = df["total_time_ms"].dropna().mean()

        if pd.isna(mean_latency):
            raise EvaluationDataError(
                f"{target_file.name} contains no valid latency values."
            )

        # ---------------------------------------------------------
        # Metric strategies will populate this dictionary in 11.2
        # ---------------------------------------------------------

        return ModelEvaluation(
            model_name=model_name,
            model_version=model_version,
            total_samples=total_samples,
            failed_samples=failed_samples,
            mean_latency_ms=float(mean_latency),
        )

    def evaluate_all(self) -> list[ModelEvaluation]:
        """
        Discovers and evaluates all benchmarks in the directory.
        Safely catches and logs ingestion errors.
        """

        results = []
        targets = self.discover()

        for target in targets:

            try:

                evaluation = self.evaluate_file(
                    target
                )

                results.append(evaluation)

            except EvaluationDataError as e:

                self._logger.error(
                    "Skipping %s due to data error: %s",
                    target.name,
                    str(e),
                )

        return results