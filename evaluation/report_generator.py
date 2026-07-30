"""
BridgeDEUX Core Framework
Evaluation Report Generator
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from evaluation.exceptions import EvaluationError
from evaluation.result import ModelEvaluation


class ReportGenerator:
    """
    Generates reproducible evaluation reports from
    ModelEvaluation objects.
    """

    def __init__(
        self,
        output_directory: Path | None = None,
    ) -> None:

        if output_directory is None:
            output_directory = (
                ProjectConfig.EVALUATION_DIR
            )

        self._output_directory = output_directory

        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._logger = BridgeLogger.get_logger(
            self.__class__.__name__
        )

    def generate(
        self,
        evaluations: list[ModelEvaluation],
    ) -> None:
        """
        Generates CSV, Parquet, and Markdown reports.
        """
        
        if not evaluations:
            raise EvaluationError(
                "No evaluation results supplied for report generation."
            )

        dataframe = self._to_dataframe(
            evaluations
        )

        self._write_csv(dataframe)

        self._write_parquet(dataframe)

        self._write_markdown(dataframe)

        self._logger.info(
            "Evaluation reports generated successfully."
        )

    def _to_dataframe(
        self,
        evaluations: list[ModelEvaluation],
    ) -> pd.DataFrame:
        """
        Converts evaluation objects into a tabular format.
        """

        rows = []

        for evaluation in evaluations:

            row = {
                "experiment_name": evaluation.experiment_name,

                "model_name":
                    evaluation.model_name,

                "model_version":
                    evaluation.model_version,

                "total_samples":
                    evaluation.total_samples,

                "failed_samples":
                    evaluation.failed_samples,

                "mean_latency_ms":
                    evaluation.mean_latency_ms,
            }

            row.update(
                evaluation.metrics
            )

            rows.append(row)

        return pd.DataFrame(rows)

    def _write_csv(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        output_file = (
            self._output_directory /
            "evaluation_summary.csv"
        )

        dataframe.to_csv(
            output_file,
            index=False,
        )

        self._logger.info(
            "CSV report written to %s",
            output_file,
        )

    def _write_parquet(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        output_file = (
            self._output_directory /
            "evaluation_summary.parquet"
        )

        dataframe.to_parquet(
            output_file,
            index=False,
        )

        self._logger.info(
            "Parquet report written to %s",
            output_file,
        )

    def _write_markdown(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        output_file = (
            self._output_directory /
            "evaluation_summary.md"
        )

        markdown = dataframe.to_markdown(
            index=False
        )

        output_file.write_text(
            markdown,
            encoding="utf-8",
        )

        self._logger.info(
            "Markdown report written to %s",
            output_file,
        )