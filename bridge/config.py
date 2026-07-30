"""
BridgeDEUX

Global Project Configuration

This module centralizes every directory and project-wide constant.
No other module should hardcode filesystem paths.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import pandas


@dataclass(frozen=True)
class ProjectConfig:
    """
    Global configuration for the BridgeDEUX project.
    """

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    PROJECT_NAME: str = "BridgeDEUX"

    DATASET_REPOSITORY: str = "fixie-ai/covost2"

    DATASET_CONFIGURATION: str = "de_en"

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # ------------------------------------------------------------------
    # Dataset directories
    # ------------------------------------------------------------------

    EVALUATION_DIR = PROJECT_ROOT / "evaluation_reports"

    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"

    RAW_DATA_DIR: Path = DATASETS_DIR / "raw"

    CACHE_DIR: Path = DATASETS_DIR / "cache"

    REPORT_DIR: Path = DATASETS_DIR / "reports"

    LOG_DIR: Path = DATASETS_DIR / "logs"

    PROVIDER_DIR: Path = DATASETS_DIR / "providers"

    BUILDER_DIR: Path = DATASETS_DIR / "builders"

    VALIDATOR_DIR: Path = DATASETS_DIR / "validators"


    # ------------------------------------------------------------------
    # Experiment directories
    # ------------------------------------------------------------------

    BENCHMARK_DIR: Path = CACHE_DIR

    MODEL_DIR: Path = PROJECT_ROOT / "models"

    ANDROID_DIR: Path = PROJECT_ROOT / "android"

    EXPERIMENT_DIR: Path = PROJECT_ROOT / "experiments"

    RESULTS_DIR: Path = PROJECT_ROOT / "results"


    # ------------------------------------------------------------------
    # Downloader
    # ------------------------------------------------------------------

    DOWNLOAD_TIMEOUT: int = 300

    HASH_BLOCK_SIZE: int = 65536

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    LOG_FILE_NAME: str = "bridgedeux.log"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def environment() -> dict:
        """
        Returns execution environment information.
        """

        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas_version": pandas.__version__,
            "huggingface_hub_version": huggingface_hub.__version__,
        }

    # ------------------------------------------------------------------
    # Directory initialization
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls) -> None:
        """
        Create every required project directory.
        """

        directories = [

            cls.DATASETS_DIR,

            cls.RAW_DATA_DIR,

            cls.CACHE_DIR,

            cls.REPORT_DIR,

            cls.LOG_DIR,

            cls.PROVIDER_DIR,

            cls.BUILDER_DIR,

            cls.VALIDATOR_DIR,

            cls.BENCHMARK_DIR,

            cls.MODEL_DIR,

            cls.ANDROID_DIR,

            cls.EXPERIMENT_DIR,

            cls.RESULTS_DIR,

        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)