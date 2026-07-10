"""
BridgeDEUX
Core Framework

Module
------
bridge.metadata

Purpose
-------
Collect reproducible execution metadata for every production
operation.

This module extends the basic environment information provided
by ProjectConfig with additional package versions, timestamps,
and execution identifiers required for reproducible research.
"""

from __future__ import annotations

import importlib.metadata
import uuid

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from bridge.config import ProjectConfig


# ============================================================
# Data Model
# ============================================================

@dataclass(slots=True, frozen=True)
class ExecutionMetadata:
    """
    Immutable execution metadata.

    Every downloader, benchmark, exporter and evaluation
    should generate one instance.
    """

    execution_id: str

    timestamp_utc: str

    project_name: str

    dataset_repository: str

    dataset_configuration: str

    environment: dict[str, Any]

    packages: dict[str, str]


# ============================================================
# Collector
# ============================================================

class MetadataCollector:
    """
    Centralized metadata collector.

    Future versions can extend this class with

    • Git commit
    • Android device info
    • CPU governor
    • Battery level
    • Thermal state
    """

    EXTRA_PACKAGES = (
        "datasets",
        "pyarrow",
        "torch",
        "transformers",
        "optimum",
        "onnxruntime",
    )

    # --------------------------------------------------------

    @staticmethod
    def _version(package: str) -> str:
        """
        Return installed package version.
        """

        try:
            return importlib.metadata.version(package)

        except importlib.metadata.PackageNotFoundError:
            return "Not Installed"

        except Exception:
            return "Unknown"

    # --------------------------------------------------------

    @classmethod
    def collect(cls) -> ExecutionMetadata:
        """
        Collect execution metadata.
        """

        environment = dict(ProjectConfig.environment())

        packages = {
            package: cls._version(package)
            for package in cls.EXTRA_PACKAGES
        }

        return ExecutionMetadata(

            execution_id=str(uuid.uuid4()),

            timestamp_utc=datetime.now(
                timezone.utc
            ).isoformat(),

            project_name=ProjectConfig.PROJECT_NAME,

            dataset_repository=ProjectConfig.DATASET_REPOSITORY,

            dataset_configuration=ProjectConfig.DATASET_CONFIGURATION,

            environment=environment,

            packages=packages,
        )

    # --------------------------------------------------------

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        """
        Return metadata as a JSON-serializable dictionary.
        """

        return asdict(cls.collect())