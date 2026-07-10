"""
BridgeDEUX

Downloader Verification Engine

Purpose
-------
Verify downloaded dataset shards before they are accepted into the
local BridgeDEUX data layer.

Responsibilities
----------------
- File existence
- File size validation
- SHA256 integrity hash
- Parquet readability
- Required schema validation
- Row counting

This module contains NO download logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from bridge.config import ProjectConfig
from datasets.downloaders.exceptions import VerificationError


# ============================================================
# Verification Result
# ============================================================

@dataclass(slots=True)
class VerificationResult:
    """
    Result returned after verifying a downloaded file.
    """

    success: bool

    message: str

    file_path: Path

    file_size_bytes: int

    file_size_mb: float

    sha256: str

    row_count: int

    columns: list[str]


# ============================================================
# Verifier
# ============================================================

class FileVerifier:

    """
    Performs integrity verification for downloaded dataset files.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """
        Compute SHA256 hash using chunked reading.
        """

        sha = hashlib.sha256()

        with open(file_path, "rb") as file:

            while True:

                block = file.read(ProjectConfig.HASH_BLOCK_SIZE)

                if not block:
                    break

                sha.update(block)

        return sha.hexdigest()

    # --------------------------------------------------------

    @staticmethod
    def verify(
        file_path: Path,
        required_columns: Iterable[str],
    ) -> VerificationResult:

        file_path = Path(file_path)

        # ----------------------------------------------------
        # Exists
        # ----------------------------------------------------

        if not file_path.exists():

            raise VerificationError(
                f"File does not exist:\n{file_path}"
            )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        file_size = file_path.stat().st_size

        if file_size <= 0:

            raise VerificationError(
                f"File is empty:\n{file_path}"
            )

        # ----------------------------------------------------
        # SHA256
        # ----------------------------------------------------

        sha256 = FileVerifier.compute_sha256(file_path)

        # ----------------------------------------------------
        # Read parquet
        # ----------------------------------------------------

        try:

            dataframe = pd.read_parquet(file_path)

        except Exception as error:

            raise VerificationError(
                f"Unable to read parquet file:\n{file_path}\n\n{error}"
            )

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        missing = [

            column

            for column in required_columns

            if column not in dataframe.columns

        ]

        if missing:

            raise VerificationError(
                "Missing required columns:\n"
                + ", ".join(missing)
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return VerificationResult(

            success=True,

            message="Verification successful.",

            file_path=file_path,

            file_size_bytes=file_size,

            file_size_mb=round(file_size / (1024 * 1024), 2),

            sha256=sha256,

            row_count=len(dataframe),

            columns=list(dataframe.columns),

        )