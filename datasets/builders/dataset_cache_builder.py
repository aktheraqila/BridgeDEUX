#!/usr/bin/env python3
"""
BridgeDEUX
MSLT German Test Cache Builder

Builds a verified local cache from:
    C:\\Users\\user\\Downloads\\MSLT_Corpus.zip

Only the German Test split is extracted.

MSLT structure:
    T0.de.wav  -> German audio
    T1.de.snt  -> German verbatim transcript
    T2.de.snt  -> German cleaned transcript
    T3.en.snt  -> English translation
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from datasets.downloaders.manifest import ManifestManager, ManifestRecord
from datasets.downloaders.verifier import FileVerifier


logger = BridgeLogger.get_logger("MSLTCacheBuilder")


DATASET_NAME = "mslt"
LANGUAGE_PAIR = "de_en"
SPLIT = "test"

ZIP_ROOT = "MSLT_Corpus/Data/MSLT_Test_DE_20160516"

AUDIO_PATTERN = re.compile(
    r"MSLT_Test_DE_(\d+)\.T0\.de\.wav$",
    re.IGNORECASE,
)


def backup_if_exists(path: Path) -> None:
    """Create a .bak copy before replacing an existing output."""
    if path.exists():
        backup_path = Path(str(path) + ".bak")
        shutil.copy2(path, backup_path)
        logger.info("Backup created: %s", backup_path)


def decode_mslt_text(data: bytes) -> str:
    """
    Decode MSLT transcript files.

    The inspected MSLT files begin with FF FE, confirming UTF-16LE.
    """
    if data.startswith(b"\xff\xfe"):
        text = data.decode("utf-16-le")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-16-le")

    return text.replace("\ufeff", "").strip()


def read_zip_text(
    archive: zipfile.ZipFile,
    filename: str,
) -> str:
    """Read and decode one transcript file from the archive."""
    return decode_mslt_text(archive.read(filename))


def build_mslt_cache(zip_path: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"MSLT ZIP file does not exist:\n{zip_path}"
        )

    raw_dir = (
        ProjectConfig.RAW_DATA_DIR
        / DATASET_NAME
        / LANGUAGE_PAIR
        / SPLIT
    )

    cache_dir = (
        ProjectConfig.CACHE_DIR
        / DATASET_NAME
        / LANGUAGE_PAIR
        / SPLIT
    )

    report_dir = ProjectConfig.REPORT_DIR

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cache_dir / "mslt_de_en_test.parquet"
    csv_path = cache_dir / "mslt_de_en_test.csv"
    metadata_path = cache_dir / "mslt_de_en_test.json"

    logger.info("=" * 70)
    logger.info("MSLT GERMAN TEST CACHE BUILD")
    logger.info("=" * 70)
    logger.info("Source ZIP: %s", zip_path)
    logger.info("Raw output: %s", raw_dir)
    logger.info("Cache output: %s", cache_dir)

    records: list[dict] = []

    missing_t0: list[str] = []
    missing_t1: list[str] = []
    missing_t2: list[str] = []
    missing_t3: list[str] = []

    duplicate_ids: list[str] = []
    empty_t1: list[str] = []
    empty_t2: list[str] = []
    empty_t3: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as archive:

        all_names = set(archive.namelist())

        # --------------------------------------------------
        # Discover German Test T0 audio files.
        # --------------------------------------------------

        audio_files = sorted(
            name
            for name in all_names
            if name.startswith(ZIP_ROOT + "/")
            and AUDIO_PATTERN.search(Path(name).name)
        )

        logger.info(
            "German Test T0 audio files discovered: %d",
            len(audio_files),
        )

        seen_ids: set[str] = set()

        for audio_member in audio_files:

            filename = Path(audio_member).name

            match = AUDIO_PATTERN.search(filename)

            if not match:
                continue

            utterance_id = match.group(1)

            if utterance_id in seen_ids:
                duplicate_ids.append(utterance_id)
                continue

            seen_ids.add(utterance_id)

            t0_member = audio_member
            base = audio_member[:-len(".T0.de.wav")]

            t1_member = base + ".T1.de.snt"
            t2_member = base + ".T2.de.snt"
            t3_member = base + ".T3.en.snt"

            # ----------------------------------------------
            # Check alignment before extracting anything.
            # ----------------------------------------------

            has_t0 = t0_member in all_names
            has_t1 = t1_member in all_names
            has_t2 = t2_member in all_names
            has_t3 = t3_member in all_names

            if not has_t0:
                missing_t0.append(utterance_id)

            if not has_t1:
                missing_t1.append(utterance_id)

            if not has_t2:
                missing_t2.append(utterance_id)

            if not has_t3:
                missing_t3.append(utterance_id)

            if not (has_t0 and has_t2 and has_t3):
                continue

            # ----------------------------------------------
            # Read transcripts.
            # ----------------------------------------------

            t1 = read_zip_text(archive, t1_member) if has_t1 else ""
            t2 = read_zip_text(archive, t2_member)
            t3 = read_zip_text(archive, t3_member)

            if not t1:
                empty_t1.append(utterance_id)

            if not t2:
                empty_t2.append(utterance_id)

            if not t3:
                empty_t3.append(utterance_id)

            if not t2 or not t3:
                continue

            # ----------------------------------------------
            # Extract audio to local raw dataset.
            # ----------------------------------------------

            audio_output = raw_dir / filename

            if not audio_output.exists():
                with archive.open(t0_member) as source:
                    with open(audio_output, "wb") as target:
                        shutil.copyfileobj(source, target)

            relative_audio_path = audio_output.relative_to(
                ProjectConfig.PROJECT_ROOT
            )

            records.append(
                {
                    "id": utterance_id,
                    "audio_path": str(relative_audio_path).replace("\\", "/"),
                    "source_text": t2,
                    "target_text": t3,
                    "raw_transcript": t1,
                    "dataset": DATASET_NAME,
                    "split": SPLIT,
                    "language_pair": LANGUAGE_PAIR,
                }
            )

    # ------------------------------------------------------
    # Alignment validation.
    # ------------------------------------------------------

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate MSLT utterance IDs detected: "
            + ", ".join(duplicate_ids[:20])
        )

    if missing_t0 or missing_t2 or missing_t3:
        raise RuntimeError(
            "MSLT alignment validation failed.\n"
            f"Missing T0: {len(missing_t0)}\n"
            f"Missing T2: {len(missing_t2)}\n"
            f"Missing T3: {len(missing_t3)}"
        )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "No valid MSLT records were created."
        )

    if dataframe["id"].duplicated().any():
        raise RuntimeError(
            "Duplicate IDs remain after cache construction."
        )

    if dataframe[
        ["audio_path", "source_text", "target_text"]
    ].isnull().any().any():
        raise RuntimeError(
            "Null values detected in required evaluation fields."
        )

    # ------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------

    backup_if_exists(parquet_path)
    backup_if_exists(csv_path)
    backup_if_exists(metadata_path)

    dataframe.to_parquet(
        parquet_path,
        index=False,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "dataset": DATASET_NAME,
        "language_pair": LANGUAGE_PAIR,
        "split": SPLIT,
        "source_zip": str(zip_path),
        "zip_root": ZIP_ROOT,
        "row_count": len(dataframe),
        "missing_t0": missing_t0,
        "missing_t1": missing_t1,
        "missing_t2": missing_t2,
        "missing_t3": missing_t3,
        "duplicate_ids": duplicate_ids,
        "empty_t1": empty_t1,
        "empty_t2": empty_t2,
        "empty_t3": empty_t3,
        "columns": list(dataframe.columns),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ------------------------------------------------------
    # Verify Parquet with existing BridgeDEUX verifier.
    # ------------------------------------------------------

    verification = FileVerifier.verify(
        file_path=parquet_path,
        required_columns=[
            "id",
            "audio_path",
            "source_text",
            "target_text",
            "raw_transcript",
            "dataset",
            "split",
            "language_pair",
        ],
    )

    if not verification.success:
        raise RuntimeError(
            f"MSLT Parquet verification failed: "
            f"{verification.message}"
        )

    # ------------------------------------------------------
    # Register verified cache in manifest.
    # ------------------------------------------------------

    manifest_path = (
        report_dir / "mslt_test_manifest.json"
    )

    manifest = ManifestManager(manifest_path)

    manifest.add(
        ManifestRecord(
            filename=parquet_path.name,
            relative_path=str(
                parquet_path.relative_to(
                    ProjectConfig.PROJECT_ROOT
                )
            ),
            file_size_bytes=verification.file_size_bytes,
            file_size_mb=verification.file_size_mb,
            sha256=verification.sha256,
            row_count=verification.row_count,
            status="SUCCESS",
        )
    )

    # ------------------------------------------------------
    # Final report.
    # ------------------------------------------------------

    logger.info("=" * 70)
    logger.info("MSLT CACHE BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info("Valid records : %d", len(dataframe))
    logger.info("Parquet       : %s", parquet_path)
    logger.info("CSV           : %s", csv_path)
    logger.info("Metadata JSON : %s", metadata_path)
    logger.info("Manifest      : %s", manifest_path)
    logger.info("SHA256        : %s", verification.sha256)
    logger.info("Audio files   : %d", len(list(raw_dir.glob("*.wav"))))
    logger.info("=" * 70)

    return parquet_path


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build the BridgeDEUX MSLT German Test cache."
    )

    parser.add_argument(
        "--zip_path",
        type=Path,
        default=Path(
            r"C:\Users\user\Downloads\MSLT_Corpus.zip"
        ),
        help="Path to MSLT_Corpus.zip",
    )

    args = parser.parse_args()

    ProjectConfig.initialize()

    build_mslt_cache(args.zip_path)


if __name__ == "__main__":
    main()