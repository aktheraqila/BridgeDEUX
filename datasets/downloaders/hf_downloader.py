"""
BridgeDEUX Core Framework
Downloader Subsystem

Module
------
datasets.downloaders.hf_downloader

Purpose
-------
Production-grade orchestrator for downloading Parquet shards.
Includes atomic `.part` workflow, retry logic, strict verification,
and configurable strict/best-effort error handling.
"""

from __future__ import annotations

import time
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List

from huggingface_hub import HfFileSystem, hf_hub_download

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from datasets.downloaders.exceptions import DiscoveryError, DownloadError, VerificationError
from datasets.downloaders.manifest import ManifestManager, ManifestRecord
from datasets.downloaders.verifier import FileVerifier, VerificationResult


# ============================================================
# Download Statistics
# ============================================================

@dataclass(slots=True)
class DownloadStatistics:
    """Runtime statistics collected during a downloader session."""
    discovered: int = 0
    downloaded: int = 0
    copied: int = 0
    skipped: int = 0
    verified: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0


# ============================================================
# Downloader Engine
# ============================================================

class HuggingFaceDownloader:
    """
    Downloads and verifies Parquet shards for a specific split.
    """

    def __init__(self, split: str, max_retries: int = 3, strict_mode: bool = False):
        self.split = split
        self.max_retries = max_retries
        self.strict_mode = strict_mode
        self.logger = BridgeLogger.get_logger(f"Downloader[{split}]")
        self.filesystem = HfFileSystem()
        self.statistics = DownloadStatistics()

        # ----------------------------------------------------
        # Path Configuration
        # ----------------------------------------------------
        self.target_dir = (
            ProjectConfig.RAW_DATA_DIR / 
            ProjectConfig.DATASET_CONFIGURATION / 
            self.split
        )
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # Manifest Initialization
        # ----------------------------------------------------
        manifest_path = ProjectConfig.REPORT_DIR / f"{split}_manifest.json"
        self.manifest = ManifestManager(manifest_path)
        
        self.logger.info(f"Initialized atomic downloader for '{split}' split (Strict Mode: {self.strict_mode}).")
        self.logger.info(f"Target Directory: {self.target_dir}")

    # ========================================================
    # Discovery
    # ========================================================

    def discover_shards(self) -> List[str]:
        """Discovers available Parquet shards directly from Hugging Face."""
        repo_path = f"datasets/{ProjectConfig.DATASET_REPOSITORY}/{ProjectConfig.DATASET_CONFIGURATION}"
        pattern = f"{repo_path}/{self.split}-*.parquet"

        self.logger.info("Scanning Hugging Face repository...")
        try:
            shards = sorted(self.filesystem.glob(pattern))
        except Exception as error:
            raise DiscoveryError(f"Unable to discover shards.\n\n{error}")

        if not shards:
            raise DiscoveryError(f"No shards found for split '{self.split}'.")

        filenames = [Path(shard).name for shard in shards]
        self.statistics.discovered = len(filenames)
        self.logger.info(f"Discovered {len(filenames)} shard(s).")
        
        return filenames

    # ========================================================
    # Sub-Routines
    # ========================================================

    def _download_to_cache(self, filename: str) -> Path:
        """Downloads the file securely to the HF cache with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                cached_path = hf_hub_download(
                    repo_id=ProjectConfig.DATASET_REPOSITORY,
                    filename=f"{ProjectConfig.DATASET_CONFIGURATION}/{filename}",
                    repo_type="dataset"
                )
                return Path(cached_path)
            except Exception as error:
                if attempt == self.max_retries:
                    raise DownloadError(f"HF Hub download failed after {self.max_retries} attempts.\n\n{error}")
                self.logger.warning(f"Download attempt {attempt} failed for '{filename}'. Retrying in {2**attempt}s...")
                time.sleep(2 ** attempt)

    def _copy_to_workspace(self, cached_path: Path, target_part_path: Path) -> None:
        """Copies the cached file into our isolated workspace with explicit checking."""
        if not cached_path.exists():
            raise DownloadError(f"Cached file does not exist at source destination: {cached_path}")
            
        try:
            shutil.copy2(cached_path, target_part_path)
            self.statistics.copied += 1
        except Exception as error:
            raise DownloadError(f"Failed to copy file to workspace.\n\n{error}")

    # ========================================================
    # Shard Processing (Atomic Workflow)
    # ========================================================

    def process_shard(self, filename: str) -> None:
        """Executes the atomic workflow: Check -> DL -> Copy -> Verify -> Rename."""
        final_path = self.target_dir / filename
        part_path = self.target_dir / f"{filename}.part"

        # 1. Check Manifest & File Existence
        record = self.manifest.get(filename)
        if record and record.get("status") == "SUCCESS":
            if final_path.exists():
                self.logger.info(f"Skipping '{filename}' (Already verified).")
                self.statistics.skipped += 1
                return
            else:
                self.logger.warning(f"Manifest indicates SUCCESS, but file missing. Redownloading '{filename}'.")

        self.logger.info(f"Processing '{filename}'...")
        start_time = time.perf_counter()

        try:
            # 2. Download with telemetry
            dl_start = time.perf_counter()
            cached_path = self._download_to_cache(filename)
            download_duration = time.perf_counter() - dl_start

            # Telemetry: Log downloaded file size
            size_mb = cached_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Downloaded file located in cache ({size_mb:.2f} MB)")

            # 3. Copy to .part
            self._copy_to_workspace(cached_path, part_path)

            # 4. Verify .part with hardcoded columns (respecting frozen config)
            ver_start = time.perf_counter()
            verification: VerificationResult = FileVerifier.verify(
                file_path=part_path,
                required_columns=[
                    "client_id", "file", "audio", "sentence", "translation", "id"
                ]
            )
            verification_duration = time.perf_counter() - ver_start

            # 5. Atomic Rename
            part_path.replace(final_path)

            # 6. Record Success
            elapsed = time.perf_counter() - start_time
            self.statistics.downloaded += 1
            self.statistics.verified += 1
            
            self.logger.info(
                f"✅ Verified & Saved '{filename}' ({elapsed:.2f}s total | "
                f"DL: {download_duration:.1f}s, Verify: {verification_duration:.1f}s | "
                f"{verification.row_count:,} rows)"
            )

            manifest_record = ManifestRecord(
                filename=filename,
                relative_path=str(final_path.relative_to(ProjectConfig.PROJECT_ROOT)),
                file_size_bytes=verification.file_size_bytes,
                file_size_mb=verification.file_size_mb,
                sha256=verification.sha256,
                row_count=verification.row_count,
                status="SUCCESS"
            )
            self.manifest.add(manifest_record)

        except (DownloadError, VerificationError) as error:
            self.statistics.failed += 1
            self.logger.error(f"❌ Failed processing '{filename}': {error}")
            if self.manifest.contains(filename):
                self.manifest.update_status(filename, "FAILED")
            
            if self.strict_mode:
                self.logger.error("Strict mode enabled. Aborting immediately execution.")
                raise error
                
        finally:
            # Guaranteed cleanup of stale .part files
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    self.logger.warning(f"Could not delete temporary file: {part_path}")

    # ========================================================
    # Orchestration
    # ========================================================

    def process_split(self) -> None:
        """Orchestrates the entire download loop for the split."""
        self.logger.info("=" * 70)
        self.logger.info(f"STARTING INGESTION ENGINE | SPLIT: {self.split.upper()}")
        self.logger.info("=" * 70)

        start = time.perf_counter()
        shards = self.discover_shards()

        for index, filename in enumerate(shards, start=1):
            self.logger.info(f"\n--- Shard {index}/{len(shards)} ---")
            self.process_shard(filename)

        self.statistics.elapsed_seconds = time.perf_counter() - start
        self.logger.info("\nIngestion loop completed.")

    def print_summary(self) -> None:
        """Outputs the final session statistics."""
        self.logger.info("=" * 70)
        self.logger.info("INGESTION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Split             : {self.split}")
        self.logger.info(f"Discovered Shards : {self.statistics.discovered}")
        self.logger.info(f"Successfully DL'd : {self.statistics.downloaded}")
        self.logger.info(f"Copied to Wkspc   : {self.statistics.copied}")
        self.logger.info(f"Skipped (Cached)  : {self.statistics.skipped}")
        self.logger.info(f"Failed            : {self.statistics.failed}")
        self.logger.info(f"Elapsed Time      : {self.statistics.elapsed_seconds:.2f} seconds")
        self.logger.info("=" * 70)


# ============================================================
# CLI Entry Point
# ============================================================

def main() -> None:
    ProjectConfig.initialize()
    
    parser = argparse.ArgumentParser(description="BridgeDEUX Parquet Downloader")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to download (e.g., test, validation, train)")
    parser.add_argument("--strict", action="store_true", help="Halt the engine immediately if any single shard fails")
    args = parser.parse_args()
    
    downloader = HuggingFaceDownloader(split=args.split, strict_mode=args.strict)
    
    try:
        downloader.process_split()
        downloader.print_summary()
    except Exception:
        downloader.logger.exception("Downloader terminated unexpectedly.")
        raise

if __name__ == "__main__":
    main()