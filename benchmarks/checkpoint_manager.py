"""
BridgeDEUX Core Framework
Resilient Checkpoint Manager
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from benchmarks.exceptions import CheckpointError


class CheckpointManager:
    """
    Manages atomic JSONL Write-Ahead Logging, metadata state tracking, 
    and Parquet compilation for long-running benchmark experiments.
    """

    def __init__(
        self,
        model_identifier: str,
        checkpoint_interval: int = 25,
        output_dir: Path | None = None,
    ) -> None:

        self._model_id = model_identifier.lower().replace("/", "_")
        self._interval = checkpoint_interval

        base_dir = output_dir or ProjectConfig.BENCHMARK_DIR

        self._output_dir = base_dir / self._model_id
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger = BridgeLogger.get_logger(self.__class__.__name__)

        self._jsonl_path = self._output_dir / f"{self._model_id}_results.jsonl"
        self._parquet_path = self._output_dir / f"{self._model_id}_results.parquet"
        self._metadata_path = self._output_dir / f"{self._model_id}_metadata.json"
        
        # Lifecycle State
        self._is_finalized = False
        
        # State separation: Disk vs RAM
        self._persisted_ids: set[str] = set()
        self._buffered_ids: set[str] = set()
        self._buffer: list[dict[str, Any]] = []

    @property
    def processed_samples(self) -> int:
        return len(self._persisted_ids)

    @property
    def checkpoint_interval(self) -> int:
        return self._interval

    @property
    def has_pending_records(self) -> bool:
        """Returns True if there are uncommitted records in the memory buffer."""
        return len(self._buffer) > 0

    def load_completed_samples(self) -> set[str]:
        """
        Detects system state based on artifacts (JSONL/Parquet). 
        Ensures strict schema validation for both formats.
        """
        self._is_finalized = False
        
        self._persisted_ids.clear()
        self._buffered_ids.clear()
        self._buffer.clear()

        if self._jsonl_path.exists():
            self._logger.info("Found active JSONL log. Resuming...")
            self._persisted_ids = self._recover_from_jsonl()
        elif self._parquet_path.exists():
            self._logger.info("Found final Parquet artifact. Run already completed.")
            self._persisted_ids = self._recover_from_parquet()
        else:
            self._logger.info("No existing artifacts. Starting fresh run.")

        return set(self._persisted_ids)

    def _recover_from_jsonl(self) -> set[str]:
        completed_ids: set[str] = set()
        try:
            with open(self._jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if "sample_id" not in record:
                            raise CheckpointError(f"Invalid record at line {line_num}: missing 'sample_id'")
                        completed_ids.add(str(record["sample_id"]))
                    except json.JSONDecodeError:
                        self._logger.warning("Truncated JSONL record at line %d. Log ended.", line_num)
                        break
        except Exception as e:
            if isinstance(e, CheckpointError):
                raise
            raise CheckpointError(f"Failed to read JSONL: {e}") from e
        return completed_ids

    def _recover_from_parquet(self) -> set[str]:
        try:
            df = pd.read_parquet(self._parquet_path)
            if "sample_id" not in df.columns:
                raise CheckpointError("Parquet missing required column 'sample_id'.")
            return set(df["sample_id"].astype(str).tolist())
        except Exception as e:
            if isinstance(e, CheckpointError):
                raise
            raise CheckpointError(f"Failed to read Parquet: {e}") from e

    def save(self, record: dict[str, Any]) -> None:
        """Buffers a record and flushes automatically. Fails fast on duplicates/invalid schemas."""
        if self._is_finalized:
            raise CheckpointError("Cannot save after checkpoint has been finalized.")
            
        if "sample_id" not in record:
            raise CheckpointError("Cannot buffer record: missing required key 'sample_id'.")
            
        sid = str(record["sample_id"])
        
        if sid in self._persisted_ids or sid in self._buffered_ids:
            raise CheckpointError(f"Duplicate sample_id detected: {sid}. Record rejected.")
            
        self._buffered_ids.add(sid)
        self._buffer.append(record)
        
        if len(self._buffer) >= self._interval:
            self.flush()

    def flush(self) -> None:
        """
        Public API to forcefully flush the active buffer to disk.
        Safe to call multiple times.
        """
        self._flush()

    def _flush(self) -> None:
        """Appends buffer to JSONL log, safely clears RAM, and updates metadata."""
        if self._is_finalized or not self._buffer:
            return

        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                for record in self._buffer:
                    f.write(json.dumps(self._normalize_record(record), ensure_ascii=False) + "\n")
        except Exception as e:
            raise CheckpointError(f"IO error during checkpoint flush: {e}")

        # State transition strictly happens only after successful disk write
        self._persisted_ids.update(self._buffered_ids)
        self._buffer.clear()
        self._buffered_ids.clear()
        
        # Metadata is treated as best-effort.
        try:
            self._update_metadata("running")
        except Exception as e:
            self._logger.warning("Non-critical failure: could not update metadata tracking: %s", str(e))

    def finalize(self) -> None:
        """Compiles, validates, checks for duplicates, and creates a timestamped archive."""
        if self._buffer:
            self.flush()

        self._logger.info("Finalizing experiment artifacts...")

        try:
            if not self._jsonl_path.exists():
                if self._parquet_path.exists():
                    self._logger.info("Artifacts already finalized (Parquet exists, JSONL archived).")
                    self._is_finalized = True
                    return
                raise CheckpointError("No checkpoint exists to finalize.")

            if self._jsonl_path.stat().st_size == 0:
                raise CheckpointError("JSONL checkpoint file is empty. Nothing to finalize.")

            df_jsonl = pd.read_json(self._jsonl_path, lines=True)
            
            if df_jsonl.empty:
                raise CheckpointError("Parsed JSONL DataFrame is empty.")
            
            if "sample_id" not in df_jsonl.columns:
                raise CheckpointError("JSONL log missing 'sample_id' column.")

            if self._parquet_path.exists() and self._parquet_path.stat().st_size > 0:
                self._logger.info("Existing Parquet found. Merging historical state...")
                df_existing = pd.read_parquet(self._parquet_path)
                
                df_jsonl = pd.concat([df_existing, df_jsonl], ignore_index=True)
                
                df_jsonl = (
                    df_jsonl
                    .drop_duplicates(subset=["sample_id"], keep="last")
                    .sort_values(by="sample_id", kind="stable")
                    .reset_index(drop=True)
                )

            if not df_jsonl["sample_id"].is_unique:
                duplicates = df_jsonl[df_jsonl["sample_id"].duplicated()]["sample_id"].unique()
                raise CheckpointError(
                    f"Data integrity failure: found {len(duplicates)} duplicate "
                    f"sample_id(s) in JSONL log. Example duplicates: {duplicates[:5]}"
                )

            df_jsonl.to_parquet(self._parquet_path, index=False)
            csv_path = self._parquet_path.with_suffix('.csv')
            df_jsonl.to_csv(csv_path, index=False)
            
            df_parquet = pd.read_parquet(self._parquet_path)
            if len(df_parquet) != len(df_jsonl):
                raise CheckpointError(f"Row count mismatch: {len(df_jsonl)} vs {len(df_parquet)}")
            if set(df_parquet.columns) != set(df_jsonl.columns):
                raise CheckpointError("Column mismatch between JSONL and Parquet.")
                
            if len(set(df_parquet["sample_id"].astype(str))) != len(df_jsonl):
                raise CheckpointError("Data corruption detected during Parquet compilation: unique sample_id mismatch.")

            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            archive_path = self._jsonl_path.with_name(f"{self._jsonl_path.stem}_{timestamp}.jsonl.bak")
            try:
                self._jsonl_path.rename(archive_path)
            except Exception as e:
                raise CheckpointError(f"Failed to create timestamped JSONL archive: {e}") from e

            try:
                self._update_metadata("completed")
            except Exception as e:
                self._logger.warning("Final metadata update failed, but artifact creation succeeded: %s", str(e))
            
            self._is_finalized = True
            self._logger.info("Successfully validated and finalized artifacts.")
            
        except Exception as e:
            if isinstance(e, CheckpointError):
                raise
            raise CheckpointError(f"Finalization failed: {e}") from e

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {k: self._normalize_value(v) for k, v in record.items()}

    def _normalize_value(self, val: Any) -> Any:
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, (list, tuple)):
            return [self._normalize_value(v) for v in val]
        if isinstance(val, dict):
            return {k: self._normalize_value(v) for k, v in val.items()}
        if isinstance(val, (Path, UUID)):
            return str(val)
        if isinstance(val, Enum):
            return val.value
            
        raise CheckpointError(f"Type {type(val)} not JSON serializable")

    def _update_metadata(self, status: str) -> None:
        metadata = {
            "model_identifier": self._model_id,
            "schema_version": "1.0",
            "checkpoint_format": "jsonl-v1",
            "checkpoint_interval": self._interval,
            "processed_samples": len(self._persisted_ids),
            "status": status,
            "last_updated": datetime.now(UTC).isoformat(),
        }

        tmp_path = self._metadata_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            tmp_path.replace(self._metadata_path)
        except Exception as e:
            self._logger.warning("Metadata update failed: %s", str(e))
            tmp_path.unlink(missing_ok=True)