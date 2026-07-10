"""
BridgeDEUX

Download Manifest Manager

Purpose
-------
Maintain a persistent record of downloaded dataset shards.

Responsibilities
----------------
- Resume interrupted downloads
- Track verification results
- Maintain reproducibility metadata
- Persist manifest atomically
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bridge.logger import BridgeLogger
from bridge.metadata import MetadataCollector

logger = BridgeLogger.get_logger("Manifest")


# ============================================================
# Manifest Record
# ============================================================

@dataclass(slots=True)
class ManifestRecord:

    filename: str

    relative_path: str

    file_size_bytes: int

    file_size_mb: float

    sha256: str

    row_count: int

    status: str


# ============================================================
# Manifest Manager
# ============================================================

class ManifestManager:

    SCHEMA_VERSION = 1

    def __init__(self, manifest_path: Path):

        self.manifest_path = Path(manifest_path)

        self.metadata = MetadataCollector.as_dict()

        self.records: dict[str, dict] = {}

        self._load()

    # --------------------------------------------------------

    def _load(self) -> None:

        if not self.manifest_path.exists():
            return

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(file)

        self.metadata = payload.get(
            "metadata",
            self.metadata,
        )

        self.records = payload.get(
            "records",
            {},
        )

    # --------------------------------------------------------

    def save(self) -> None:

        payload = {

            "schema_version": self.SCHEMA_VERSION,

            "last_updated_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "metadata": self.metadata,

            "records": self.records,

        }

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self.manifest_path.with_suffix(".tmp")

        with open(
            temporary,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=4,
            )

        temporary.replace(self.manifest_path)

    # --------------------------------------------------------

    def contains(
        self,
        filename: str,
    ) -> bool:

        return filename in self.records

    # --------------------------------------------------------

    def get(
        self,
        filename: str,
    ) -> dict | None:

        return self.records.get(filename)

    # --------------------------------------------------------

    def list_records(self) -> list[dict]:

        return list(self.records.values())

    # --------------------------------------------------------

    def add(
        self,
        record: ManifestRecord,
    ) -> None:

        self.records[record.filename] = asdict(record)

        self.save()

        logger.info(
            f"Recorded '{record.filename}'."
        )

    # --------------------------------------------------------

    def update_status(
        self,
        filename: str,
        status: str,
    ) -> None:

        if filename not in self.records:
            return

        self.records[filename]["status"] = status

        self.save()

    # --------------------------------------------------------

    def remove(
        self,
        filename: str,
    ) -> None:

        if filename not in self.records:
            return

        del self.records[filename]

        self.save()

        logger.info(
            f"Removed '{filename}' from manifest."
        )