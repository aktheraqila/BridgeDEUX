from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class DatasetSample:
    id: str
    source_text: str
    target_text: str
    client_id: str | None = None
    file_name: str | None = None
    audio: dict | None = None

@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    split: str
    num_samples: int
    columns: tuple[str, ...]
