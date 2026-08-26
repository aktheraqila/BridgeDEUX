"""
BridgeDEUX
Provider for the MSLT German ASR evaluation dataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from datasets.providers.base_provider import DatasetProvider
from datasets.providers.sample import DatasetInfo, DatasetSample


class MSLTASRProvider(DatasetProvider):
    """Read-only provider for the MSLT German ASR cache."""

    TARGET_COLUMNS = (
        "id",
        "audio_path",
        "t1_reference",
        "t2_reference",
        "dataset",
        "split",
        "language",
    )

    def __init__(
        self,
        split: str = "test",
        include_audio: bool = False,
        manifest_path: str | Path | None = None,
    ) -> None:
        self._logger = BridgeLogger.get_logger(
            self.__class__.__name__
        )

        self._split = split
        self._include_audio = include_audio

        self._manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else None
        )

        self._cache_path = (
            ProjectConfig.RAW_DATA_DIR.parent
            / "cache"
            / "mslt"
            / "de_en"
            / split
            / "mslt_de_asr_test.parquet"
        )

        if not self._cache_path.exists():
            raise FileNotFoundError(
                f"MSLT ASR cache does not exist: "
                f"{self._cache_path}"
            )

        self._cache: pd.DataFrame | None = None

        self._logger.info(
            "Initialized MSLTASRProvider (split=%s)",
            split,
        )

    @staticmethod
    def _canonical_id(value: str) -> str:
        """
        Normalize numeric sample IDs for comparison.

        Examples:
            '0001' -> '1'
            '0010' -> '10'
            '2275' -> '2275'

        Non-numeric IDs are preserved after stripping whitespace.
        """
        value = value.strip()

        try:
            return str(int(value))
        except ValueError:
            return value

    def _load_cache(self) -> None:
        if self._cache is not None:
            return

        self._cache = pd.read_parquet(
            self._cache_path,
            columns=list(self.TARGET_COLUMNS),
        )

        if self._manifest_path is not None:
            if not self._manifest_path.exists():
                raise FileNotFoundError(
                    f"MSLT evaluation manifest does not exist: "
                    f"{self._manifest_path}"
                )

            manifest_ids = [
                line.strip()
                for line in self._manifest_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            manifest_canonical = [
                self._canonical_id(sample_id)
                for sample_id in manifest_ids
            ]

            if len(manifest_canonical) != len(
                set(manifest_canonical)
            ):
                raise ValueError(
                    "MSLT evaluation manifest contains duplicate "
                    "sample IDs."
                )

            self._cache["_canonical_id"] = (
                self._cache["id"]
                .astype(str)
                .map(self._canonical_id)
            )

            cache_canonical_ids = set(
                self._cache["_canonical_id"]
            )

            missing_ids = sorted(
                set(manifest_canonical)
                - cache_canonical_ids
            )

            if missing_ids:
                raise ValueError(
                    f"MSLT evaluation manifest contains "
                    f"{len(missing_ids)} IDs missing from the cache. "
                    f"Examples: {missing_ids[:5]}"
                )

            order = {
                sample_id: position
                for position, sample_id in enumerate(
                    manifest_canonical
                )
            }

            self._cache = (
                self._cache[
                    self._cache["_canonical_id"].isin(
                        manifest_canonical
                    )
                ]
                .assign(
                    _manifest_order=lambda df:
                        df["_canonical_id"].map(order)
                )
                .sort_values("_manifest_order")
                .drop(
                    columns=[
                        "_canonical_id",
                        "_manifest_order",
                    ]
                )
                .reset_index(drop=True)
            )

            if len(self._cache) != len(manifest_ids):
                raise ValueError(
                    "MSLT evaluation manifest and cache produced "
                    "different sample counts."
                )

            self._logger.info(
                "Applied evaluation manifest: %d samples.",
                len(self._cache),
            )

        self._logger.info(
            "Loaded %d ASR samples into cache.",
            len(self._cache),
        )

    def _row_to_sample(self, row) -> DatasetSample:
        audio_data = None

        if self._include_audio:
            audio_path = Path(row.audio_path)

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"MSLT ASR audio file not found: "
                    f"{audio_path}"
                )

            audio_data = {
                "bytes": audio_path.read_bytes()
            }

        return DatasetSample(
            id=str(row.id),
            # ASR reference = MSLT T1 verbatim transcript.
            source_text=str(row.t1_reference),
            # T2 is retained for diagnostic/reference comparison.
            target_text=str(row.t2_reference),
            client_id=None,
            file_name=str(row.audio_path),
            audio=audio_data,
        )

    def __len__(self) -> int:
        self._load_cache()
        return len(self._cache)

    def __getitem__(self, index: int) -> DatasetSample:
        self._load_cache()

        if index < 0 or index >= len(self._cache):
            raise IndexError(
                f"Sample index {index} is out of range."
            )

        row = next(
            self._cache.iloc[[index]].itertuples(
                index=False
            )
        )

        return self._row_to_sample(row)

    def __iter__(self) -> Iterator[DatasetSample]:
        self._logger.info(
            "Beginning sequential MSLT ASR iteration."
        )

        self._load_cache()

        for row in self._cache.itertuples(index=False):
            yield self._row_to_sample(row)

    def get_info(self) -> DatasetInfo:
        self._load_cache()

        return DatasetInfo(
            name="MSLT-ASR",
            split=self._split,
            num_samples=len(self._cache),
            columns=self.TARGET_COLUMNS,
        )