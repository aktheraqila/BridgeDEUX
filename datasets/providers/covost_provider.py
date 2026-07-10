"""
BridgeDEUX

Concrete provider for the locally verified CoVoST2 dataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from datasets.providers.base_provider import DatasetProvider
from datasets.providers.sample import DatasetInfo, DatasetSample


class CoVoSTProvider(DatasetProvider):
    """
    Read-only provider for the locally verified CoVoST2 dataset.
    """

    TARGET_COLUMNS = (
        "id",
        "sentence",
        "translation",
        "client_id",
        "file",
    )

    def __init__(self, split: str = "test") -> None:
        self._logger = BridgeLogger.get_logger(self.__class__.__name__)

        self._split = split

        self._dataset_dir = (
            ProjectConfig.RAW_DATA_DIR
            / ProjectConfig.DATASET_CONFIGURATION
            / split
        )

        if not self._dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset split does not exist: {self._dataset_dir}"
            )

        self._shards = sorted(self._dataset_dir.glob("*.parquet"))

        if not self._shards:
            raise FileNotFoundError(
                f"No Parquet shards found in {self._dataset_dir}"
            )

        self._cache: pd.DataFrame | None = None

        self._logger.info(
            "Initialized CoVoSTProvider "
            "(split=%s, shards=%d)",
            split,
            len(self._shards),
        )

    def _load_cache(self) -> None:
        """
        Load all text columns into memory.

        This cache is used only for random access.
        Audio is intentionally excluded.
        """

        if self._cache is not None:
            return

        frames = []

        for shard in self._shards:
            frames.append(
                pd.read_parquet(
                    shard,
                    columns=list(self.TARGET_COLUMNS),
                )
            )

        self._cache = pd.concat(
            frames,
            ignore_index=True,
        )

        self._logger.info(
            "Loaded %d samples into cache.",
            len(self._cache),
        )

    @staticmethod
    def _row_to_sample(row) -> DatasetSample:
        return DatasetSample(
            id=str(row.id),
            source_text=str(row.sentence),
            target_text=str(row.translation),
            client_id=(
                None
                if pd.isna(row.client_id)
                else str(row.client_id)
            ),
            file_name=(
                None
                if pd.isna(row.file)
                else str(row.file)
            ),
        )

    def __len__(self) -> int:
        self._load_cache()
        return len(self._cache)
       
    def __getitem__(self, index: int) -> DatasetSample:
        """
        Return the dataset sample at the given index.
        """

        self._load_cache()

        if index < 0 or index >= len(self._cache):
            raise IndexError(
                f"Sample index {index} is out of range."
            )

        row = self._cache.iloc[index]

        return DatasetSample(
            id=str(row["id"]),
            source_text=str(row["sentence"]),
            target_text=str(row["translation"]),
            client_id=(
                None
                if pd.isna(row["client_id"])
                else str(row["client_id"])
            ),
            file_name=(
                None
                if pd.isna(row["file"])
                else str(row["file"])
            ),
        )

    def __iter__(self) -> Iterator[DatasetSample]:
        """
        Lazily iterate over the dataset one shard at a time.
        """

        self._logger.info(
            "Beginning sequential dataset iteration."
        )

        for shard in self._shards:

            dataframe = pd.read_parquet(
                shard,
                columns=list(self.TARGET_COLUMNS),
            )

            for row in dataframe.itertuples(index=False):
                yield self._row_to_sample(row)

    def get_info(self) -> DatasetInfo:
        """
        Return metadata describing the dataset.
        """

        self._load_cache()

        return DatasetInfo(
            name=ProjectConfig.DATASET_REPOSITORY,
            split=self._split,
            num_samples=len(self._cache),
            columns=self.TARGET_COLUMNS,
        )