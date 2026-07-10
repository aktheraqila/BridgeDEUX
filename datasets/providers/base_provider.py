"""
Abstract interface for dataset providers.

Every dataset provider in BridgeDEUX must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from datasets.providers.sample import DatasetInfo, DatasetSample


class DatasetProvider(ABC):
    """
    Base interface for all dataset providers.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetSample]:
        """
        Iterate over all dataset samples.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> DatasetSample:
        """
        Return the sample at the given index.
        """
        raise NotImplementedError