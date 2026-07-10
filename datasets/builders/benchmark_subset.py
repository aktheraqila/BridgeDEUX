from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from datasets.providers.base_provider import DatasetProvider


logger = BridgeLogger.get_logger(__name__)


def build_subset(
    provider: DatasetProvider,
    subset_size: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a deterministic benchmark subset.
    """

    dataset_size = len(provider)

    if subset_size > dataset_size:
        raise ValueError(
            f"Subset size ({subset_size}) exceeds "
            f"dataset size ({dataset_size})."
        )

    rng = random.Random(seed)

    indices = rng.sample(range(dataset_size), subset_size)

    indices.sort()

    records = []

    for index in indices:
        sample = provider[index]

        records.append(
            {
                "id": sample.id,
                "source_text": sample.source_text,
                "target_text": sample.target_text,
                "client_id": sample.client_id,
                "file_name": sample.file_name,
            }
        )

    logger.info(
        "Created benchmark subset "
        "(samples=%d, seed=%d).",
        subset_size,
        seed,
    )

    return pd.DataFrame(records)


def save_subset(
    subset: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Save the benchmark subset.
    """

    if output_path is None:
        output_path = (
            ProjectConfig.CACHE_DIR /
            "benchmark_subset_100.parquet"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Benchmark subset written to %s",
        output_path,
    )

    return output_path