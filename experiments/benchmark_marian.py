"""
BridgeDEUX
MarianMT Benchmark Runner

Runs MarianMT on the deterministic benchmark subset and
stores the inference results as a Parquet file.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

from models.translators.marian import MarianTranslator
from models.translators.exceptions import TranslatorError


BENCHMARK_FILE = (
    ProjectConfig.CACHE_DIR /
    "benchmark_subset_100.parquet"
)

RESULT_DIR = ProjectConfig.BENCHMARK_DIR

RESULT_FILE = RESULT_DIR / "marian_results.parquet"


def main() -> None:

    ProjectConfig.initialize()

    logger = BridgeLogger.get_logger("BenchmarkMarian")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark subset...")

    benchmark = pd.read_parquet(BENCHMARK_FILE)

    logger.info("Benchmark samples: %d", len(benchmark))

    translator = MarianTranslator()

    try:

        logger.info("Loading MarianMT...")

        translator.load()

        rows = []

        total = len(benchmark)

        for index, sample in benchmark.iterrows():

            print(
                f"[{index + 1}/{total}] "
                f"{sample['id']}"
            )

            result = translator.translate(sample["source_text"])

            record = asdict(result)

            record["sample_id"] = sample["id"]
            record["reference_translation"] = sample["target_text"]

            rows.append(record)

        results = pd.DataFrame(rows)

        results.to_parquet(
            RESULT_FILE,
            index=False
        )

        logger.info(
            "Benchmark completed successfully."
        )

        logger.info(
            "Results written to %s",
            RESULT_FILE
        )

        print("\n======================================")
        print("Benchmark Complete")
        print("======================================")
        print(f"Samples : {len(results)}")
        print(f"Output  : {RESULT_FILE}")

        print("\nAverage Latency")

        print(
            f"{results['total_time_ms'].mean():.2f} ms"
        )

    except TranslatorError as e:

        logger.exception(str(e))
        raise

    finally:

        translator.unload()


if __name__ == "__main__":
    main()