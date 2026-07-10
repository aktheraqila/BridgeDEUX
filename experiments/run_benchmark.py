"""
BridgeDEUX Core Framework
Unified Benchmark Runner

Executes inference over the deterministic benchmark subset
using any BridgeDEUX translator implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

from models.translators.base_translator import BaseTranslator
from models.translators.exceptions import TranslatorError


def create_translator(model: str) -> BaseTranslator:
    """
    Factory for BridgeDEUX translators.
    """

    if model == "marian":
        from models.translators.marian import MarianTranslator

        return MarianTranslator()

    if model == "m2m100":
        from models.translators.m2m100 import M2M100Translator

        return M2M100Translator(
            source_lang="de",
            target_lang="en",
        )

    raise ValueError(f"Unsupported model: {model}")


def main() -> None:

    # ---------------------------------------------------------
    # CLI
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="BridgeDEUX Unified Benchmark Runner"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "marian",
            "m2m100",
        ],
        help="Translator to benchmark.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Framework Initialization
    # ---------------------------------------------------------

    ProjectConfig.initialize()

    logger = BridgeLogger.get_logger(
        f"Benchmark_{args.model.upper()}"
    )

    benchmark_file = (
        ProjectConfig.CACHE_DIR
        / "benchmark_subset_100.parquet"
    )

    result_dir = ProjectConfig.BENCHMARK_DIR

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        result_dir
        / f"{args.model}_results.parquet"
    )

    logger.info("Loading benchmark subset...")

    benchmark = pd.read_parquet(
        benchmark_file
    )

    total = len(benchmark)

    logger.info(
        "Benchmark samples: %d",
        total,
    )

    translator = create_translator(
        args.model
    )

    try:

        logger.info(
            "Loading %s...",
            translator.model_name(),
        )

        translator.load()

        # ---------------------------------------------
        # Warm-up
        # ---------------------------------------------

        logger.info(
            "Executing warm-up inference..."
        )

        translator.translate(
            "Dies ist ein Test."
        )

        # ---------------------------------------------
        # Benchmark
        # ---------------------------------------------

        rows = []

        for index, sample in benchmark.iterrows():

            print(
                f"[{index + 1}/{total}] "
                f"{sample['id']}"
            )

            try:

                result = translator.translate(
                    sample["source_text"]
                )

                record = asdict(result)

                record["sample_id"] = sample["id"]
                record["reference_translation"] = sample["target_text"]

                rows.append(record)

            except TranslatorError as e:

                logger.error(
                    "Sample %s failed: %s",
                    sample["id"],
                    str(e),
                )

                rows.append(
                    {
                        "model_name": translator.model_name(),
                        "model_version": translator.model_version(),
                        "source_text": sample["source_text"],
                        "translation": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "tokenization_time_ms": None,
                        "generation_time_ms": None,
                        "decoding_time_ms": None,
                        "total_time_ms": None,
                        "sample_id": sample["id"],
                        "reference_translation": sample["target_text"],
                        "error": str(e),
                    }
                )

        # ---------------------------------------------
        # Save Results
        # ---------------------------------------------

        results = pd.DataFrame(rows)

        results.to_parquet(
            result_file,
            index=False,
        )

        successful = (
            results["translation"]
            .notna()
            .sum()
        )

        failed = total - successful

        average_latency = (
            results["total_time_ms"]
            .dropna()
            .mean()
        )

        logger.info(
            "Benchmark completed successfully."
        )

        logger.info(
            "Results written to %s",
            result_file,
        )

        print("\n" + "=" * 50)
        print(" BENCHMARK COMPLETE")
        print("=" * 50)
        print(f"Model             : {translator.model_name()}")
        print(f"Checkpoint        : {translator.model_version()}")
        print(f"Samples           : {total}")
        print(f"Successful        : {successful}")
        print(f"Failed            : {failed}")
        print(f"Average Latency   : {average_latency:.2f} ms")
        print(f"Output            : {result_file}")
        print("=" * 50)

    except Exception:

        logger.exception(
            "Critical benchmark failure."
        )

        raise

    finally:

        translator.unload()


if __name__ == "__main__":
    main()