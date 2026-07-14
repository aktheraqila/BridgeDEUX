from pathlib import Path

from bridge.config import ProjectConfig
from datasets.builders.benchmark_subset import (
    build_subset,
    save_subset,
)
from datasets.providers.covost_provider import CoVoSTProvider


def main():
    ProjectConfig.initialize()

    provider = CoVoSTProvider()

    total_samples = len(provider)

    print(f"Dataset size: {total_samples:,}")

    benchmark = build_subset(
        provider=provider,
        subset_size=total_samples,
        seed=42,
    )

    output = save_subset(
        benchmark,
        output_path=(
            ProjectConfig.CACHE_DIR /
            "covost2_de_en_test.parquet"
        ),
    )

    print()
    print("Saved:")
    print(output)


if __name__ == "__main__":
    main()