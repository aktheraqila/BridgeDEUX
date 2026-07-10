from bridge.config import ProjectConfig
from datasets.builders.benchmark_subset import (
    build_subset,
    save_subset,
)
from datasets.providers.covost_provider import CoVoSTProvider


def main():

    ProjectConfig.initialize()

    provider = CoVoSTProvider()

    subset = build_subset(
        provider,
        subset_size=100,
        seed=42,
    )

    print(subset.head())

    output = save_subset(subset)

    print(f"\nSaved to:\n{output}")


if __name__ == "__main__":
    main()