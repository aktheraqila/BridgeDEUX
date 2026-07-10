from datasets.builders.benchmark_subset_builder import BenchmarkSubsetBuilder
from datasets.providers.covost_provider import CoVoSTProvider

provider = CoVoSTProvider(split="test")

builder = BenchmarkSubsetBuilder(
    provider=provider,
    seed=42,
    subset_size=100,
)

path = builder.save()

print(path)