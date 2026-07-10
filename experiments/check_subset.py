import pandas as pd

df = pd.read_parquet("datasets/cache/benchmark_subset_100.parquet")

print(df.columns.tolist())
print()
print(df.head())