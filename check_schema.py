import pandas as pd

df = pd.read_parquet(
    "datasets/raw/de_en/test/test-00000-of-00008.parquet"
)

print(df.columns.tolist())
print()
print(df.dtypes)