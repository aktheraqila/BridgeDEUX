import pandas as pd

dataset = pd.read_parquet(
    "datasets/cache/covost2_de_en_test.parquet"
)

results = pd.read_parquet(
    "results/marianmt_covost2_de_en_test/marianmt_covost2_de_en_test_results.parquet"
)

dataset_ids = set(dataset["id"].astype(str))
result_ids = set(results["sample_id"].astype(str))

missing = sorted(dataset_ids - result_ids)

print(f"Dataset rows : {len(dataset)}")
print(f"Dataset IDs  : {len(dataset_ids)}")
print(f"Result rows  : {len(results)}")
print(f"Result IDs   : {len(result_ids)}")
print(f"Missing IDs  : {len(missing)}")

for sample_id in missing:
    print(sample_id)