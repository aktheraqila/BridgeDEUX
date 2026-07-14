from pathlib import Path
import pandas as pd

# Define paths
marian_path = Path("results/marianmt_covost2_de_en_test/marianmt_covost2_de_en_test_results.parquet")
m2m_path = Path("results/m2m100_covost2_de_en_test/m2m100_covost2_de_en_test_results.parquet")

# Load data
marian_df = pd.read_parquet(marian_path)
m2m_df = pd.read_parquet(m2m_path)

# Merge datasets on source text to compare row-by-row
merged = pd.merge(
    marian_df[['source_text', 'reference_translation', 'translation']], 
    m2m_df[['source_text', 'translation']], 
    on='source_text', 
    suffixes=('_marian', '_m2m')
)

# Extract a deterministic sample of 40 sentences for review
sample_df = merged.sample(n=40, random_state=42)

# Save to CSV for easy spreadsheet opening
output_path = Path("mt_qualitative_analysis_sample.csv")
sample_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"[SUCCESS] Qualitative sample generated at: {output_path.resolve()}")