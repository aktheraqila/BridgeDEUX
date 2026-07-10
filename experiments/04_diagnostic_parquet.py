"""
Diagnostic: Parquet Remote Column Projection
Tests whether we can list files and read ONLY text columns 
over the network without downloading the heavy audio bytes.
"""

from huggingface_hub import HfFileSystem
import pandas as pd
import time

def run_diagnostic():
    print("=" * 60)
    print("Diagnostic 1: File System Check")
    print("=" * 60)
    
    fs = HfFileSystem()
    repo_path = "datasets/fixie-ai/covost2"
    
    # 1. Find the actual Parquet files
    try:
        print("Searching for Parquet files in repository...")
        all_files = fs.glob(f"{repo_path}/**/*.parquet")
        
        # Filter for the German-English test split
        test_files = [f for f in all_files if "de_en" in f and "test" in f]
        
        if not test_files:
            print("❌ Could not find de_en test files. Repository structure is different.")
            print(f"First 5 files found: {all_files[:5]}")
            return
            
        target_file = f"hf://{test_files[0]}"
        print(f"✅ Found exact file path: {target_file}")
        
    except Exception as e:
        print(f"❌ Failed to search repository: {e}")
        return

    print("\n" + "=" * 60)
    print("Diagnostic 2: Remote Column Read")
    print("=" * 60)
    
    # 2. Test reading only the text columns
    try:
        print(f"Attempting to read ONLY 'sentence' and 'translation' columns...")
        start_time = time.time()
        
        # Read the file, asking Pandas to ignore the audio column
        df = pd.read_parquet(
            target_file, 
            columns=["sentence", "translation"]
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ Success! Read {len(df)} rows in {elapsed:.2f} seconds.")
        print("-" * 60)
        print("First 3 rows:")
        print(df.head(3))
        print("-" * 60)
        
        if elapsed < 15:
            print("Conclusion: Column projection WORKS! The audio was bypassed.")
        else:
            print("Conclusion: It took a long time. It might be downloading the full file in the background.")
            
    except Exception as e:
        print(f"❌ Failed to read Parquet columns: {e}")

if __name__ == "__main__":
    run_diagnostic()