import pandas as pd
from pathlib import Path

# Adjust this path if your raw CoVoST2 shards are stored elsewhere
raw_dir = Path("datasets/raw/de_en/test")

if raw_dir.exists():
    parquet_files = list(raw_dir.glob("*.parquet"))
    if parquet_files:
        target_file = parquet_files[0]
        print(f"--- Inspecting {target_file.name} ---")
        
        df = pd.read_parquet(target_file)
        print("\nColumns:", df.columns.tolist())
        
        if "audio" in df.columns:
            audio_data = df.iloc[0]["audio"]
            print(f"\nAudio column type for first row: {type(audio_data)}")
            
            # Unpack the Hugging Face Audio dict securely
            if isinstance(audio_data, dict):
                print("Audio dict keys:", audio_data.keys())
                for key, value in audio_data.items():
                    if isinstance(value, bytes):
                        print(f"  - {key}: [Binary Data, {len(value)} bytes]")
                    elif hasattr(value, 'shape'):
                        print(f"  - {key}: [Numpy Array, shape: {value.shape}]")
                    else:
                        print(f"  - {key}: {value}")
        else:
            print("\nWARNING: 'audio' column not found in raw dataset.")
    else:
        print(f"No .parquet files found in {raw_dir}")
else:
    print(f"Directory not found: {raw_dir.resolve()}")