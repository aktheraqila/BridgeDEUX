#!/usr/bin/env python3
"""
BridgeDEUX: Step 6B - Export 100 Mobile Benchmark Audio Files
============================================================
Extracts the exact 100 patch audio samples using the framework-native 
AudioProcessor (soundfile + scipy) and exports 16kHz mono 16-bit PCM WAV files.
"""

import sys
from pathlib import Path

# Ensure root directory is on sys.path for local module imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from bridge.config import ProjectConfig
from bridge.audio import AudioProcessor
from datasets.providers.covost_provider import CoVoSTProvider


def main():
    analysis_dir = repo_root / "analysis"
    patch_path = analysis_dir / "qirg_cohort_patch_100.parquet"
    out_audio_dir = analysis_dir / "mobile_benchmark_audio_100"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    if not patch_path.exists():
        print(f"FATAL: Missing patch artifact at {patch_path}")
        sys.exit(1)

    df_patch = pd.read_parquet(patch_path)
    target_ids = set(df_patch["sample_id"].astype(str).tolist())

    print("=" * 80)
    print(" STEP 6B: EXPORTING 100 BENCHMARK AUDIO FILES (16kHz Mono WAV)")
    print("=" * 80)
    print(f"Target Samples : {len(target_ids)}")

    ProjectConfig.initialize()
    provider = CoVoSTProvider(split="test", include_audio=True)
    
    extracted_count = 0
    pbar = tqdm(total=len(target_ids), desc="Exporting Audio")

    for sample in provider:
        sid = str(sample.id)
        if sid in target_ids:
            dst_path = out_audio_dir / f"{sid}.wav"

            if not dst_path.exists():
                audio_bytes = sample.audio.get("bytes") if sample.audio else None
                if not audio_bytes:
                    print(f"\n[!] Warning: Missing audio bytes for {sid}")
                    continue

                # 1. Native MP3 -> Float32 Mono array
                raw_array, sample_rate = AudioProcessor.decode_mp3_to_pcm(audio_bytes)

                # 2. Native Polyphase Anti-Aliasing Resample -> 16,000 Hz
                resampled_array = AudioProcessor.resample_to_16k(raw_array, sample_rate)

                # 3. Write standard 16-bit linear PCM WAV
                sf.write(str(dst_path), resampled_array, 16000, subtype="PCM_16")

            extracted_count += 1
            pbar.update(1)

            if extracted_count == len(target_ids):
                break

    pbar.close()

    exported_files = list(out_audio_dir.glob("*.wav"))
    print("-" * 80)
    print(f"Successfully Exported : {len(exported_files)} / {len(target_ids)} files")
    print(f"Destination Path      : {out_audio_dir.resolve()}")
    print("=" * 80)

    if len(exported_files) != 100:
        print(f"ERROR: Expected 100 files, found {len(exported_files)}.")
        sys.exit(1)


if __name__ == "__main__":
    main()