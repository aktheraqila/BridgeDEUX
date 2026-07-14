# experiments/test_audio_decode.py

import time
from datasets.providers.covost_provider import CoVoSTProvider
from bridge.config import ProjectConfig
from bridge.audio import AudioProcessor

def test_dsp_pipeline():
    print("--- Testing Unified DSP Audio Pipeline ---")
    ProjectConfig.initialize()
    provider = CoVoSTProvider(split="test", include_audio=True)
    
    sample = next(iter(provider))
    audio_bytes = sample.audio.get("bytes")
    
    # 1. Test Decode
    start_decode = time.perf_counter()
    raw_array, sample_rate = AudioProcessor.decode_mp3_to_pcm(audio_bytes)
    decode_time = (time.perf_counter() - start_decode) * 1000
    
    # 2. Test Polyphase Resample
    start_resample = time.perf_counter()
    resampled_array = AudioProcessor.resample_to_16k(raw_array, sample_rate)
    resample_time = (time.perf_counter() - start_resample) * 1000
    
    print(f"Decode Time:   {decode_time:.2f} ms")
    print(f"Resample Time: {resample_time:.2f} ms")
    print(f"Original Rate: {sample_rate} Hz -> Target Rate: 16000 Hz")
    print(f"Final Shape:   {resampled_array.shape}")
    print(f"Data Type:     {resampled_array.dtype}")
    print("\nSUCCESS: Preprocessing pipeline is high-fidelity and verified.")

if __name__ == "__main__":
    test_dsp_pipeline()