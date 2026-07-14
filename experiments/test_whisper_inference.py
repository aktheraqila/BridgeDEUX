# experiments/test_whisper_inference.py

import time
from datasets.providers.covost_provider import CoVoSTProvider
from bridge.config import ProjectConfig
from bridge.audio import AudioProcessor
from models.asr.whisper_cpp import WhisperCppASR

def run_isolated_telemetry_test() -> None:
    print("=== BridgeDEUX Multimodal Speech Telemetry Smoke Test ===")
    
    # Initialize framework filesystem contexts
    ProjectConfig.initialize()
    
    # 1. Pipeline Input Phase
    provider = CoVoSTProvider(split="test", include_audio=True)
    sample = next(iter(provider))
    audio_bytes = sample.audio.get("bytes")
    
    if not audio_bytes:
        raise ValueError("Target test shard contains an invalid or empty audio byte payload.")

    # 2. Reusable DSP Preprocessing Phase
    print("\n[DSP] Initializing audio front-end processing...")
    start_decode = time.perf_counter()
    raw_array, sample_rate = AudioProcessor.decode_mp3_to_pcm(audio_bytes)
    decode_time_ms = (time.perf_counter() - start_decode) * 1000
    
    start_resample = time.perf_counter()
    pcm_16k = AudioProcessor.resample_to_16k(raw_array, sample_rate)
    resample_time_ms = (time.perf_counter() - start_resample) * 1000
    
    print(f"  └─ Decode Latency:    {decode_time_ms:.2f} ms")
    print(f"  └─ Resample Latency:  {resample_time_ms:.2f} ms")
    print(f"  └─ Target Array Area: {pcm_16k.shape} @ 16000 Hz")

    # 3. Isolated Resource Allocation & Initialization Phase
    print("\n[Engine] Allocation phase initialization...")
    asr_model = WhisperCppASR(model_size="base", n_threads=4)
    
    try:
        asr_model.load()
        print(f"  └─ Cold Boot Weight Initialization: {asr_model.load_time_ms:.2f} ms")
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Execution Halted: {e}")
        print(f"Please ensure you place 'ggml-base.bin' inside your configured models/whisper directory.")
        return

    # 4. Neural Execution Inference Phase
    print("\n[Inference] Executing acoustic grid forward pass...")
    result = asr_model.transcribe(pcm_16k)
    
    # 5. Scientific Telemetry Summary (Fix 3: Synchronized with MT terminologies)
    print("\n================ TELEMETRY OVERVIEW ================")
    print(f"Model Architecture:  {asr_model.model_name}")
    print(f"Model Load Time:     {asr_model.load_time_ms:.2f} ms")
    print(f"Front-End DSP Time:  {(decode_time_ms + resample_time_ms):.2f} ms")
    print(f"Inference Time:      {result.generation_time_ms:.2f} ms")
    print(f"Total Pipeline Time: {(decode_time_ms + resample_time_ms + result.generation_time_ms):.2f} ms")
    print("----------------------------------------------------")
    print(f"Ground Truth Label:  {sample.source_text}")
    print(f"Hypothesis Output:   {result.transcription}")
    print("====================================================")
    
    asr_model.unload()
    print("Telemetry tracking verified. Context cleared successfully.")

if __name__ == "__main__":
    run_isolated_telemetry_test()