from models.asr.vosk import VoskASR
from bridge.config import ProjectConfig
from pathlib import Path
import numpy as np
import time

def run_vosk_smoke_test():
    # 1. Initialization
    ProjectConfig.initialize()
    model_path = Path(ProjectConfig.MODEL_DIR) / "vosk" / "vosk-model-small-de-0.15"
    
    # 2. Instantiate and Load
    print(f"Instantiating VoskASR with model path: {model_path}")
    engine = VoskASR(model_path)
    
    print("Loading model into memory...")
    start_load = time.perf_counter()
    engine.load()
    print(f"Model loaded in {(time.perf_counter() - start_load):.2f} seconds.")

    # 3. Create Audio (Dummy test to verify interface compliance)
    print("Creating dummy float32 audio array (16kHz, 1 second)...")
    # Note: Once this passes, replace `dummy_audio` with real data from your AudioProcessor
    dummy_audio = np.zeros(16000, dtype=np.float32)

    # 4. Transcribe
    print("Running transcription...")
    try:
        # FIXED: Strictly one argument to match the new BaseASR contract
        result = engine.transcribe(dummy_audio)
        
        print("\n--- Transcription Result ---")
        print(f"Model: {engine.model_name}")
        # FIXED: Only printing fields that actually exist in your ASRResult dataclass
        print(f"Text: '{result.transcription}'")
        print(f"Generation Time: {result.generation_time_ms:.2f} ms")
        print("----------------------------\n")
        print("Smoke test PASSED! (Interface contract successfully verified)")
        
    except Exception as e:
        print(f"Transcription FAILED: {e}")
        
    finally:
        # 5. Cleanup
        print("Unloading model...")
        engine.unload()
        print("Done.")

if __name__ == "__main__":
    run_vosk_smoke_test()