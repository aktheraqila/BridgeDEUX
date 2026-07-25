"""
BridgeDEUX - Translator API Contract Test (Production Integration)
Verifies state machine boundaries, explicit data typings, and strict 
adherence to the BaseTranslator structural interface.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.translators.marian_onnx import MarianONNXTranslator
from models.translators.exceptions import BridgeTranslatorError

def run_contract_test():
    print("=" * 80)
    print(" BRIDGEDEUX: TRANSLATOR API CONTRACT INTEGRATION TEST")
    print("=" * 80)

    # 1. Instantiate the component
    print("[1/5] Instantiating MarianONNXTranslator...")
    translator = MarianONNXTranslator()
    
    # Verify initial lifecycle state
    if translator.is_loaded():
        print("❌ Contract Violation: Component reporting is_loaded before initialization.")
        sys.exit(1)

    try:
        # 2. Trigger hardware allocation context
        print("[2/5] Initializing underlying ONNX Runtime graph sessions...")
        translator.load()
        
        if not translator.is_loaded():
            print("❌ Contract Violation: Component failed to toggle is_loaded status true.")
            sys.exit(1)
            
        print(f"  • Registered Model Identity: {translator.model_name}")
        print(f"  • Registered Version/Backend: {translator.model_version}")
        print(f"  • Registered Execution Target: {translator.device}")

        # 3. Discharging a payload through the interface boundaries
        test_payload = "Guten Morgen. Wie geht es dir heute?"
        print(f"[3/5] Discharging single inference transaction: '{test_payload}'...")
        
        result = translator.translate(test_payload)
        
        # 4. Programmatic Data Structure Affirmation
        print("[4/5] Executing programmatic validation of the resulting dataclass structure...")
        
        # Assert structural integrity metrics
        if result.model_name != translator.model_name:
            print("❌ Contract Error: Resulting dataclass model_name field mismatch.")
            sys.exit(1)
            
        if not result.translation.strip():
            print("❌ Contract Error: Resulting string translation payload is empty.")
            sys.exit(1)
            
        if result.total_time_ms <= 0 or result.generation_time_ms <= 0:
            print("❌ Contract Error: Telemetry latency engines reporting invalid zero values.")
            sys.exit(1)
            
        print(f"  • Verified String Result: \"{result.translation}\"")
        print(f"  • Core Graph Execution Latency: {result.generation_time_ms:.2f} ms")
        print(f"  • Cumulative Execution Latency: {result.total_time_ms:.2f} ms")

    except BridgeTranslatorError as bte:
        print(f"❌ Framework Integration Failure: Component raised custom exception: {bte}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unhandled Engine Failure: Component crashed standard runtime loop: {e}")
        sys.exit(1)
        
    finally:
        # 5. Clear execution layer context cleanly
        print("[5/5] Releasing execution provider hooks via final context teardown...")
        translator.unload()

    # Final post-condition check
    if translator.is_loaded():
        print("❌ Contract Violation: Component failed to release is_loaded state after unload call.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 SUCCESS: MarianONNXTranslator API Contract fully verified for Stage B deployment.")
    print("=" * 80)

if __name__ == "__main__":
    run_contract_test()