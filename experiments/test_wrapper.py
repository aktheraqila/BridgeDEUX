import sys
from pathlib import Path

# Force Python to recognize the project root directory
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.translators.marian_onnx import MarianONNXTranslator

def run_wrapper_test():
    print("=" * 60)
    print(" LAYER 4 VALIDATION: MarianONNXTranslator Wrapper")
    print("=" * 60)

    print("[1/4] Instantiating wrapper...")
    translator = MarianONNXTranslator(onnx_model_dir="models/onnx/opus_mt_de_en")

    try:
        print("[2/4] Calling .load()...")
        translator.load()
        assert translator.is_loaded(), "is_loaded() should be True after load()"

        print("[3/4] Calling .translate()...")
        result = translator.translate("Guten Morgen. Wie geht es dir heute?")

        print("\n--- TranslationResult Object ---")
        print(f"Model:       {result.model_name} [{result.model_version}]")
        print(f"Source:      {result.source_text}")
        print(f"Translation: {result.translation}")
        print(f"Generation:  {result.generation_time_ms:.2f} ms")
        print(f"Total Time:  {result.total_time_ms:.2f} ms")
        print("--------------------------------\n")

        assert result.translation.strip() != "", "Translation is empty!"
        assert result.total_time_ms > 0, "Total time metric is missing!"
        assert result.generation_time_ms > 0, "Generation time metric is missing!"
        
        print("✅ SUCCESS: Wrapper contract is fully validated.")

    finally:
        print("[4/4] Calling .unload()...")
        translator.unload()
        # The new check added based on the review
        assert not translator.is_loaded(), "is_loaded() should be False after unload()"

if __name__ == "__main__":
    run_wrapper_test()