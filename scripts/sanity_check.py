import argparse
import time
import pandas as pd
from pathlib import Path

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from models.translators.marian_onnx import MarianONNXTranslator

logger = BridgeLogger.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sanity check for ONNX models.")
    parser.add_argument("--model", type=str, required=True, help="Full directory name of the model to test")
    args = parser.parse_args()

    ProjectConfig.initialize()

    model_dir = Path(ProjectConfig.MODEL_DIR) / "onnx" / args.model
    dataset_path = Path(ProjectConfig.CACHE_DIR) / "benchmark_subset_100.parquet"

    if not model_dir.exists():
        logger.error(f"[Model: {args.model}] Directory not found: {model_dir}")
        return

    logger.info(f"Initializing translator from: {model_dir}")
    translator = MarianONNXTranslator(onnx_model_dir=str(model_dir), provider="CPUExecutionProvider")
    
    try:
        translator.load()
        
        # Measure warm-up time explicitly
        start_warmup = time.perf_counter()
        translator.warm_up()
        warmup_time_ms = (time.perf_counter() - start_warmup) * 1000.0
        logger.info(f"[Model: {args.model}] Warm-up completed in {warmup_time_ms:.2f} ms")
        
        logger.info(f"Sampling first 3 deterministic rows from: {dataset_path.name}")
        df = pd.read_parquet(dataset_path)
        sample_df = df.head(3)

        print("\n" + "=" * 80)
        print("TRANSLATION SANITY CHECK")
        print("=" * 80)

        for idx, row in sample_df.iterrows():
            source = row["source_text"]
            expected = row["target_text"]
            
            result = translator.translate(source)
            
            print(f"\n[{idx}] DE (Source) : {source}")
            print(f"    EN (Expected) : {expected}")
            print(f"    EN (Actual)   : {result.translation}")
            print("-" * 80)
            print(f"    Tokens        : {result.input_tokens} In -> {result.output_tokens} Out")
            print(f"    Timing        : Tok: {result.tokenization_time_ms:.2f}ms | "
                  f"Gen: {result.generation_time_ms:.2f}ms | "
                  f"Dec: {result.decoding_time_ms:.2f}ms")
            print(f"    Total Latency : {result.total_time_ms:.2f}ms")
            
    except Exception as e:
        logger.error(f"[Model: {args.model}] Sanity check failed: {e}", exc_info=True)
    finally:
        logger.info("Unloading model and freeing resources...")
        translator.unload()
        logger.info("Sanity check gracefully terminated.")


if __name__ == "__main__":
    main()