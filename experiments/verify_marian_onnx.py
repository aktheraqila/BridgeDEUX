# experiments/verify_marian_onnx.py
from __future__ import annotations

import time
import optimum
import onnxruntime
import transformers
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer
from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from importlib.metadata import version

def verify_onnx_export() -> None:
    # Initialize central project configuration
    ProjectConfig.initialize()
    logger = BridgeLogger.get_logger("ONNX_POC")
    
    # 1. Environment Provenance
    logger.info("Environment Version Check:")
    logger.info("Transformers: %s", transformers.__version__)
    logger.info("ONNX Runtime: %s", onnxruntime.__version__)
    logger.info("Transformers: %s", transformers.__version__)
    logger.info("Optimum: %s", version("optimum"))
    logger.info("ONNX Runtime: %s", onnxruntime.__version__)

    # 2. Hardcoded identifier for disposable POC
    model_id = "Helsinki-NLP/opus-mt-de-en"
    
    # Use ProjectConfig for deterministic pathing
    save_directory = ProjectConfig.MODEL_DIR / "onnx" / "marian"
    save_directory.mkdir(parents=True, exist_ok=True)
    
    test_text = "Zieht euch bitte draußen die Schuhe aus."

    logger.info("Starting Marian ONNX Deployment Verification POC")
    logger.info("Target Model ID: %s", model_id)
    logger.info("Output Directory: %s", save_directory)

    # 3. Tokenizer Allocation
    logger.info("Loading tokenizer engine...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_id)
    except Exception as e:
        logger.error("Failed to load tokenizer from pretrained checkpoint.")
        raise

    # 4. Defensive ONNX Graph Export
    logger.info("Initiating PyTorch to ONNX graph translation via Optimum...")
    try:
        # export=True forces the conversion from PyTorch to ONNX graph structures
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
    except Exception as e:
        logger.exception("CRITICAL: ONNX graph tracing or compilation failed during export.")
        raise

    # 5. Serialize Artifacts
    logger.info("Export successful. Serializing ONNX runtimes to disk...")
    try:
        onnx_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
    except Exception as e:
        logger.error("Failed to write ONNX models to target directory.")
        raise

    # 6. Single-Pass Inference Verification
    logger.info("Validating standalone ONNX Runtime execution...")
    try:
        inputs = tokenizer(test_text, return_tensors="pt")
        
        start_time = time.perf_counter()
        generated_tokens = onnx_model.generate(
            **inputs, 
            max_new_tokens=128, 
            num_beams=4, 
            early_stopping=True
        )
        inference_time = (time.perf_counter() - start_time) * 1000
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        print("\n--- POC Results ---")
        print(f"Source:      {test_text}")
        print(f"Translation: {translation}")
        print(f"Latency:     {inference_time:.2f} ms")
        print("-------------------")
        logger.info("STATUS: SUCCESS. Marian model graph is fully deployable via ONNX Runtime.")
        
    except Exception as e:
        logger.error("ONNX Runtime engine failed to execute inference path.")
        raise

if __name__ == "__main__":
    verify_onnx_export()