"""
BridgeDEUX Core Framework
Publication-Grade Native ONNX Runtime Dynamic INT8 Quantization Pipeline (v8.0 Final)
"""

import sys
import shutil
import csv
import json
import time
import hashlib
import datetime
import platform
import socket
import subprocess
from pathlib import Path

# Automatically append project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

from bridge.logger import BridgeLogger

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
SOURCE_MODEL_DIR = Path("models/onnx/opus_mt_de_en_opt_extended")
TARGET_MODEL_DIR = Path("models/onnx/opus_mt_de_en_opt_extended_int8")

GRAPH_MODE = "standard"
GRAPH_STRATEGIES = {
    "merged": ["encoder_model.onnx", "decoder_model_merged.onnx"],
    "standard": ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]
}

# Preprocessing Constants
PREPROCESSING_ALWAYS = "always"
PREPROCESSING_NEVER = "never"
PREPROCESSING_MODE = PREPROCESSING_NEVER 

EXECUTION_PROVIDER = "CPUExecutionProvider"

QUANTIZATION_PARAMS = {
    "weight_type": QuantType.QUInt8,
    "per_channel": False,
    "reduce_range": False,
    "extra_options": {"DefaultTensorType": onnx.TensorProto.FLOAT}
}

# ==============================================================================
# PROVENANCE & UTILITY FUNCTIONS
# ==============================================================================
# ... (rest of the script remains unchanged) ...

# ==============================================================================
# PROVENANCE & UTILITY FUNCTIONS
# ==============================================================================
def compute_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_NON_GIT_BUILD"

def resolve_target_graphs(source_dir: Path, mode: str) -> list[str]:
    if mode in GRAPH_STRATEGIES:
        candidates = GRAPH_STRATEGIES[mode]
        discovered = [g for g in candidates if (source_dir / g).exists()]
        if not discovered:
            raise FileNotFoundError(f"GRAPH_MODE '{mode}' selected, but no matching files found in {source_dir}")
        return discovered
    elif mode == "auto":
        discovered = [f.name for f in source_dir.glob("*.onnx")]
        if not discovered:
            raise FileNotFoundError(f"No ONNX models found in {source_dir}")
        return discovered
    else:
        raise ValueError(f"Invalid GRAPH_MODE '{mode}'. Expected 'merged', 'standard', or 'auto'.")

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main() -> None:
    logger = BridgeLogger.get_logger("NativeQuantizer")
    pipeline_start_time = time.perf_counter()
    
    if not SOURCE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_MODEL_DIR}")

    logger.info("==================================================================")
    logger.info(" Starting Robust INT8 Quantization Pipeline (v8.0 Final)")
    logger.info(" Source Directory   : %s", SOURCE_MODEL_DIR)
    logger.info(" Target Directory   : %s", TARGET_MODEL_DIR)
    logger.info(" Selection Mode     : %s", GRAPH_MODE)
    logger.info(" Preprocessing Mode : %s", PREPROCESSING_MODE)
    logger.info(" Execution Provider : %s", EXECUTION_PROVIDER)
    logger.info("==================================================================")

    if TARGET_MODEL_DIR.exists():
        logger.info("Cleaning existing target directory for reproducibility...")
        shutil.rmtree(TARGET_MODEL_DIR)
    TARGET_MODEL_DIR.mkdir(parents=True)

    target_graphs = resolve_target_graphs(SOURCE_MODEL_DIR, GRAPH_MODE)
    logger.info("Target graphs selected for quantization: %s", target_graphs)

    source_artifacts = []
    for g in target_graphs:
        src_path = SOURCE_MODEL_DIR / g
        source_artifacts.append({
            "file": g,
            "sha256": compute_sha256(src_path),
            "size_mb": round(src_path.stat().st_size / (1024 * 1024), 2)
        })

    logger.info("Mirroring non-ONNX assets (tokenizer, vocabulary, configs)...")
    copied_assets = []
    for file_path in SOURCE_MODEL_DIR.iterdir():
        if file_path.is_file() and file_path.suffix != ".onnx":
            dest_path = TARGET_MODEL_DIR / file_path.name
            shutil.copy2(file_path, dest_path)
            copied_assets.append(file_path.name)
            
    logger.info("Successfully copied %d non-ONNX assets.", len(copied_assets))

    report_data = []
    graph_decisions = []
    any_failures = False

    for graph_name in target_graphs:
        input_path = SOURCE_MODEL_DIR / graph_name
        output_path = TARGET_MODEL_DIR / graph_name
        preprocessed_path = TARGET_MODEL_DIR / f"{graph_name}.preprocessed"
        
        logger.info("------------------------------------------------------------------")
        logger.info("Processing Graph: %s", graph_name)
        
        src_sha256 = compute_sha256(input_path)
        original_bytes = input_path.stat().st_size
        original_mb = original_bytes / (1024 * 1024)
        
        graph_start_time = time.perf_counter()
        
        model_for_quantization = input_path
        shape_inference_status = "Skipped"
        quantized_model_type = "Original"

        # =======================================================
        # STAGE 1: Configurable Shape Inference Pre-processing
        # =======================================================
        if PREPROCESSING_MODE == PREPROCESSING_ALWAYS:
            try:
                logger.info("  [STAGE 1] Running symbolic shape inference...")
                quant_pre_process(
                    input_model_path=str(input_path),
                    output_model_path=str(preprocessed_path),
                    skip_optimization=False
                )
                model_for_quantization = preprocessed_path
                shape_inference_status = "Success"
                quantized_model_type = "Preprocessed"
                logger.info("  [STAGE 1] Shape inference completed successfully.")
            except Exception as e:
                shape_inference_status = "Failed (Fallback Used)"
                quantized_model_type = "Original"
                error_type = type(e).__name__
                logger.warning("  [STAGE 1] Shape inference failed (%s): %s", error_type, str(e))
                logger.warning("  [STAGE 1] Falling back to original ONNX model.")
        elif PREPROCESSING_MODE == PREPROCESSING_NEVER:
            logger.info("  [STAGE 1] Symbolic shape inference disabled by configuration.")
            shape_inference_status = "Disabled"
            quantized_model_type = "Original"
        else:
            raise ValueError(
                f"Invalid PREPROCESSING_MODE '{PREPROCESSING_MODE}'. "
                f"Expected '{PREPROCESSING_ALWAYS}' or '{PREPROCESSING_NEVER}'."
            )

        # =======================================================
        # STAGE 2: Mandatory Dynamic Quantization
        # =======================================================
        try:
            logger.info("  [STAGE 2] Executing dynamic quantization...")
            quantize_dynamic(
                model_input=str(model_for_quantization),
                model_output=str(output_path),
                **QUANTIZATION_PARAMS
            )
            
            graph_elapsed = time.perf_counter() - graph_start_time
            int8_bytes = output_path.stat().st_size
            int8_mb = int8_bytes / (1024 * 1024)
            int8_sha256 = compute_sha256(output_path)
            reduction_pct = (1 - (int8_bytes / original_bytes)) * 100
            
            logger.info("  [SUCCESS] Quantized %s", graph_name)
            logger.info("  Original Size : %.2f MB (SHA256: %s...)", original_mb, src_sha256[:12])
            logger.info("  INT8 Size     : %.2f MB (SHA256: %s...)", int8_mb, int8_sha256[:12])
            logger.info("  Reduction     : %.2f%%", reduction_pct)
            
            report_data.append({
                "Graph": graph_name,
                "Status": "Success",
                "Shape_Inference": shape_inference_status,
                "Quantized_Model": quantized_model_type,
                "Original_SHA256": src_sha256,
                "INT8_SHA256": int8_sha256,
                "Original_MB": round(original_mb, 2),
                "INT8_MB": round(int8_mb, 2),
                "Reduction_Percent": round(reduction_pct, 2),
                "Quantization_Time_s": round(graph_elapsed, 2),
                "Error": ""
            })

            graph_decisions.append({
                "graph": graph_name,
                "status": "Success",
                "preprocessing_mode": PREPROCESSING_MODE,
                "shape_inference": shape_inference_status,
                "quantized_from": quantized_model_type
            })
            
        except Exception as e:
            any_failures = True
            graph_elapsed = time.perf_counter() - graph_start_time
            error_type = type(e).__name__
            logger.exception("  [FAILED] Quantization failed for %s", graph_name)
            
            report_data.append({
                "Graph": graph_name,
                "Status": "Failed",
                "Shape_Inference": shape_inference_status,
                "Quantized_Model": quantized_model_type if quantized_model_type == "Original" else "None",
                "Original_SHA256": src_sha256,
                "INT8_SHA256": "N/A",
                "Original_MB": round(original_mb, 2),
                "INT8_MB": 0.0,
                "Reduction_Percent": 0.0,
                "Quantization_Time_s": round(graph_elapsed, 2),
                "Error": f"{error_type}: {str(e)}"
            })

            graph_decisions.append({
                "graph": graph_name,
                "status": "Failed",
                "preprocessing_mode": PREPROCESSING_MODE,
                "shape_inference": shape_inference_status,
                "quantized_from": "None"
            })
        finally:
            if preprocessed_path.exists():
                preprocessed_path.unlink()

    total_pipeline_time = time.perf_counter() - pipeline_start_time
    overall_status = "partial_failure" if any_failures else "success"

    # Export Quantization Report CSV
    report_file = TARGET_MODEL_DIR / "quantization_report.csv"
    fieldnames = [
        "Graph", "Status", "Shape_Inference", "Quantized_Model", "Original_SHA256", "INT8_SHA256", 
        "Original_MB", "INT8_MB", "Reduction_Percent", 
        "Quantization_Time_s", "Error"
    ]
    with open(report_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_data)
    logger.info("Exported CSV report to %s", report_file.name)

    # Export Model Metadata JSON with explicit graph decisions
    metadata_file = TARGET_MODEL_DIR / "quantization_metadata.json"
    serializable_params = {
        k: (v.name if isinstance(v, QuantType) else v)
        for k, v in QUANTIZATION_PARAMS.items()
    }
    metadata_payload = {
        "graph_mode": GRAPH_MODE,
        "preprocessing_mode": PREPROCESSING_MODE,
        "graph_decisions": graph_decisions,
        "source_artifacts": source_artifacts,
        "copied_assets": copied_assets,
        "quantization_params": serializable_params,
        "framework": "onnxruntime.quantization",
        "onnxruntime_version": ort.__version__
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4)
    logger.info("Exported metadata JSON to %s", metadata_file.name)

    # Export Structured Experiment Provenance Info JSON
    info_file = TARGET_MODEL_DIR / "experiment_info.json"
    experiment_info = {
        "experiment": {
            "git_revision": get_git_revision(),
            "timestamp": datetime.datetime.now().isoformat(),
            "status": overall_status
        },
        "environment": {
            "python_version": platform.python_version(),
            "onnxruntime_version": ort.__version__,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "execution_provider": EXECUTION_PROVIDER
        },
        "execution": {
            "source_dir": str(SOURCE_MODEL_DIR),
            "target_dir": str(TARGET_MODEL_DIR),
            "total_pipeline_time_s": round(total_pipeline_time, 2)
        }
    }
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(experiment_info, f, indent=4)
    logger.info("Exported experiment info JSON to %s", info_file.name)

    logger.info("==================================================================")
    logger.info(" Pipeline Finished with Status [%s] in %.2f seconds", overall_status.upper(), total_pipeline_time)
    logger.info(" Final Artifact Location: %s", TARGET_MODEL_DIR)
    logger.info("==================================================================")

if __name__ == "__main__":
    main()