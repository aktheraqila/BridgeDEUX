import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import onnx
import onnxruntime as ort

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

logger = BridgeLogger.get_logger(__name__)


def hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_nodes(path: Path) -> int:
    try:
        model = onnx.load(str(path))
        return len(model.graph.node)
    except Exception:
        return -1


def optimize_graph(input_path: Path, output_path: Path, opt_level: ort.GraphOptimizationLevel, model_name: str) -> dict:
    orig_size = input_path.stat().st_size
    orig_hash = hash_file(input_path)
    orig_nodes = count_nodes(input_path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = opt_level
    sess_options.optimized_model_filepath = str(output_path)

    start_time = time.perf_counter()
    try:
        _ = ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.error(f"[Model: {model_name} | File: {input_path.name}] ORT optimization failed")
        raise RuntimeError(f"Optimization failure: {e}")

    opt_time = time.perf_counter() - start_time

    try:
        onnx.checker.check_model(str(output_path))
    except Exception as e:
        logger.error(f"[Model: {model_name} | File: {output_path.name}] ONNX checker validation failed")
        raise RuntimeError(f"Validation failure: {e}")

    opt_size = output_path.stat().st_size
    opt_hash = hash_file(output_path)
    opt_nodes = count_nodes(output_path)

    return {
        "file": input_path.name,
        "input_sha256": orig_hash,
        "output_sha256": opt_hash,
        "original_size_bytes": orig_size,
        "optimized_size_bytes": opt_size,
        "original_nodes": orig_nodes,
        "optimized_nodes": opt_nodes,
        "optimization_time_s": round(opt_time, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX computational graphs.")
    parser.add_argument("--model", type=str, default="opus_mt_de_en", help="Base model directory name")
    parser.add_argument("--suffix", type=str, default="opt_extended", help="Suffix for target directory")
    parser.add_argument("--level", choices=["extended", "all"], default="extended", help="ORT optimization level")
    args = parser.parse_args()

    ProjectConfig.initialize()
    
    src_dir = Path(ProjectConfig.MODEL_DIR) / "onnx" / args.model
    target_dir = Path(ProjectConfig.MODEL_DIR) / "onnx" / f"{args.model}_{args.suffix}"
    tmp_dir = target_dir.with_name(target_dir.name + "_tmp")
    backup_dir = target_dir.with_name(target_dir.name + "_backup")
    
    if not src_dir.exists():
        logger.error(f"[Model: {args.model}] Source directory not found: {src_dir}")
        return

    # Clean previous temp directories if interrupted
    for path in (tmp_dir, backup_dir):
        if path.exists():
            shutil.rmtree(path)
            
    tmp_dir.mkdir(parents=True, exist_ok=True)

    opt_level = (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED 
                 if args.level == "extended" 
                 else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    logger.info(f"Applying {opt_level.name} to {args.model} -> {target_dir.name}")
    
    metadata = {
        "experiment": "onnx_graph_optimization",
        "source_model": args.model,
        "target_model": target_dir.name,
        "optimization_level": opt_level.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "graphs": [],
    }

    try:
        for src_file in sorted(src_dir.iterdir()):
            tmp_file = tmp_dir / src_file.name
            if src_file.suffix == ".onnx":
                logger.info(f"Optimizing graph: {src_file.name}")
                stats = optimize_graph(src_file, tmp_file, opt_level, args.model)
                metadata["graphs"].append(stats)
            else:
                logger.info(f"Copying auxiliary artifact: {src_file.name}")
                shutil.copy2(src_file, tmp_file)
                
        # Truly Atomic Commit Pattern (Backup -> Move -> Delete Backup)
        if target_dir.exists():
            target_dir.rename(backup_dir)
            
        tmp_dir.rename(target_dir)
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
            
    except Exception:
        logger.error(f"[Model: {args.model}] Pipeline failed. Rolling back changes...")
        logger.error(traceback.format_exc())
        
        # Rollback logic
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.rename(target_dir)
        return

    metadata_path = Path(ProjectConfig.REPORT_DIR) / f"opt_execution_{target_dir.name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Optimization complete. Deployable artifacts atomically committed to: {target_dir}")
    logger.info(f"Execution metadata recorded at: {metadata_path}")


if __name__ == "__main__":
    main()