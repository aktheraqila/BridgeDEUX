import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

logger = BridgeLogger.get_logger(__name__)


def hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def analyze_model(path: Path, model_name: str) -> dict:
    model = onnx.load(str(path))
    
    # Validate structural integrity of original graph
    try:
        onnx.checker.check_model(model)
        is_valid = True
    except Exception as e:
        logger.error(f"[Model: {model_name} | File: {path.name}] Graph check failed: {e}")
        is_valid = False
        raise

    return {
        "file": path.name,
        "sha256": hash_file(path),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "is_valid": is_valid,
        "ir_version": getattr(model, "ir_version", None),
        "producer_name": getattr(model, "producer_name", None),
        "producer_version": getattr(model, "producer_version", None),
        "graph_name": getattr(model.graph, "name", None),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "input_count": len(model.graph.input),
        "output_count": len(model.graph.output),
        "opset": model.opset_import[0].version if model.opset_import else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ONNX graph topology.")
    parser.add_argument("--model", type=str, default="opus_mt_de_en", help="Target model directory")
    args = parser.parse_args()

    ProjectConfig.initialize()
    model_dir = Path(ProjectConfig.MODEL_DIR) / "onnx" / args.model

    if not model_dir.exists():
        logger.error(f"[Model: {args.model}] Directory not found: {model_dir}")
        return

    logger.info(f"Starting baseline analysis for: {args.model}")
    
    results = {
        "model": args.model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "onnx_version": onnx.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "graphs": [],
    }

    for model_path in sorted(model_dir.glob("*.onnx")):
        logger.info(f"Analyzing {model_path.name}...")
        try:
            results["graphs"].append(analyze_model(model_path, args.model))
        except Exception as e:
            logger.error(f"[Model: {args.model} | File: {model_path.name}] Analysis aborted: {e}", exc_info=True)
            raise

    output_path = Path(ProjectConfig.REPORT_DIR) / f"onnx_baseline_{args.model}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)

    logger.info(f"Analysis complete. Artifact saved to: {output_path}")


if __name__ == "__main__":
    main()