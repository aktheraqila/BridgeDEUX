
"""
BridgeDEUX
ONNX Topology Comparator

Purpose
-------
Automatically pairs the most recent baseline and optimized topology reports 
and computes the structural deltas (node counts, file sizes).
"""

import argparse
import json
from pathlib import Path

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

logger = BridgeLogger.get_logger(__name__)


def load_latest_report(report_dir: Path, prefix: str) -> dict:
    # Matches files like onnx_baseline_opus_mt_de_en_2026...json
    files = sorted(report_dir.glob(f"{prefix}*.json"))
    if not files:
        raise FileNotFoundError(f"No reports found matching {prefix}*.json in {report_dir}")
    
    latest_file = files[-1]
    logger.info(f"Loaded report: {latest_file.name}")
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare ONNX topology reports.")
    parser.add_argument("--model", type=str, default="opus_mt_de_en", help="Base model name")
    parser.add_argument("--variant", type=str, default="opt_extended", help="Optimized variant suffix")
    args = parser.parse_args()

    ProjectConfig.initialize()
    report_dir = Path(ProjectConfig.REPORT_DIR)

    try:
        baseline_data = load_latest_report(report_dir, f"onnx_baseline_{args.model}_2")
        optimized_data = load_latest_report(report_dir, f"onnx_baseline_{args.model}_{args.variant}_2")
    except FileNotFoundError as e:
        logger.error(e)
        return

    print("\n" + "="*85)
    print(f"TOPOLOGY COMPARISON: {args.model} (Baseline vs {args.variant})")
    print("="*85)
    print(f"{'Graph':<30} | {'Nodes (Base)':<12} | {'Nodes (Opt)':<12} | {'Delta':<8} | {'Size Base':<9} | {'Size Opt':<9}")
    print("-" * 85)

    base_graphs = {g["file"]: g for g in baseline_data["graphs"]}
    opt_graphs = {g["file"]: g for g in optimized_data["graphs"]}

    for file_name in sorted(base_graphs.keys()):
        base = base_graphs[file_name]
        opt = opt_graphs.get(file_name)
        
        if not opt:
            print(f"{file_name:<30} | Missing in optimized report")
            continue
            
        node_delta = opt["node_count"] - base["node_count"]
        
        print(f"{file_name:<30} | "
              f"{base['node_count']:<12} | "
              f"{opt['node_count']:<12} | "
              f"{node_delta:+8d} | "
              f"{base['size_mb']:>5.2f} MB | "
              f"{opt['size_mb']:>5.2f} MB")
        
    print("="*85)


if __name__ == "__main__":
    main()