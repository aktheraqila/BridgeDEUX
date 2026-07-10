"""
BridgeDEUX Core Framework
Evaluation Runner
"""

from __future__ import annotations

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

from evaluation.evaluator import EvaluationEngine


def main() -> None:

    ProjectConfig.initialize()

    logger = BridgeLogger.get_logger(
        "EvaluationRunner"
    )

    engine = EvaluationEngine()

    print("\n========================================================")
    print(" BridgeDEUX Evaluation Framework")
    print("========================================================\n")

    results = engine.evaluate_all()

    if not results:
        
        logger.error(
            "No valid evaluations produced."
        )
        
        raise SystemExit(1)

    for result in results:

        print(
            f"Model   : {result.model_name} ({result.model_version})\n"
            f"Samples : {result.total_samples} (Failed: {result.failed_samples})\n"
            f"Latency : {result.mean_latency_ms:.2f} ms (Mean)"
        )

        if result.metrics:
            for metric_name, score in result.metrics.items():
                print(f"{metric_name:<7} : {score:.2f}")
        else:
            print("Metrics : [Pending Execution]")

        print("-" * 56)

    print("\nDiscovery & Ingestion completed successfully.")


if __name__ == "__main__":
    main()