"""
BridgeDEUX Core Framework
Evaluation Data Structures
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(
    frozen=True,
    slots=True,
)
class ModelEvaluation:
    """
    Immutable data transfer object capturing the evaluation
    metadata and dynamic metrics for a specific model benchmark.
    """

    model_name: str
    model_version: str
    total_samples: int
    failed_samples: int
    mean_latency_ms: float

    # Dynamic metric container satisfying the Open/Closed Principle
    metrics: dict[str, float] = field(default_factory=dict)