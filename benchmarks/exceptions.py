"""
BridgeDEUX Core Framework
Benchmark Subsystem Exceptions
"""

class BenchmarkError(Exception):
    """Base exception for the benchmark subsystem."""
    pass


class CheckpointError(BenchmarkError):
    """Failures specific to the CheckpointManager."""
    pass


class CircuitBreakerError(BenchmarkError):
    """Circuit breaker triggered due to excessive consecutive failures."""
    pass