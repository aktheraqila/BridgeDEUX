"""
BridgeDEUX Core Framework
Offline Translation Benchmark Runner
Milestone: 11.5 (FROZEN)

This orchestrator enforces deterministic dataset ordering, strict experiment
provenance, isolated circuit-breaking, and resilient artifact-first recovery.
"""

from __future__ import annotations

import threading
import psutil
import os
import argparse
import logging
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Set

import pandas as pd

from pathlib import Path
from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger

from benchmarks.checkpoint_manager import CheckpointManager
from benchmarks.exceptions import (
    BenchmarkError,
    CheckpointError,
    CircuitBreakerError
)
from models.translators.exceptions import TranslationError
from models.translators.base_translator import BaseTranslator
from models.translators.marian import MarianTranslator
from models.translators.m2m100 import M2M100Translator


# --- Runner Configuration ---
LOG_INTERVAL = 10
MAX_CONSECUTIVE_FAILURES = 5
REQUIRED_DATASET_COLUMNS = {"sample_id", "source_text", "reference_translation"}
RANDOM_SEED = 42


def enforce_determinism(seed: int) -> None:
    """Configures deterministic behavior where supported by the underlying ML backends."""
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_revision() -> str:
    """Safely attempts to retrieve the current Git commit hash for provenance."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            stderr=subprocess.DEVNULL
        )
        return rev.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_experiment_provenance(logger: logging.Logger) -> None:
    """Records the execution environment for scientific reproducibility."""
    logger.info("--- Experiment Provenance ---")
    logger.info("Git Revision: %s", get_git_revision())
    logger.info("OS: %s %s (%s)", platform.system(), platform.release(), platform.version())
    logger.info("CPU Family: %s", platform.processor() or "unknown")
    logger.info("Python Version: %s", platform.python_version())
    logger.info("Pandas Version: %s", pd.__version__)
    logger.info("Random Seed: %d", RANDOM_SEED)
    
    try:
        import torch
        logger.info("PyTorch Version: %s", torch.__version__)
    except ImportError:
        pass
        
    logger.info("-----------------------------")


def get_translator(model_name: str) -> BaseTranslator:
    """Factory to instantiate the requested translator architecture."""
    name = model_name.lower()
    try:
        if name == "marian":
            return MarianTranslator()
        elif name == "m2m100":
            return M2M100Translator()
        else:
            raise BenchmarkError(f"Unknown model architecture requested: {model_name}")
    # Wrap ONLY expected infrastructure failures (e.g., missing weights, CUDA out of memory).
    # Programming bugs (TypeError, AttributeError) will deliberately bypass this and crash fast.
    except (OSError, RuntimeError, FileNotFoundError) as e:
        raise BenchmarkError(f"Failed to load {model_name} backend: {e}") from e


class HardwareMonitor:
    """Non-blocking background thread to track CPU and RAM without skewing latency."""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.keep_running = True
        self.cpu_measurements = []
        self.ram_measurements = [] # in MB
        self.peak_ram = 0.0
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        # Initial baseline read (ignored)
        self.process.cpu_percent()
        while self.keep_running:
            try:
                # Poll every 1 second
                cpu = self.process.cpu_percent(interval=1.0)
                ram = self.process.memory_info().rss / (1024 * 1024)
                
                self.cpu_measurements.append(cpu)
                self.ram_measurements.append(ram)
                if ram > self.peak_ram:
                    self.peak_ram = ram
            except Exception:
                pass

    def start(self):
        self.thread.start()

    def stop(self) -> dict:
        self.keep_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        avg_cpu = sum(self.cpu_measurements) / len(self.cpu_measurements) if self.cpu_measurements else 0.0
        peak_cpu = max(self.cpu_measurements) if self.cpu_measurements else 0.0
        avg_ram = sum(self.ram_measurements) / len(self.ram_measurements) if self.ram_measurements else 0.0
        
        return {
            "avg_cpu": avg_cpu,
            "peak_cpu": peak_cpu,
            "avg_ram": avg_ram,
            "peak_ram": self.peak_ram
        }

def run_benchmark(
    model_name: str,
    dataset_name: str,
    limit: int | None = None,
) -> None:
    """
    Executes the offline benchmark pipeline with strict crash recovery, 
    deterministic state resolution, and telemetry tracking.
    """
    logger = BridgeLogger.get_logger("BenchmarkRunner")
    
    start_wall_time = time.time()
    start_perf_time = time.perf_counter() 
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    enforce_determinism(RANDOM_SEED)
    
    logger.info("Run started at: %s", start_timestamp)
    log_experiment_provenance(logger)

    # 1. Initialization (Surgically timing the model load)
    logger.info("Initializing translation engine: %s...", model_name)
    load_start_time = time.perf_counter()

    translator = get_translator(model_name)
    translator.load()

    load_time_seconds = time.perf_counter() - load_start_time
    logger.info("Model load time: %.2f seconds", load_time_seconds)

    cached_model_name = translator.model_name
    cached_model_version = translator.model_version

    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)

    dataset_stem = Path(dataset_name).stem
    experiment_id = f"{cached_model_name}_{dataset_stem}"

    manager = CheckpointManager(
        model_identifier=experiment_id,
        output_dir=results_folder,
        checkpoint_interval=25,
    )

    # Launch the non-blocking hardware monitor thread here
    hw_monitor = HardwareMonitor()
    hw_monitor.start()

    # Execution Block
    try:
        # 2. Dataset Loading & Strict Schema Validation
        dataset_path = ProjectConfig.BENCHMARK_DIR / dataset_name
        if not dataset_path.exists():
            raise BenchmarkError(f"Dataset not found at {dataset_path}")

        logger.info("Loading dataset from %s", dataset_path.name)
        df = pd.read_parquet(dataset_path)

        df = df.rename(columns={"target_text": "reference_translation"})

        if "sample_id" not in df.columns and "id" in df.columns:
            df["sample_id"] = df["id"]

        if "reference_translation" not in df.columns and "target_text" in df.columns:
            df["reference_translation"] = df["target_text"]

        if not REQUIRED_DATASET_COLUMNS.issubset(df.columns):
            missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
            raise BenchmarkError(f"Dataset schema invalid. Missing columns: {missing}")

        # 3. Strict Deterministic Sorting
        df = (
            df.sort_values(by="sample_id", kind="stable")
            .reset_index(drop=True)
        )

        # 4. State Recovery & O(1) Filtering
        completed_ids: Set[str] = manager.load_completed_samples()
        df_remaining = df[~df["sample_id"].astype(str).isin(completed_ids)]

        if limit:
            df_remaining = df_remaining.head(limit)

        total_remaining = len(df_remaining)
        if total_remaining == 0:
            logger.info("All requested samples have already been processed.")
            # Safely spin down metrics threads before returning
            hw_monitor.stop()
            manager.flush()
            manager.finalize()
            return

        logger.info(
            "Starting inference on %d remaining samples (Limit: %s).", 
            total_remaining, 
            limit if limit else "None"
        )

        # 5. Telemetry & State Tracking
        consecutive_failures = 0
        session_processed_samples = 0
        session_generated_tokens = 0
        
        # 6. Inference Execution Loop
        inference_start_perf = time.perf_counter()
        
        for idx, row in enumerate(df_remaining.itertuples(index=False), start=1):
            sample_id = str(row.sample_id)
            source_text = str(row.source_text)
            reference_text = str(row.reference_translation)

            try:
                result = translator.translate(source_text)
                
                record = {
                    "sample_id": sample_id,
                    "model_name": cached_model_name,
                    "model_version": cached_model_version,
                    "source_text": source_text,
                    "translation": result.translation,
                    "reference_translation": reference_text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "tokenization_time_ms": result.tokenization_time_ms,
                    "generation_time_ms": result.generation_time_ms,
                    "decoding_time_ms": result.decoding_time_ms,
                    "total_time_ms": result.total_time_ms,
                }

                manager.save(record)
                
                consecutive_failures = 0  
                session_processed_samples += 1
                session_generated_tokens += result.output_tokens

                if idx % LOG_INTERVAL == 0 or idx == total_remaining:
                    elapsed = time.perf_counter() - inference_start_perf
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_remaining - idx) / rate if rate > 0 else 0
                        
                    pct = (idx / total_remaining) * 100
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    
                    logger.info(
                        "Processed %d/%d (%.2f%%) | Elapsed: %s | ETA: %s",
                        idx, total_remaining, pct, elapsed_str, eta_str
                    )

            except TranslationError as e:
                logger.error("Inference failed for sample_id %s: %s", sample_id, e)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise CircuitBreakerError(
                        f"Circuit breaker tripped: {consecutive_failures} consecutive translation failures."
                    ) from e
                continue

            except (TypeError, AttributeError, NameError, KeyError) as e:
                logger.critical("Structural error detected for sample_id %s. Crashing immediately.", sample_id)
                raise e

        logger.info("Dataset exhausted. Flushing final buffer and finalizing checkpoint...")
        manager.flush()
        manager.finalize()

    finally:
        # Stop monitor thread immediately to collect clean hardware metrics
        hw_stats = hw_monitor.stop()

        try:
            translator.close()
        except Exception as e:
            logger.warning("Non-fatal error during translator teardown: %s", e)
            
        try:
            manager.flush()
        except Exception as e:
            logger.error("Error during final WAL flush in teardown: %s", e)
            
    # 8. Final Thesis-Grade Telemetry (Updated for complete summary view)
    total_perf_time = time.perf_counter() - start_perf_time
    total_wall_time = time.time() - start_wall_time
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    samples_per_sec = (session_processed_samples / total_perf_time) if total_perf_time > 0 else 0
    tokens_per_sec = (session_generated_tokens / total_perf_time) if total_perf_time > 0 else 0
    
    logger.info("--- Benchmark Session Summary ---")
    logger.info("Run started at:       %s", start_timestamp)
    logger.info("Run finished at:      %s", end_timestamp)
    logger.info("Total wall time:      %s", str(timedelta(seconds=int(total_wall_time))))
    logger.info("Model load time:      %.2f seconds", load_time_seconds)
    logger.info("Sample throughput:    %.2f samples/sec", samples_per_sec)
    logger.info("Token throughput:     %.2f output tokens/sec", tokens_per_sec)
    logger.info("Peak RAM Usage:       %.2f MB", hw_stats["peak_ram"])
    logger.info("Average RAM Usage:    %.2f MB", hw_stats["avg_ram"])
    logger.info("Average CPU Usage:    %.2f %%", hw_stats["avg_cpu"])
    logger.info("Peak CPU Usage:       %.2f %%", hw_stats["peak_cpu"])
    logger.info("-----------------------------")


def main() -> None:
    """CLI Entrypoint - Manages process lifecycle and exception rendering."""
    ProjectConfig.initialize()
    logger = BridgeLogger.get_logger("Main")
    
    parser = argparse.ArgumentParser(description="Run BridgeDEUX offline translation benchmark.")
    parser.add_argument("--model", type=str, required=True, choices=["marian", "m2m100"])
    parser.add_argument("--dataset", type=str, default="benchmark_subset_100.parquet")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process.")
    
    args = parser.parse_args()
    
    try:
        run_benchmark(
            model_name=args.model, 
            dataset_name=args.dataset, 
            limit=args.limit
        )
    except KeyboardInterrupt:
        logger.warning("\nRun interrupted by user (Ctrl+C). Checkpoint manager has secured progress. Exiting gracefully.")
        sys.exit(0)
    # Exceptions ordered from most specific to least specific
    except CircuitBreakerError as e:
        logger.critical("Benchmark Aborted by Safety Policy: %s", str(e))
        sys.exit(1)
    except CheckpointError as e:
        logger.critical("Infrastructure I/O Aborted: %s", str(e))
        sys.exit(1)
    except BenchmarkError as e:
        logger.critical("Benchmark Initialization/Dataset Error: %s", str(e))
        sys.exit(1)
    except Exception as e:
        logger.critical("Benchmark Aborted due to unhandled fatal error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()