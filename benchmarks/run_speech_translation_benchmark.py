# benchmarks/run_speech_translation_benchmark.py
"""
BridgeDEUX Core Framework
Offline Cascaded Speech Translation Benchmark Runner
Milestone: 12.0 (PRODUCTION)

Surgically adapts the verified offline ASR pipeline execution loop to route
transcribed acoustic hypotheses directly into the machine translation subsystem.
Inherits deterministic constraints, WAL checkpoint tracking, and hardware telemetry.
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
from pathlib import Path

import pandas as pd
from jiwer import wer, cer

from bridge.config import ProjectConfig
from bridge.logger import BridgeLogger
from bridge.audio import AudioProcessor

from models.asr.base_asr import BaseASR
from models.asr.whisper_cpp import WhisperCppASR
from models.asr.vosk import VoskASR

from models.translators.base_translator import BaseTranslator
from models.translators.marian import MarianTranslator
from models.translators.m2m100 import M2M100Translator
from models.translators.exceptions import TranslationError

from datasets.providers.covost_provider import CoVoSTProvider
from benchmarks.checkpoint_manager import CheckpointManager
from benchmarks.exceptions import (
    BenchmarkError,
    CheckpointError,
    CircuitBreakerError
)

# --- Runner Configuration ---
LOG_INTERVAL = 10
MAX_CONSECUTIVE_FAILURES = 5
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
    logger.info("-----------------------------")


def get_asr_model(model_name: str) -> BaseASR:
    """Factory to instantiate the requested ASR architecture."""
    name = model_name.lower()
    try:
        if name == "whisper":
            return WhisperCppASR(model_size="base", n_threads=4)
        elif name == "vosk":
            model_path = ProjectConfig.MODEL_DIR / "vosk" / "vosk-model-small-de-0.15"
            return VoskASR(model_path)
        else:
            raise BenchmarkError(f"Unknown ASR architecture requested: {model_name}")
    except (OSError, RuntimeError, FileNotFoundError) as e:
        raise BenchmarkError(f"Failed to load {model_name} backend: {e}") from e


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
    except (OSError, RuntimeError, FileNotFoundError) as e:
        raise BenchmarkError(f"Failed to load {model_name} backend: {e}") from e


class HardwareMonitor:
    """Non-blocking background thread to track CPU and RAM without skewing latency."""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.keep_running = True
        self.cpu_measurements = []
        self.ram_measurements = []
        self.peak_ram = 0.0
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        self.process.cpu_percent()
        while self.keep_running:
            try:
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


def run_speech_translation_benchmark(
    asr_model_name: str,
    mt_model_name: str,
    split_name: str,
    limit: int | None = None,
) -> None:
    """Executes the complete offline speech translation pipeline workload."""
    logger = BridgeLogger.get_logger("SpeechTranslationBenchmarkRunner")
    
    start_wall_time = time.time()
    start_perf_time = time.perf_counter() 
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    enforce_determinism(RANDOM_SEED)
    
    logger.info("Run started at: %s", start_timestamp)
    log_experiment_provenance(logger)

    # 1. Initialization (ASR & MT Backends)
    logger.info("Initializing ASR engine: %s...", asr_model_name)
    load_start_time = time.perf_counter()
    asr_model = get_asr_model(asr_model_name)
    asr_model.load()

    logger.info("Initializing MT engine: %s...", mt_model_name)
    translator = get_translator(mt_model_name)
    translator.load()

    load_time_seconds = time.perf_counter() - load_start_time
    logger.info("Total models load time: %.2f seconds", load_time_seconds)

    cached_asr_name = asr_model.model_name
    cached_mt_name = translator.model_name
    cached_mt_version = getattr(translator, "model_version", "unknown")

    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)

    experiment_id = f"cascaded_{cached_asr_name}_{cached_mt_name}_{split_name}"

    manager = CheckpointManager(
        model_identifier=experiment_id,
        output_dir=results_folder,
        checkpoint_interval=25,
    )

    hw_monitor = HardwareMonitor()
    hw_monitor.start()

    try:
        # 2. Dataset Loading via Provider Abstraction
        logger.info("Initializing CoVoSTProvider for split: %s", split_name)
        provider = CoVoSTProvider(split=split_name, include_audio=True)

        # 3. State Recovery & O(1) Filtering
        completed_ids: Set[str] = manager.load_completed_samples()

        logger.info("Starting inference phase (Limit: %s).", limit if limit else "None")

        consecutive_failures = 0
        session_processed_samples = 0
        inference_start_perf = time.perf_counter()
        
        # 4. Inference Execution Loop
        for sample in provider:
            sample_id = str(sample.id)
            
            if sample_id in completed_ids:
                continue

            if limit and session_processed_samples >= limit:
                break

            source_text = str(sample.source_text)
            reference_translation = str(sample.target_text) # Crucial Fix: target_text is the true English string
            audio_bytes = sample.audio.get("bytes") if sample.audio else None
            
            if not audio_bytes:
                logger.error("Skipping sample_id %s: Missing raw audio bytes.", sample_id)
                continue

            try:
                # --- PHASE 1: ASR ---
                start_dsp = time.perf_counter()
                raw_array, sample_rate = AudioProcessor.decode_mp3_to_pcm(audio_bytes)
                pcm_16k = AudioProcessor.resample_to_16k(raw_array, sample_rate)
                dsp_time_ms = (time.perf_counter() - start_dsp) * 1000

                result = asr_model.transcribe(pcm_16k)
                audio_duration_ms = (len(pcm_16k) / 16000) * 1000

                reference_clean = source_text.lower().strip()
                hypothesis_clean = result.transcription.lower().strip()

                sample_wer = wer(reference_clean, hypothesis_clean) if reference_clean and hypothesis_clean else 1.0
                sample_cer = cer(reference_clean, hypothesis_clean) if reference_clean and hypothesis_clean else 1.0
                rtf = result.generation_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0.0

                # --- PHASE 2: Machine Translation ---
                translation_input = result.transcription  # Explicit capture of input layout boundary
                mt_result = translator.translate(translation_input)

                # --- PHASE 3: Record Construction ---
                record = {
                    "sample_id": sample_id,
                    "pipeline_status": "success",
                    "model_name": cached_asr_name,          # Inherited directly from run_asr_benchmark.py
                    "mt_model_name": cached_mt_name,        # Inherited directly from run_benchmark.py
                    "mt_model_version": cached_mt_version,
                    
                    # Full Transcript Cascade
                    "source_text": source_text,
                    "hypothesis": result.transcription,
                    "translation_input": translation_input,
                    "translation": mt_result.translation,
                    "reference_translation": reference_translation,
                    
                    # Core Subsystem Alignment Metrics
                    "wer": sample_wer,
                    "cer": sample_cer,
                    "rtf": rtf,
                    
                    # Precise Architectural Timings & Token Profiles
                    "audio_duration_ms": audio_duration_ms,
                    "dsp_time_ms": dsp_time_ms,
                    "inference_time_ms": result.generation_time_ms,  # Inherited ASR execution time key
                    
                    "mt_input_tokens": mt_result.input_tokens,
                    "mt_output_tokens": mt_result.output_tokens,
                    "mt_tokenization_time_ms": mt_result.tokenization_time_ms,
                    "mt_generation_time_ms": mt_result.generation_time_ms,
                    "mt_decoding_time_ms": mt_result.decoding_time_ms,
                    "mt_total_time_ms": mt_result.total_time_ms,
                    
                    "total_pipeline_time_ms": dsp_time_ms + result.generation_time_ms + mt_result.total_time_ms,
                }

                manager.save(record)
                
                consecutive_failures = 0  
                session_processed_samples += 1

                if session_processed_samples % LOG_INTERVAL == 0 or session_processed_samples == limit:
                    elapsed = time.perf_counter() - inference_start_perf
                    rate = session_processed_samples / elapsed if elapsed > 0 else 0
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    
                    logger.info(
                        "Processed %d samples | Elapsed: %s | Rate: %.2f samples/s | Pipeline Status: %s",
                        session_processed_samples, elapsed_str, rate, record["pipeline_status"]
                    )

            except TranslationError as e:
                logger.error("MT Phase failed for sample_id %s: %s", sample_id, e)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise CircuitBreakerError(f"Circuit breaker tripped: {consecutive_failures} consecutive MT faults.") from e
                continue

            except Exception as e:
                if isinstance(e, (TypeError, AttributeError, NameError, KeyError)):
                    logger.critical("Structural framework error detected for sample_id %s. Crashing.", sample_id)
                    raise e
                    
                logger.error("ASR Phase failed for sample_id %s: %s", sample_id, e)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise CircuitBreakerError(f"Circuit breaker tripped: {consecutive_failures} consecutive ASR faults.") from e
                continue

        logger.info("Dataset exhausted or limit reached. Flushing final data buffers...")
        manager.flush()
        manager.finalize()

    finally:
        hw_stats = hw_monitor.stop()

        try:
            asr_model.unload()
        except Exception as e:
            logger.warning("Non-fatal error during ASR backend teardown: %s", e)
            
        try:
            translator.close()
        except Exception as e:
            logger.warning("Non-fatal error during MT backend teardown: %s", e)
            
        try:
            manager.flush()
        except Exception as e:
            logger.error("Error during structural cleanup flush: %s", e)
            
    # 5. Final Telemetry Summary
    total_perf_time = time.perf_counter() - start_perf_time
    total_wall_time = time.time() - start_wall_time
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    samples_per_sec = (session_processed_samples / total_perf_time) if total_perf_time > 0 else 0
    
    logger.info("--- Benchmark Session Summary ---")
    logger.info("Run started at:       %s", start_timestamp)
    logger.info("Run finished at:      %s", end_timestamp)
    logger.info("Total wall time:      %s", str(timedelta(seconds=int(total_wall_time))))
    logger.info("Model load time:      %.2f seconds", load_time_seconds)
    logger.info("Sample throughput:    %.2f samples/sec", samples_per_sec)
    logger.info("Peak RAM Usage:       %.2f MB", hw_stats["peak_ram"])
    logger.info("Average RAM Usage:    %.2f MB", hw_stats["avg_ram"])
    logger.info("Average CPU Usage:    %.2f %%", hw_stats["avg_cpu"])
    logger.info("Peak CPU Usage:       %.2f %%", hw_stats["peak_cpu"])
    logger.info("-----------------------------")


def main() -> None:
    """CLI Entrypoint."""
    ProjectConfig.initialize()
    logger = BridgeLogger.get_logger("Main")
    
    parser = argparse.ArgumentParser(description="Run BridgeDEUX offline Speech Translation benchmark.")
    parser.add_argument("--model", type=str, required=True, choices=["whisper", "vosk"], help="ASR model to use.")
    parser.add_argument("--mt", type=str, required=True, choices=["marian", "m2m100"], help="Translation model to use.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate via CoVoSTProvider.")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process.")
    
    args = parser.parse_args()

    try:
        run_speech_translation_benchmark(
            asr_model_name=args.model, 
            mt_model_name=args.mt,
            split_name=args.split, 
            limit=args.limit
        )
    except KeyboardInterrupt:
        logger.warning("\nRun explicitly interrupted by user. Saving WAL checkpoint logs. Exiting.")
        sys.exit(0)
    except CircuitBreakerError as e:
        logger.critical("Benchmark Aborted by Safety Policy: %s", str(e))
        sys.exit(1)
    except CheckpointError as e:
        logger.critical("Infrastructure I/O Aborted: %s", str(e))
        sys.exit(1)
    except BenchmarkError as e:
        logger.critical("Benchmark Initialization Error: %s", str(e))
        sys.exit(1)
    except Exception as e:
        logger.critical("Benchmark Aborted due to unhandled fatal error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()