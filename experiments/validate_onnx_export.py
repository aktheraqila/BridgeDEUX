"""
BridgeDEUX - Comprehensive ONNX Verification Suite (Final v4.1)
Strict smoke-test regression suite featuring dynamic package metadata, 
flexible pre-load artifact verification, and character-level differential debugging.
"""

import time
import sys
import json
import platform
import statistics
import difflib
from importlib.metadata import version
from pathlib import Path
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import production baseline
from models.translators.marian import MarianTranslator
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MarianTokenizer

TEST_CORPUS = [
    "Guten Morgen.",
    "Die Architektur des Übersetzungssystems ist äußerst robust.",
    "Wenn wir das Modell auf ein mobiles Gerät portieren, müssen wir die Latenz optimieren.",
    "Der Compiler optimiert den Rechengraphen durch Operatorfusionierung und Quantisierung.",
    "Obwohl die Validierung fehlschlagen könnte, werden wir den Prozess fortsetzen."
]

# =====================================================================
# IMPORTANT:
# These values MUST remain synchronized with MarianTranslator v1.0 
# generation parameters. If MarianTranslator changes, update this 
# validation script accordingly to maintain strict A/B symmetry.
# =====================================================================
BASELINE_KWARGS = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "early_stopping": True,
}

def generate_char_diff(expected: str, actual: str) -> str:
    """Generates a clean visual diff identifying exact structural variations."""
    diff = difflib.ndiff(expected, actual)
    return "".join(diff)

def run_isolated_smoke_test():
    temp_onnx_dir = Path("temp/onnx_export")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" BRIDGEDEUX ONNX STAGE A: ISOLATED REGRESSION SMOKE TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Phase 1: Pre-load Artifact Verification (Version-Resilient)
    # ---------------------------------------------------------
    print("[1/6] Verifying exported artifact integrity...")
    
    if not temp_onnx_dir.exists():
        print(f"❌ Critical Error: Staged ONNX directory missing at {temp_onnx_dir}")
        sys.exit(1)

    # 1a. Check immutable base artifacts
    required_base_files = [
        "encoder_model.onnx", 
        "decoder_model.onnx", 
        "source.spm", 
        "target.spm"
    ]
    
    for file_name in required_base_files:
        if not (temp_onnx_dir / file_name).exists():
            print(f"❌ Critical Error: Missing required ONNX artifact: {file_name}")
            sys.exit(1)

    # 1b. Check for at least one valid KV-cache decoder graph
    cached_decoder_exists = any(
        (temp_onnx_dir / name).exists()
        for name in ("decoder_model_merged.onnx", "decoder_with_past_model.onnx")
    )
    
    if not cached_decoder_exists:
        print("❌ Critical Error: Missing cached decoder ONNX graph (merged or with_past).")
        sys.exit(1)

    # ---------------------------------------------------------
    # Phase 2: Environment Metadata Profiling
    # ---------------------------------------------------------
    print("[2/6] Profiling runtime environment metadata...")
    env_metadata = {
        "python_version": sys.version.split()[0],
        "os_platform": platform.platform(),
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "optimum_version": version("optimum"),
        "onnxruntime_version": version("onnxruntime"),
        "export_directory": str(temp_onnx_dir.absolute())
    }

    # ---------------------------------------------------------
    # Phase 3: Tokenizer Sanity & Vocab Matching
    # ---------------------------------------------------------
    try:
        print("[3/6] Validating tokenization vocab layers...")
        onnx_tokenizer = MarianTokenizer.from_pretrained(temp_onnx_dir)
        ref_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        
        if len(onnx_tokenizer.get_vocab()) != len(ref_tokenizer.get_vocab()):
            print("❌ Tokenizer Error: Exported vocabulary size drifts from baseline.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Tokenizer Error: Allocation failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Phase 4: Symmetrical Allocation & Parameter Lockdown
    # ---------------------------------------------------------
    try:
        print("[4/6] Initializing production baseline & ONNX Sessions...")
        pt_translator = MarianTranslator(device="cpu")
        pt_translator.load()
        
        onnx_start = time.perf_counter()
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(temp_onnx_dir, use_cache=True)
        onnx_load_time = time.perf_counter() - onnx_start
        
        env_metadata["active_execution_providers"] = onnx_model.providers
        print(f"  • Active Hardware Providers: {onnx_model.providers}")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Phase 5: Symmetrical Graph Warm-Up Sequence
    # ---------------------------------------------------------
    print("[5/6] Warming up execution providers...")
    warmup_text = "Warmup-Sequenz ausführen."
    _ = pt_translator.translate(warmup_text)
    
    warmup_inputs = onnx_tokenizer(warmup_text, return_tensors="pt")
    with torch.no_grad():
        _ = onnx_model.generate(**warmup_inputs, **BASELINE_KWARGS)
    print("  • Memory buffers settled. Execution layers live.")

    # ---------------------------------------------------------
    # Phase 6: Symmetrical Regression Loop
    # ---------------------------------------------------------
    print("[6/6] Running A/B inference pipeline...")
    
    pt_latencies = []
    onnx_latencies = []
    run_records = []
    exact_matches = 0

    for sentence in TEST_CORPUS:
        # Production PyTorch Run
        pt_res = pt_translator.translate(sentence)
        pt_latencies.append(pt_res.generation_time_ms)
        
        # Symmetrical ONNX Run
        inputs = onnx_tokenizer(sentence, return_tensors="pt")
        onnx_run_start = time.perf_counter()
        with torch.no_grad():
            onnx_tokens = onnx_model.generate(**inputs, **BASELINE_KWARGS)
        onnx_latency_ms = (time.perf_counter() - onnx_run_start) * 1000
        onnx_latencies.append(onnx_latency_ms)
        
        onnx_text = onnx_tokenizer.decode(onnx_tokens[0], skip_special_tokens=True)
        
        is_exact = (pt_res.translation.strip() == onnx_text.strip())
        diff_diagnostic = None
        
        if is_exact:
            exact_matches += 1
        else:
            diff_diagnostic = generate_char_diff(pt_res.translation, onnx_text)

        run_records.append({
            "source_text": sentence,
            "pytorch_translation": pt_res.translation,
            "onnx_translation": onnx_text,
            "exact_match": is_exact,
            "character_diff": diff_diagnostic,
            "pytorch_latency_ms": pt_res.generation_time_ms,
            "onnx_latency_ms": onnx_latency_ms
        })

    # Statistical Evaluation & Serialization
    stats_summary = {
        "environment_metadata": env_metadata,
        "onnx_load_time_seconds": onnx_load_time,
        "metrics": {
            "exact_match_ratio": f"{exact_matches}/{len(TEST_CORPUS)}",
            "pytorch_mean_latency_ms": statistics.mean(pt_latencies),
            "onnx_mean_latency_ms": statistics.mean(onnx_latencies)
        },
        "detailed_runs": run_records
    }

    artifact_path = results_dir / "onnx_stage_a_validation.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=4, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print(f"🎉 Smoke Test Execution Complete. Log saved to: {artifact_path}")
    print(f" • Symmetrical Exact Matches: {exact_matches}/{len(TEST_CORPUS)}")
    print(f" • PyTorch Average Latency:  {stats_summary['metrics']['pytorch_mean_latency_ms']:.2f} ms")
    print(f" • ONNX Average Latency:     {stats_summary['metrics']['onnx_mean_latency_ms']:.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_isolated_smoke_test()