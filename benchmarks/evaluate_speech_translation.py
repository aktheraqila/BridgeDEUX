# benchmarks/evaluate_speech_translation.py
"""
BridgeDEUX Core Framework
Cascaded Speech Translation Evaluation Pipeline
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

def load_and_validate_csv(file_path: Path) -> pd.DataFrame:
    """Loads the benchmark artifact and verifies structural completeness."""
    if not file_path.exists():
        raise FileNotFoundError(f"Target benchmark file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Ensure critical columns exist before processing
    required_cols = [
        "sample_id", "hypothesis", "translation", "reference_translation",
        "wer", "cer", "dsp_time_ms", "inference_time_ms", "mt_total_time_ms", "total_pipeline_time_ms"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required infrastructure columns in CSV: {missing}")
        
    return df

def calculate_text_metrics(df: pd.DataFrame) -> dict:
    """Computes downstream translation metrics across the entire corpus."""
    metrics = {}
    
    # Fill any catastrophic NaN values with empty strings safely
    sys_translations = df["translation"].fillna("").astype(str).tolist()
    ref_translations = df["reference_translation"].fillna("").astype(str).tolist()
    
    # Mean ASR Metrics
    metrics["mean_wer"] = df["wer"].mean()
    metrics["mean_cer"] = df["cer"].mean()
    
    if sacrebleu and len(sys_translations) > 0:
        # sacreBLEU expects a list of references per system output (list of lists)
        bleu = sacrebleu.corpus_bleu(sys_translations, [ref_translations])
        chrf = sacrebleu.corpus_chrf(sys_translations, [ref_translations])
        metrics["sacreBLEU"] = bleu.score
        metrics["chrF++"] = chrf.score
    else:
        metrics["sacreBLEU"] = np.nan
        metrics["chrF++"] = np.nan
        
    return metrics

def calculate_telemetry_metrics(df: pd.DataFrame) -> dict:
    """Aggregates compute latency windows to profile efficiency bottlenecks."""
    timing_cols = ["dsp_time_ms", "inference_time_ms", "mt_total_time_ms", "total_pipeline_time_ms"]
    stats = {}
    
    for col in timing_cols:
        stats[f"{col}_mean"] = df[col].mean()
        stats[f"{col}_p95"] = df[col].quantile(0.95)
        stats[f"{col}_p99"] = df[col].quantile(0.99)
        
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate BridgeDEUX Cascaded Speech Translation Artifacts.")
    parser.add_argument("--file", type=str, required=True, help="Path to the generated benchmark CSV.")
    args = parser.parse_args()
    
    file_path = Path(args.file)
    print(f"Parsing performance metrics from: {file_path.name}...")
    
    try:
        df = load_and_validate_csv(file_path)
        
        text_stats = calculate_text_metrics(df)
        time_stats = calculate_telemetry_metrics(df)
        
        print("\n=== 📝 Translation Quality Metrics ===")
        print(f"Evaluated Samples : {len(df)}")
        print(f"Mean ASR WER      : {text_stats['mean_wer'] * 100:.2f}%")
        print(f"Mean ASR CER      : {text_stats['mean_cer'] * 100:.2f}%")
        if sacrebleu:
            print(f"Corpus sacreBLEU  : {text_stats['sacreBLEU']:.2f}")
            print(f"Corpus chrF++     : {text_stats['chrF++']:.2f}")
        else:
            print("WARNING: 'sacrebleu' library missing. Run `pip install sacrebleu` to compute BLEU/chrF++.")
            
        print("\n=== ⚡ Latency & Telemetry Profiles ===")
        print(f"DSP Resampling    -> Mean: {time_stats['dsp_time_ms_mean']:.1f}ms | p99: {time_stats['dsp_time_ms_p99']:.1f}ms")
        print(f"ASR Inference     -> Mean: {time_stats['inference_time_ms_mean']:.1f}ms | p99: {time_stats['inference_time_ms_p99']:.1f}ms")
        print(f"MT Engine         -> Mean: {time_stats['mt_total_time_ms_mean']:.1f}ms | p99: {time_stats['mt_total_time_ms_p99']:.1f}ms")
        print(f"Total Cascade     -> Mean: {time_stats['total_pipeline_time_ms_mean']:.1f}ms | p99: {time_stats['total_pipeline_time_ms_p99']:.1f}ms")
        
    except Exception as e:
        print(f"\n❌ Evaluation Failed: {str(e)}")

if __name__ == "__main__":
    main()