import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_benchmark_results(csv_path: str, output_dir: str):
    # Set publication-ready aesthetics for your thesis
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    file_path = Path(csv_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        print(f"Error: Could not find {file_path}")
        return

    print(f"Loading data from {file_path.name}...")
    df = pd.read_csv(file_path)
    
    # 1. Summary Statistics
    print("\n--- Final Whisper Base Telemetry (Full Dataset) ---")
    print(f"Total Samples Processed : {len(df)}")
    print(f"Mean Word Error Rate    : {df['wer'].mean():.4f}")
    print(f"Mean Character Error    : {df['cer'].mean():.4f}")
    print(f"Mean Inference Time     : {df['inference_time_ms'].mean():.2f} ms")
    print(f"Mean DSP Processing     : {df['dsp_time_ms'].mean():.2f} ms")
    print(f"Mean Total Pipeline     : {df['total_pipeline_time_ms'].mean():.2f} ms")
    print("---------------------------------------------------")
    
    # 2. Graph: WER Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['wer'], bins=50, kde=True, color='#2c3e50', element='step')
    plt.title('Word Error Rate (WER) Distribution - Whisper Base')
    plt.xlabel('Word Error Rate')
    plt.ylabel('Frequency')
    plt.tight_layout()
    wer_fig_path = out_path / 'whisper_wer_distribution.png'
    plt.savefig(wer_fig_path, dpi=300)
    print(f"Saved WER distribution graph to: {wer_fig_path}")
    plt.close()

    # 3. Graph: Latency Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['inference_time_ms'], bins=50, kde=True, color='#27ae60', element='step')
    plt.title('Inference Latency Distribution - Whisper Base')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    latency_fig_path = out_path / 'whisper_latency_distribution.png'
    plt.savefig(latency_fig_path, dpi=300)
    print(f"Saved Latency distribution graph to: {latency_fig_path}")
    plt.close()
    
    print("\nAnalysis complete! Your thesis graphics are ready in the output folder.")

if __name__ == "__main__":
    # Adjust the csv_path if your file is located elsewhere
    analyze_benchmark_results(
        csv_path="results/whisper.cpp (base)_test/whisper.cpp (base)_test_results.csv",
        output_dir="results/analysis_graphs"
    )