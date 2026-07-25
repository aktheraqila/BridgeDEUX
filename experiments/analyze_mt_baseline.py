import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_mt_graphs(marian_csv: str, m2m_csv: str, output_dir: str):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df_marian = pd.read_csv(marian_csv)
    df_m2m = pd.read_csv(m2m_csv)
    
    # 1. Comparative Latency Distribution Graph
    plt.figure(figsize=(9, 6))
    
    sns.kdeplot(df_marian['total_time_ms'], label='MarianMT (Helsinki-NLP)', color='#3498db', fill=True, alpha=0.4)
    sns.kdeplot(df_m2m['total_time_ms'], label='M2M100 (Facebook)', color='#e74c3c', fill=True, alpha=0.4)
    
    plt.title('Inference Latency Comparison: MarianMT vs M2M100')
    plt.xlabel('Total Processing Time (ms)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    latency_fig_path = out_path / 'mt_latency_comparison.png'
    plt.savefig(latency_fig_path, dpi=300)
    print(f"Saved Latency Comparison graph to: {latency_fig_path}")
    plt.close()

    # 2. Token Generation Speed Graph
    # Calculate time per token (ms)
    df_marian['ms_per_token'] = df_marian['generation_time_ms'] / df_marian['output_tokens']
    df_m2m['ms_per_token'] = df_m2m['generation_time_ms'] / df_m2m['output_tokens']

    plt.figure(figsize=(9, 6))
    sns.kdeplot(df_marian['ms_per_token'], label='MarianMT', color='#2ecc71', fill=True, alpha=0.4)
    sns.kdeplot(df_m2m['ms_per_token'], label='M2M100', color='#9b59b6', fill=True, alpha=0.4)
    
    plt.title('Token Generation Speed Comparison')
    plt.xlabel('Milliseconds per Output Token')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    speed_fig_path = out_path / 'mt_token_speed_comparison.png'
    plt.savefig(speed_fig_path, dpi=300)
    print(f"Saved Token Speed Comparison graph to: {speed_fig_path}")
    plt.close()

    print("\nMachine Translation analysis complete! Check the output folder.")

if __name__ == "__main__":
    generate_mt_graphs(
        marian_csv="results/marianmt_covost2_de_en_test/marianmt_covost2_de_en_test_results.csv",
        m2m_csv="results/m2m100_covost2_de_en_test/m2m100_covost2_de_en_test_results.csv",
        output_dir="results/analysis_graphs"
    )