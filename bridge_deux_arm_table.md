# ARM 100-Sample Benchmark

| Dataset | ASR | Platform | N | FP32 MT latency (ms/sample) | INT8 MT latency (ms/sample) | MT speedup (×) | FP32 CPU time (ms) | INT8 CPU time (ms) | CPU-time reduction (%) | FP32 E2E latency (ms/sample) | INT8 E2E latency (ms/sample) | E2E speedup (×) | FP32 tokens/s | INT8 tokens/s | FP32 memory Δ (MB) | INT8 memory Δ (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoVoST2 | Whisper | ARMv8 | 100 | 609.41 | 275.42 | 2.21 | 1815.01 | 719.76 | 60.3 | 253249.54 | 252915.55 | 1.001 | 25.29 | 56.11 | 2.00 | -0.02 |
| CoVoST2 | Whisper | ARMv9 | 100 | 198.18 | 83.90 | 2.36 | 940.37 | 279.82 | 70.2 | 50974.32 | 50860.04 | 1.002 | 70.44 | 165.69 | 2.78 | 1.79 |
| CoVoST2 | W1 | ARMv8 | 100 | 551.77 | 253.69 | 2.17 | 1730.47 | 655.59 | 62.1 | 220739.91 | 220441.83 | 1.001 | 27.30 | 60.42 | 0.79 | 0.16 |
| CoVoST2 | W1 | ARMv9 | 100 | 201.43 | 106.58 | 1.89 | 974.74 | 323.36 | 66.8 | 51200.78 | 51105.93 | 1.002 | 67.90 | 134.79 | 2.31 | -0.12 |

**Interpretation:** MT speedup = FP32 MT latency / INT8 MT latency. E2E latency includes the measured ASR latency plus MarianMT latency. Memory columns are **memory delta (MB)**, not absolute peak memory.
