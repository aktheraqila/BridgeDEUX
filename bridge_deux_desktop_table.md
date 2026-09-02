# Desktop 100-Sample Benchmark

| Dataset | ASR | N | ASR latency (ms/sample) | Clean FP32 chrF++ | Clean INT8 chrF++ | Clean Δ chrF++ | ASR FP32 chrF++ | ASR INT8 chrF++ | ASR Δ chrF++ | chrF++ DiD | Clean FP32 COMET | Clean INT8 COMET | Clean Δ COMET | ASR FP32 COMET | ASR INT8 COMET | ASR Δ COMET | COMET DiD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoVoST2 | Whisper | 100 | 3062.95 | 67.04 | 66.89 | -0.16 | 44.43 | 44.91 | 0.49 | 0.64 | 0.659 | 0.665 | 0.007 | -0.307 | -0.276 | 0.031 | 0.024 |
| CoVoST2 | W1 | 100 | 3253.72 | 67.04 | 66.89 | -0.16 | 46.76 | 46.70 | -0.06 | 0.10 | 0.659 | 0.665 | 0.007 | -0.207 | -0.248 | -0.040 | -0.047 |

**Interpretation:** Δ = INT8 − FP32. DiD = (ASR INT8 − ASR FP32) − (Clean INT8 − Clean FP32).
