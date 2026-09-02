#!/usr/bin/env python3
import sys
sys.path.insert(0, r"D:\BridgeDEUX")

from figure_scripts.generate_bridge_deux_tables import desktop_summary

try:
    result = desktop_summary(
        filename="analysis/covost2_100_desktop/covost2_100_whisper_desktop/covost2_100_whisper_desktop_results.csv",
        dataset="CoVoST2",
        asr="Whisper",
        asr_latency_key="whisper_inference_time_ms",
        clean_chrf_fp32="chrf_clean_f",
        clean_chrf_int8="chrf_clean_i",
        asr_chrf_fp32="chrf_whisper_asr_f",
        asr_chrf_int8="chrf_whisper_asr_i",
        clean_comet_fp32="comet_clean_f",
        clean_comet_int8="comet_clean_i",
        asr_comet_fp32="comet_whisper_asr_f_e2e",
        asr_comet_int8="comet_whisper_asr_i_e2e",
    )
    print(f"Success! Result keys: {result.keys()}")
    print(f"Sample values: n={result['n']}, asr_latency={result['asr_latency']}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
