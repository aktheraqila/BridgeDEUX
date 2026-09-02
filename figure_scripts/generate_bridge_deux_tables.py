#!/usr/bin/env python3
"""
BridgeDEUX — generate the Desktop and ARM benchmark tables from the
attached 100-sample CSV result files.

This script DOES NOT create replacement CSV files.
It reads the raw CSVs and prints two publication/thesis-ready tables:
  1) Desktop benchmark table
  2) ARM/mobile benchmark table

It also writes:
  - bridge_deux_desktop_table.md
  - bridge_deux_arm_table.md

Requirements:
    Python 3.10+
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_csv(filename: str) -> list[dict[str, str]]:
    path = ROOT / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        raise ValueError(f"No numeric values found for column '{key}'")
    return statistics.fmean(vals)


def fmt(x: float | int | str, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


def desktop_summary(
    filename: str,
    dataset: str,
    asr: str,
    asr_latency_key: str,
    clean_chrf_fp32: str,
    clean_chrf_int8: str,
    asr_chrf_fp32: str,
    asr_chrf_int8: str,
    clean_comet_fp32: str,
    clean_comet_int8: str,
    asr_comet_fp32: str,
    asr_comet_int8: str,
) -> dict[str, float | int | str]:
    rows = read_csv(filename)

    n = len(rows)

    cfp = mean(rows, clean_chrf_fp32)
    cit = mean(rows, clean_chrf_int8)
    afp = mean(rows, asr_chrf_fp32)
    ait = mean(rows, asr_chrf_int8)

    ccomet_fp = mean(rows, clean_comet_fp32)
    ccomet_i8 = mean(rows, clean_comet_int8)
    acomet_fp = mean(rows, asr_comet_fp32)
    acomet_i8 = mean(rows, asr_comet_int8)

    return {
        "dataset": dataset,
        "asr": asr,
        "n": n,
        "asr_latency": mean(rows, asr_latency_key),

        "clean_fp32_chrf": cfp,
        "clean_int8_chrf": cit,
        "clean_chrf_delta": cit - cfp,

        "asr_fp32_chrf": afp,
        "asr_int8_chrf": ait,
        "asr_chrf_delta": ait - afp,

        "chrf_did": (ait - afp) - (cit - cfp),

        "clean_fp32_comet": ccomet_fp,
        "clean_int8_comet": ccomet_i8,
        "clean_comet_delta": ccomet_i8 - ccomet_fp,

        "asr_fp32_comet": acomet_fp,
        "asr_int8_comet": acomet_i8,
        "asr_comet_delta": acomet_i8 - acomet_fp,

        "comet_did": (acomet_i8 - acomet_fp) - (ccomet_i8 - ccomet_fp),
    }


def mobile_summary(
    filename: str,
    dataset: str,
    asr: str,
    platform: str,
) -> dict[str, float | int | str]:
    rows = read_csv(filename)

    fp32_mt = mean(rows, "fp32_latency_ms")
    int8_mt = mean(rows, "int8_latency_ms")
    fp32_cpu = mean(rows, "fp32_cpu_time_ms")
    int8_cpu = mean(rows, "int8_cpu_time_ms")
    fp32_tps = mean(rows, "fp32_tps")
    int8_tps = mean(rows, "int8_tps")
    fp32_mem = mean(rows, "fp32_memory_delta_mb")
    int8_mem = mean(rows, "int8_memory_delta_mb")

    # The mobile files contain sample-level ASR latency. Keep the
    # end-to-end calculation paired per sample rather than adding
    # separate means.
    e2e_fp32 = statistics.fmean(
        float(r["whisper_latency_ms"]) + float(r["fp32_latency_ms"])
        for r in rows
    )
    e2e_int8 = statistics.fmean(
        float(r["whisper_latency_ms"]) + float(r["int8_latency_ms"])
        for r in rows
    )

    return {
        "dataset": dataset,
        "asr": asr,
        "platform": platform,
        "n": len(rows),

        "fp32_mt": fp32_mt,
        "int8_mt": int8_mt,
        "mt_speedup": fp32_mt / int8_mt,

        "fp32_cpu": fp32_cpu,
        "int8_cpu": int8_cpu,
        "cpu_reduction_pct": 100.0 * (fp32_cpu - int8_cpu) / fp32_cpu,

        "fp32_e2e": e2e_fp32,
        "int8_e2e": e2e_int8,
        "e2e_speedup": e2e_fp32 / e2e_int8,

        "fp32_tps": fp32_tps,
        "int8_tps": int8_tps,

        "fp32_mem": fp32_mem,
        "int8_mem": int8_mem,
    }


# ----------------------------------------------------------------------
# DATA CONFIGURATION
# ----------------------------------------------------------------------

DESKTOP_SOURCES = [
    {
        "filename": "analysis/covost2_100_desktop/covost2_100_whisper_desktop/covost2_100_whisper_desktop_results.csv",
        "dataset": "CoVoST2",
        "asr": "Whisper",
        "asr_latency_key": "whisper_inference_time_ms",
        "clean_chrf_fp32": "chrf_clean_f",
        "clean_chrf_int8": "chrf_clean_i",
        "asr_chrf_fp32": "chrf_whisper_asr_f",
        "asr_chrf_int8": "chrf_whisper_asr_i",
        "clean_comet_fp32": "comet_clean_f",
        "clean_comet_int8": "comet_clean_i",
        "asr_comet_fp32": "comet_whisper_asr_f_e2e",
        "asr_comet_int8": "comet_whisper_asr_i_e2e",
    },
    {
        "filename": "analysis/covost2_100_desktop/covost2_100_w1_desktop/covost2_100_w1_desktop_results.csv",
        "dataset": "CoVoST2",
        "asr": "W1",
        "asr_latency_key": "w1_inference_time_ms",
        "clean_chrf_fp32": "chrf_clean_f",
        "clean_chrf_int8": "chrf_clean_i",
        "asr_chrf_fp32": "chrf_w1_asr_f",
        "asr_chrf_int8": "chrf_w1_asr_i",
        "clean_comet_fp32": "comet_clean_f",
        "clean_comet_int8": "comet_clean_i",
        "asr_comet_fp32": "comet_w1_asr_f_e2e",
        "asr_comet_int8": "comet_w1_asr_i_e2e",
    },
    # NOTE: MSLT files are incomplete (missing inference time and/or metric columns); skipping
    # {
    #     "filename": "analysis/mslt_100_desktop/whisper/mslt_100_desktop_results.csv",
    #     ...
    # },
    # {
    #     "filename": "analysis/mslt_100_desktop/w1/mslt_100_desktop_w1_results.csv",
    #     ...
    # },
]

ARM_SOURCES = [
    ("CoVoST2", "Whisper", "ARMv8",
     "analysis/Mobile Benchmark Results/Whisper Model-20260830T205442Z-1-001/Whisper Model/covost2/mobile_benchmark_results(HUAWEI_Kirin710F).csv"),
    ("CoVoST2", "Whisper", "ARMv9",
     "analysis/Mobile Benchmark Results/Whisper Model-20260830T205442Z-1-001/Whisper Model/covost2/mobile_benchmark_results(Honor600_Qualcomm_Snapdragon7Gen4).csv"),
    ("CoVoST2", "W1", "ARMv8",
     "analysis/Mobile Benchmark Results/Student Model-20260830T205443Z-1-001/Student Model/covost2/mobile_student_benchmark_results_covost2_HONOR9X.csv"),
    ("CoVoST2", "W1", "ARMv9",
     "analysis/Mobile Benchmark Results/Student Model-20260830T205443Z-1-001/Student Model/covost2/mobile_student_benchmark_results_covost2_HONOR600.csv"),
    # NOTE: MSLT mobile files are incomplete; skipping
    # ("MSLT", "Whisper", "ARMv8",
    #  "analysis/Mobile Benchmark Results/Whisper Model.../mobile_benchmark_results_mslt_HUAWEI_STK-LX1.csv"),
    # ("MSLT", "Whisper", "ARMv9",
    #  "analysis/Mobile Benchmark Results/Whisper Model.../mobile_benchmark_results_mslt_HONOR600.csv"),
    # ("MSLT", "W1", "ARMv8",
    #  "analysis/Mobile Benchmark Results/Student Model.../mobile_student_benchmark_results_mslt_HONOR9X.csv"),
    # ("MSLT", "W1", "ARMv9",
    #  "analysis/Mobile Benchmark Results/Student Model.../mobile_student_benchmark_results_mslt_HONOR600.csv"),
]


def make_markdown_tables() -> tuple[str, str]:
    desktop = [
        desktop_summary(**cfg)
        for cfg in DESKTOP_SOURCES
    ]

    arm = [
        mobile_summary(
            filename=filename,
            dataset=dataset,
            asr=asr,
            platform=platform,
        )
        for dataset, asr, platform, filename in ARM_SOURCES
    ]

    desktop_table = """# Desktop 100-Sample Benchmark

| Dataset | ASR | N | ASR latency (ms/sample) | Clean FP32 chrF++ | Clean INT8 chrF++ | Clean Δ chrF++ | ASR FP32 chrF++ | ASR INT8 chrF++ | ASR Δ chrF++ | chrF++ DiD | Clean FP32 COMET | Clean INT8 COMET | Clean Δ COMET | ASR FP32 COMET | ASR INT8 COMET | ASR Δ COMET | COMET DiD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""

    for r in desktop:
        desktop_table += (
            f"| {r['dataset']} | {r['asr']} | {r['n']} | "
            f"{fmt(r['asr_latency'])} | "
            f"{fmt(r['clean_fp32_chrf'])} | {fmt(r['clean_int8_chrf'])} | {fmt(r['clean_chrf_delta'])} | "
            f"{fmt(r['asr_fp32_chrf'])} | {fmt(r['asr_int8_chrf'])} | {fmt(r['asr_chrf_delta'])} | {fmt(r['chrf_did'])} | "
            f"{fmt(r['clean_fp32_comet'], 3)} | {fmt(r['clean_int8_comet'], 3)} | {fmt(r['clean_comet_delta'], 3)} | "
            f"{fmt(r['asr_fp32_comet'], 3)} | {fmt(r['asr_int8_comet'], 3)} | {fmt(r['asr_comet_delta'], 3)} | {fmt(r['comet_did'], 3)} |\n"
        )

    desktop_table += (
        "\n**Interpretation:** Δ = INT8 − FP32. "
        "DiD = (ASR INT8 − ASR FP32) − (Clean INT8 − Clean FP32).\n"
    )

    arm_table = """# ARM 100-Sample Benchmark

| Dataset | ASR | Platform | N | FP32 MT latency (ms/sample) | INT8 MT latency (ms/sample) | MT speedup (×) | FP32 CPU time (ms) | INT8 CPU time (ms) | CPU-time reduction (%) | FP32 E2E latency (ms/sample) | INT8 E2E latency (ms/sample) | E2E speedup (×) | FP32 tokens/s | INT8 tokens/s | FP32 memory Δ (MB) | INT8 memory Δ (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""

    for r in arm:
        arm_table += (
            f"| {r['dataset']} | {r['asr']} | {r['platform']} | {r['n']} | "
            f"{fmt(r['fp32_mt'])} | {fmt(r['int8_mt'])} | {fmt(r['mt_speedup'])} | "
            f"{fmt(r['fp32_cpu'])} | {fmt(r['int8_cpu'])} | {fmt(r['cpu_reduction_pct'], 1)} | "
            f"{fmt(r['fp32_e2e'])} | {fmt(r['int8_e2e'])} | {fmt(r['e2e_speedup'], 3)} | "
            f"{fmt(r['fp32_tps'])} | {fmt(r['int8_tps'])} | "
            f"{fmt(r['fp32_mem'])} | {fmt(r['int8_mem'])} |\n"
        )

    arm_table += (
        "\n**Interpretation:** MT speedup = FP32 MT latency / INT8 MT latency. "
        "E2E latency includes the measured ASR latency plus MarianMT latency. "
        "Memory columns are **memory delta (MB)**, not absolute peak memory.\n"
    )

    return desktop_table, arm_table


def main() -> None:
    desktop_table, arm_table = make_markdown_tables()

    desktop_out = ROOT / "bridge_deux_desktop_table.md"
    arm_out = ROOT / "bridge_deux_arm_table.md"

    desktop_out.write_text(desktop_table, encoding="utf-8")
    arm_out.write_text(arm_table, encoding="utf-8")

    # Print with UTF-8 encoding to handle special characters
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    print(desktop_table)
    print("\n" + "=" * 120 + "\n")
    print(arm_table)

    print(f"\nSaved: {desktop_out}")
    print(f"Saved: {arm_out}")


if __name__ == "__main__":
    main()
