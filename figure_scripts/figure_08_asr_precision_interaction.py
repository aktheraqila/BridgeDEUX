from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "analysis"
    / "scored_qirg_cohort_complete.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA
# =============================================================================

def load_results():

    df = pd.read_parquet(
        INPUT_FILE,
        columns=[
            "chrf_clean_f",
            "chrf_clean_i",
            "chrf_asr_f",
            "chrf_asr_i",
        ],
    )

    df = df.dropna(
        subset=[
            "chrf_clean_f",
            "chrf_clean_i",
            "chrf_asr_f",
            "chrf_asr_i",
        ]
    ).copy()

    return df


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    df = load_results()

    n = len(df)

    # -------------------------------------------------------------------------
    # Mean translation quality
    # -------------------------------------------------------------------------

    clean_fp32 = df["chrf_clean_f"].mean()
    clean_int8 = df["chrf_clean_i"].mean()

    asr_fp32 = df["chrf_asr_f"].mean()
    asr_int8 = df["chrf_asr_i"].mean()

    # -------------------------------------------------------------------------
    # Within-condition quantization changes
    # -------------------------------------------------------------------------

    clean_delta = clean_int8 - clean_fp32
    asr_delta = asr_int8 - asr_fp32

    # -------------------------------------------------------------------------
    # Difference-in-differences
    #
    # DiD = (INT8 - FP32)_ASR - (INT8 - FP32)_Clean
    # -------------------------------------------------------------------------

    did = asr_delta - clean_delta

    # =============================================================================
    # CONSOLE OUTPUT
    # =============================================================================

    print("=" * 72)
    print("FIGURE 8 — ASR × PRECISION INTERACTION")
    print("=" * 72)

    print(f"Samples               : {n:,}")
    print()

    print("CLEAN INPUT")
    print(f"FP32 mean chrF++      : {clean_fp32:.4f}")
    print(f"INT8 mean chrF++      : {clean_int8:.4f}")
    print(f"INT8 - FP32           : {clean_delta:+.4f}")
    print()

    print("ASR INPUT")
    print(f"FP32 mean chrF++      : {asr_fp32:.4f}")
    print(f"INT8 mean chrF++      : {asr_int8:.4f}")
    print(f"INT8 - FP32           : {asr_delta:+.4f}")
    print()

    print("DIFFERENCE-IN-DIFFERENCES")
    print(
        f"DiD                   : {did:+.4f} chrF++"
    )

    print()

    # =============================================================================
    # PLOT
    # =============================================================================

    fig, ax = plt.subplots(
        figsize=(9.5, 6.2)
    )

    x = np.array([0, 1])

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------

    clean_color = "#355C7D"
    asr_color = "#2A7F62"

    # -------------------------------------------------------------------------
    # Interaction lines
    # -------------------------------------------------------------------------

    ax.plot(
        x,
        [clean_fp32, clean_int8],
        marker="o",
        markersize=9,
        linewidth=2.6,
        color=clean_color,
        label="Clean input",
        zorder=3,
    )

    ax.plot(
        x,
        [asr_fp32, asr_int8],
        marker="o",
        markersize=9,
        linewidth=2.6,
        color=asr_color,
        label="ASR input",
        zorder=3,
    )

    # =============================================================================
    # VALUE LABELS
    # =============================================================================

    # Clean labels
    ax.annotate(
        f"{clean_fp32:.2f}",
        xy=(0, clean_fp32),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=clean_color,
    )

    ax.annotate(
        f"{clean_int8:.2f}",
        xy=(1, clean_int8),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=clean_color,
    )

    # ASR labels
    ax.annotate(
        f"{asr_fp32:.2f}",
        xy=(0, asr_fp32),
        xytext=(0, -15),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color=asr_color,
    )

    ax.annotate(
        f"{asr_int8:.2f}",
        xy=(1, asr_int8),
        xytext=(0, -15),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color=asr_color,
    )

    # =============================================================================
    # DELTA ANNOTATIONS
    # =============================================================================

    # Clean delta
    ax.text(
        0.50,
        clean_fp32 + 0.65,
        f"Clean Δ = {clean_delta:+.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color=clean_color,
    )

    # ASR delta
    ax.text(
        0.50,
        asr_fp32 - 0.75,
        f"ASR Δ = {asr_delta:+.3f}",
        ha="center",
        va="top",
        fontsize=10,
        color=asr_color,
    )

    # =============================================================================
    # DiD ANNOTATION
    # =============================================================================

    ax.text(
        0.50,
        0.52,
        f"DiD = {did:+.3f} chrF++",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    # =============================================================================
    # AXES
    # =============================================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "FP32",
            "Dynamic INT8",
        ],
        fontsize=11,
    )

    ax.set_xlabel(
        "MarianMT Precision",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mean chrF++ Score",
        fontsize=12,
    )

    ax.set_title(
        "ASR × Precision Interaction in Translation Quality",
        fontsize=15,
        pad=14,
    )

    # =============================================================================
    # Y AXIS
    # =============================================================================

    all_values = [
        clean_fp32,
        clean_int8,
        asr_fp32,
        asr_int8,
    ]

    ymin = min(all_values)
    ymax = max(all_values)

    margin = (ymax - ymin) * 0.45

    ax.set_ylim(
        ymin - margin,
        ymax + margin,
    )

    # =============================================================================
    # GRID
    # =============================================================================

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )

    ax.set_axisbelow(True)

    # =============================================================================
    # FRAME
    # =============================================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # =============================================================================
    # LEGEND
    # =============================================================================

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=10,
    )

    # =============================================================================
    # SAMPLE INFORMATION
    # =============================================================================

    ax.text(
        0.02,
        0.96,
        f"CoVoST2 test set: N = {n:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    # =============================================================================
    # LAYOUT
    # =============================================================================

    fig.tight_layout()

    # =============================================================================
    # SAVE PNG
    # =============================================================================

    png_path = (
        OUTPUT_DIR
        / "figure_08_asr_precision_interaction.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # =============================================================================
    # SAVE PDF
    # =============================================================================

    pdf_path = (
        OUTPUT_DIR
        / "figure_08_asr_precision_interaction.pdf"
    )

    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print()
    print("Figure generated successfully.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    generate_figure()