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
        ],
    )

    df = df.dropna(
        subset=[
            "chrf_clean_f",
            "chrf_clean_i",
        ]
    ).copy()

    # Δ = INT8 − FP32
    df["delta"] = (
        df["chrf_clean_i"]
        - df["chrf_clean_f"]
    )

    return df


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    df = load_results()

    n = len(df)
    delta = df["delta"]

    # -------------------------------------------------------------------------
    # Outcome statistics
    # -------------------------------------------------------------------------

    identical = (delta == 0).sum()
    fp32_better = (delta < 0).sum()
    int8_better = (delta > 0).sum()

    identical_pct = identical / n * 100
    fp32_pct = fp32_better / n * 100
    int8_pct = int8_better / n * 100

    mean_delta = delta.mean()
    std_delta = delta.std()

    # Only non-identical samples are plotted.
    nonzero_delta = delta[delta != 0]

    fp32_delta = nonzero_delta[nonzero_delta < 0]
    int8_delta = nonzero_delta[nonzero_delta > 0]

    # Symmetric bins around zero.
    max_abs = max(
        abs(nonzero_delta.min()),
        abs(nonzero_delta.max()),
    )

    bins = np.linspace(
        -max_abs,
        max_abs,
        45,
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print("=" * 72)
    print("FIGURE 5 — FP32–INT8 PER-SAMPLE QUALITY DISTRIBUTION")
    print("=" * 72)
    print(f"Samples              : {n:,}")
    print(f"Identical            : {identical:,} ({identical_pct:.1f}%)")
    print(f"FP32 better          : {fp32_better:,} ({fp32_pct:.1f}%)")
    print(f"INT8 better          : {int8_better:,} ({int8_pct:.1f}%)")
    print(f"Mean Δ               : {mean_delta:+.4f}")
    print(f"Std. deviation       : {std_delta:.4f}")
    print()

    # =============================================================================
    # PLOT
    # =============================================================================

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    # -------------------------------------------------------------------------
    # Histogram
    # -------------------------------------------------------------------------

    ax.hist(
        fp32_delta,
        bins=bins,
        alpha=0.85,
        label=f"FP32 better  ({fp32_better:,})",
    )

    ax.hist(
        int8_delta,
        bins=bins,
        alpha=0.85,
        label=f"INT8 better  ({int8_better:,})",
    )

    # -------------------------------------------------------------------------
    # Zero reference
    # -------------------------------------------------------------------------

    ax.axvline(
        0,
        linewidth=1.4,
        linestyle="-",
    )

    # -------------------------------------------------------------------------
    # Mean reference
    # -------------------------------------------------------------------------

    ax.axvline(
        mean_delta,
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
    )

    # Put mean in empty lower-right area.
    ax.text(
        0.98,
        0.08,
        f"Mean Δ = {mean_delta:+.3f} chrF++",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    # =============================================================================
    # TITLE
    # =============================================================================

    ax.set_title(
        "Per-Sample FP32–INT8 Translation Quality Difference",
        fontsize=16,
        pad=14,
    )

    # =============================================================================
    # AXES
    # =============================================================================

    ax.set_xlabel(
        "Δ chrF++ (INT8 − FP32)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Number of Non-Identical Samples",
        fontsize=12,
    )

    # =============================================================================
    # METADATA
    # =============================================================================

    ax.text(
        0.98,
        0.95,
        (
            f"CoVoST2 test set: N = {n:,}\n"
            f"Identical: {identical:,} ({identical_pct:.1f}%)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    # =============================================================================
    # GRID
    # =============================================================================

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
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
        loc="upper left",
        fontsize=10,
    )

    # =============================================================================
    # LAYOUT
    # =============================================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.13,
        top=0.88,
    )

    # =============================================================================
    # SAVE PNG
    # =============================================================================

    png_path = (
        OUTPUT_DIR
        / "figure_05_fp32_int8_distribution.png"
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
        / "figure_05_fp32_int8_distribution.pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print()
    print("Figure generated successfully.")


if __name__ == "__main__":
    generate_figure()