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
# DEFENSE RESULT
# =============================================================================
# Important distinction:
#
# 3,660 / 13,511 = 27.09%  -> different generated output strings
# 3,621 / 13,511 = 26.80%  -> different chrF++ scores
# 9,890 / 13,511 = 73.20%  -> same chrF++ score
#
# The parquet columns used below contain chrF++ scores, not generated strings.
# Therefore the 3,660 output-divergence result is a reported thesis result,
# while the histogram itself is based on chrF++ score differences.
# =============================================================================

EXPECTED_N = 13_511
OUTPUT_DIVERGENT = 3_660


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

    # Delta = INT8 - FP32
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

    if n != EXPECTED_N:
        raise ValueError(
            f"Expected CoVoST2 test set N={EXPECTED_N:,}, "
            f"but found N={n:,}. "
            "Check the input parquet/cohort before generating the defense figure."
        )

    delta = df["delta"]

    # =========================================================================
    # CHRF++ SCORE STATISTICS
    # =========================================================================

    same_score = (delta == 0).sum()

    fp32_better = (delta < 0).sum()
    int8_better = (delta > 0).sum()

    score_divergent = fp32_better + int8_better

    same_score_pct = same_score / n * 100
    score_divergent_pct = score_divergent / n * 100

    fp32_pct = fp32_better / n * 100
    int8_pct = int8_better / n * 100

    mean_delta = delta.mean()
    std_delta = delta.std()

    # =========================================================================
    # OUTPUT-STRING DIVERGENCE
    # =========================================================================

    output_divergent = OUTPUT_DIVERGENT
    output_divergent_pct = output_divergent / n * 100

    same_output = n - output_divergent
    same_output_pct = same_output / n * 100

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================

    print("=" * 72)
    print("FIGURE 5 — FP32–INT8 PER-SAMPLE QUALITY DISTRIBUTION")
    print("=" * 72)

    print(f"Samples                  : {n:,}")
    print()
    print(
        f"Different output string  : "
        f"{output_divergent:,} ({output_divergent_pct:.2f}%)"
    )
    print(
        f"Same output string       : "
        f"{same_output:,} ({same_output_pct:.2f}%)"
    )
    print()
    print(
        f"Different chrF++ score   : "
        f"{score_divergent:,} ({score_divergent_pct:.2f}%)"
    )
    print(
        f"Same chrF++ score        : "
        f"{same_score:,} ({same_score_pct:.2f}%)"
    )
    print(
        f"FP32 better              : "
        f"{fp32_better:,} ({fp32_pct:.2f}%)"
    )
    print(
        f"INT8 better              : "
        f"{int8_better:,} ({int8_pct:.2f}%)"
    )
    print(
        f"Mean Δ                   : "
        f"{mean_delta:+.4f} chrF++"
    )
    print(
        f"Std. deviation           : "
        f"{std_delta:.4f}"
    )

    print("=" * 72)

    # =========================================================================
    # HISTOGRAM DATA
    # =========================================================================

    # Only non-identical chrF++ scores are plotted.
    nonzero_delta = delta[delta != 0]

    fp32_delta = nonzero_delta[
        nonzero_delta < 0
    ]

    int8_delta = nonzero_delta[
        nonzero_delta > 0
    ]

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

    # =========================================================================
    # PLOT
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    # =========================================================================
    # MAIN DEFENSE HEADLINE
    # =========================================================================

    fig.text(
        0.5,
        0.975,
        "27.09% OF SAMPLES SHOWED FP32–INT8 OUTPUT DIVERGENCE",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#174A5A",
    )

    fig.text(
        0.5,
        0.935,
        "3,660 / 13,511 samples",
        ha="center",
        va="top",
        fontsize=12,
        color="#333333",
    )

    # =========================================================================
    # HISTOGRAM
    # =========================================================================

    ax.hist(
        fp32_delta,
        bins=bins,
        alpha=0.65,
        color="#024130",
        label=f"FP32 better ({fp32_better:,})",
    )

    ax.hist(
        int8_delta,
        bins=bins,
        alpha=0.65,
        color="#0DB0C3",
        label=f"INT8 better ({int8_better:,})",
    )

    # =========================================================================
    # ZERO REFERENCE
    # =========================================================================

    ax.axvline(
        0,
        linewidth=1.4,
        linestyle="-",
        color="black",
    )

    # =========================================================================
    # MEAN REFERENCE
    # =========================================================================

    ax.axvline(
        mean_delta,
        linewidth=1.2,
        linestyle="--",
        color="#174A5A",
        alpha=0.7,
    )

    # =========================================================================
    # MEAN ANNOTATION
    # =========================================================================

    ax.text(
        0.98,
        0.08,
        f"Mean Δ = {mean_delta:+.3f} chrF++",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    # =========================================================================
    # CHART TITLE
    # =========================================================================

    ax.set_title(
        "Distribution of FP32–INT8 chrF++ Differences",
        fontsize=13,
        pad=8,
    )

    # =========================================================================
    # AXES
    # =========================================================================

    ax.set_xlabel(
        "Δ chrF++ (INT8 − FP32)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Number of Samples",
        fontsize=12,
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    ax.text(
        0.98,
        0.95,
        (
            f"CoVoST2 test set: N = {n:,}\n"
            f"Same chrF++ score: {same_score:,} ({same_score_pct:.1f}%)\n"
            f"Different chrF++ score: "
            f"{score_divergent:,} ({score_divergent_pct:.1f}%)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    # =========================================================================
    # KEY TAKEAWAY
    # =========================================================================

    fig.text(
        0.5,
        0.045,
        (
            "Key takeaway: Aggregate quality difference is small, "
            "but output divergence occurs in 27.09% of samples."
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    # =========================================================================
    # GRID
    # =========================================================================

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    # =========================================================================
    # FRAME
    # =========================================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # =========================================================================
    # LEGEND
    # =========================================================================

    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=10,
    )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.14,
        top=0.82,
    )

    # =========================================================================
    # SAVE PNG
    # =========================================================================

    png_path = (
        OUTPUT_DIR
        / "figure_05_fp32_int8_distribution.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # =========================================================================
    # SAVE PDF
    # =========================================================================

    pdf_path = (
        OUTPUT_DIR
        / "figure_05_fp32_int8_distribution.pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # OUTPUT
    # =========================================================================

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print()
    print("Figure generated successfully.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    generate_figure()
