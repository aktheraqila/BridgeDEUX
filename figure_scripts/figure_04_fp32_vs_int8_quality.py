from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    )

    return df


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    df = load_results()

    n = len(df)

    fp32_mean = df["chrf_clean_f"].mean()
    int8_mean = df["chrf_clean_i"].mean()
    delta = int8_mean - fp32_mean

    print("=" * 72)
    print("FIGURE 4 — FP32 VS INT8 TRANSLATION QUALITY")
    print("=" * 72)
    print(f"Samples              : {n:,}")
    print(f"FP32 mean chrF++     : {fp32_mean:.4f}")
    print(f"INT8 mean chrF++     : {int8_mean:.4f}")
    print(f"INT8 - FP32          : {delta:+.4f}")
    print()

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 4.8))

    fp32_color = "tab:blue"
    int8_color = "tab:orange"

    # -------------------------------------------------------------------------
    # Connecting line
    # -------------------------------------------------------------------------

    ax.plot(
        [int8_mean, fp32_mean],
        [0, 0],
        linewidth=2.5,
        color="0.65",
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # Points
    # -------------------------------------------------------------------------

    ax.scatter(
        fp32_mean,
        0,
        s=180,
        color=fp32_color,
        zorder=3,
        label="FP32",
    )

    ax.scatter(
        int8_mean,
        0,
        s=180,
        color=int8_color,
        zorder=3,
        label="Dynamic INT8",
    )

    # -------------------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------------------

    ax.annotate(
        f"{fp32_mean:.2f}",
        xy=(fp32_mean, 0),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=fp32_color,
    )

    ax.annotate(
        f"{int8_mean:.2f}",
        xy=(int8_mean, 0),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=int8_color,
    )

    # -------------------------------------------------------------------------
    # Difference annotation
    # -------------------------------------------------------------------------

    midpoint = (fp32_mean + int8_mean) / 2

    ax.annotate(
        f"Δ = {delta:+.3f} chrF++",
        xy=(midpoint, 0),
        xytext=(0, 42),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    margin = 0.035

    ax.set_xlim(
        min(int8_mean, fp32_mean) - margin,
        max(int8_mean, fp32_mean) + margin,
    )

    ax.set_ylim(-0.55, 0.55)

    ax.set_yticks([])

    ax.set_xlabel(
        "Mean chrF++ Score",
        fontsize=12,
    )

    ax.set_title(
        "MarianMT Translation Quality: FP32 vs. Dynamic INT8",
        fontsize=15,
        pad=14,
    )

    # -------------------------------------------------------------------------
    # X-axis grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Remove unnecessary frame
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # -------------------------------------------------------------------------
    # Sample information
    # -------------------------------------------------------------------------

    ax.text(
        0.98,
        0.96,
        f"CoVoST2 test set: N = {n:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=10,
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save PNG
    # -------------------------------------------------------------------------

    png_path = (
        OUTPUT_DIR
        / "figure_04_fp32_vs_int8_quality.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # -------------------------------------------------------------------------
    # Save PDF
    # -------------------------------------------------------------------------

    pdf_path = (
        OUTPUT_DIR
        / "figure_04_fp32_vs_int8_quality.pdf"
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