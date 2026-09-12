from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA
# =============================================================================

models = [
    "W0\nWhisper-base",
    "W1\nFull Teacher\nSupervision",
    "W2\nFiltered Teacher\nData",
]

wer = np.array([
    42.55,
    31.78,
    34.83,
])

improvement = wer[0] - wer[1]


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    print("=" * 72)
    print("FIGURE 09 — PRIMARY ASR KNOWLEDGE DISTILLATION")
    print("=" * 72)

    print(f"W0 — Whisper-base          : {wer[0]:.2f}% WER")
    print(f"W1 — Full Teacher         : {wer[1]:.2f}% WER")
    print(f"W2 — Filtered Teacher     : {wer[2]:.2f}% WER")
    print(f"W1 improvement vs W0      : {improvement:.2f} percentage points")
    print()

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5.8))

    x = np.arange(len(models))
    width = 0.38

    # Distinct colors
    colors = [
        "tab:blue",
        "tab:green",
        "tab:orange",
    ]

    bars = ax.bar(
        x,
        wer,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
    )

    # -------------------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------------------

    for bar, value in zip(bars, wer):
        ax.annotate(
            f"{value:.2f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # -------------------------------------------------------------------------
    # Parakeet teacher annotation
    # -------------------------------------------------------------------------

    teacher_x = 1.5

    ax.annotate(
        "Parakeet-TDT 0.6B v3\nTeacher",
        xy=(teacher_x, 36.5),
        xytext=(teacher_x, 47.5),
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
        ),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    # -------------------------------------------------------------------------
    # Teacher supervision bracket / connection
    # -------------------------------------------------------------------------

    ax.plot(
        [1, 2],
        [35.5, 35.5],
        linewidth=1.2,
        color="0.35",
        zorder=2,
    )

    ax.plot(
        [1, 1],
        [35.5, 33.8],
        linewidth=1.2,
        color="0.35",
        zorder=2,
    )

    ax.plot(
        [2, 2],
        [35.5, 33.8],
        linewidth=1.2,
        color="0.35",
        zorder=2,
    )

    # -------------------------------------------------------------------------
    # W1 improvement annotation
    # -------------------------------------------------------------------------

    ax.annotate(
        f"W1 improvement: −{improvement:.2f} pp",
        xy=(1, wer[1]),
        xytext=(0.35, 25.5),
        ha="center",
        va="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.0,
        ),
    )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    ax.set_title(
        "Primary ASR Knowledge Distillation: Whisper-base with Parakeet Supervision",
        fontsize=14.5,
        fontweight="bold",
        pad=16,
    )

    ax.set_ylabel(
        "Word Error Rate (WER, %)",
        fontsize=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        fontsize=10.5,
    )

    ax.set_ylim(0, 50)

    ax.tick_params(
        axis="y",
        labelsize=10,
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
        zorder=0,
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Remove unnecessary frame
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    fig.tight_layout()

        # -------------------------------------------------------------------------
    # Save PNG
    # -------------------------------------------------------------------------

    png_path = PROJECT_ROOT / "figures" / "figure_09_primary_asr_kd_wer.png"

    fig.savefig(
        str(png_path),
        dpi=300,
        bbox_inches="tight",
        format="png",
    )

    # -------------------------------------------------------------------------
    # Save PDF
    # -------------------------------------------------------------------------

    pdf_path = PROJECT_ROOT / "figures" / "figure_09_primary_asr_kd_wer.pdf"

    fig.savefig(
        str(pdf_path),
        bbox_inches="tight",
        format="pdf",
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