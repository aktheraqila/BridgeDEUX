from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# VERIFIED MSLT TEST RESULTS
# =============================================================================
#
# Source:
#   experiments/score_mslt.py
#
# All models:
#   N = 2,275
#   Same canonical MSLT evaluation
#   T2 reference
#
# =============================================================================

RESULTS = pd.DataFrame(
    {
        "Model": [
            "Parakeet",
            "Whisper-base (W0)",
            "W1",
            "W2",
        ],
        "WER": [
            17.19,
            25.23,
            26.74,
            28.60,
        ],
    }
)


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    print("=" * 72)
    print("FIGURE 1 — MSLT TEST WER")
    print("=" * 72)

    print(RESULTS.to_string(index=False))
    print()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    bars = ax.bar(
        RESULTS["Model"],
        RESULTS["WER"],
        width=0.62,
    )

    # -------------------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------------------

    for bar, value in zip(bars, RESULTS["WER"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    ax.set_ylabel(
        "Word Error Rate (WER, %)",
        fontsize=12,
    )

    ax.set_xlabel(
        "ASR Model",
        fontsize=12,
    )

    ax.set_title(
        "MSLT Test-Set ASR Performance",
        fontsize=14,
        pad=12,
    )

    ax.set_ylim(0, 32)

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save PNG
    # -------------------------------------------------------------------------

    png_path = OUTPUT_DIR / "figure_01_asr_wer.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # -------------------------------------------------------------------------
    # Save PDF
    # -------------------------------------------------------------------------

    pdf_path = OUTPUT_DIR / "figure_01_asr_wer.pdf"

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