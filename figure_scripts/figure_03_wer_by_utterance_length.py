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
# VERIFIED LENGTH-STRATIFIED RESULTS
# =============================================================================
#
# Source:
# experiments/results/whisper_mslt_vs_covost_length_analysis.csv
#
# These are the verified Whisper-base sentence-level WER results
# stratified by reference utterance length.
#
# CoVoST2 has no 31+ word utterances in the analyzed data.
# =============================================================================

RESULTS = pd.DataFrame(
    {
        "Length": [
            "1–3 words",
            "4–7 words",
            "8–15 words",
            "16–30 words",
            "31+ words",
        ],
        "MSLT": [
            71.33,
            35.68,
            27.27,
            23.06,
            21.38,
        ],
        "CoVoST2": [
            57.23,
            30.92,
            26.83,
            20.18,
            None,
        ],
    }
)


# =============================================================================
# FIGURE
# =============================================================================

# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    print("=" * 72)
    print("FIGURE 3 — WER BY UTTERANCE LENGTH")
    print("=" * 72)
    print(RESULTS.to_string(index=False))
    print()

    fig, ax = plt.subplots(figsize=(9, 6))

    x = list(range(len(RESULTS)))

    # -------------------------------------------------------------------------
    # Plot datasets
    # -------------------------------------------------------------------------

    mslt_line = ax.plot(
        x,
        RESULTS["MSLT"],
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="MSLT",
    )[0]

    covost_line = ax.plot(
        x,
        RESULTS["CoVoST2"],
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="CoVoST2",
    )[0]

    # -------------------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------------------

    # MSLT labels
    for i, value in enumerate(RESULTS["MSLT"]):
        ax.annotate(
            f"{value:.2f}%",
            xy=(i, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=mslt_line.get_color(),
        )

    # CoVoST2 labels
    for i, value in enumerate(RESULTS["CoVoST2"]):
        if pd.notna(value):
            ax.annotate(
                f"{value:.2f}%",
                xy=(i, value),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=10,
                color=covost_line.get_color(),
            )

    # -------------------------------------------------------------------------
    # Explicitly indicate missing CoVoST2 31+ result
    # -------------------------------------------------------------------------

    ax.annotate(
        "No samples",
        xy=(4, RESULTS["MSLT"].iloc[4]),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
        color=covost_line.get_color(),
    )

    # -------------------------------------------------------------------------
    # X axis
    # -------------------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        RESULTS["Length"],
        fontsize=11,
    )

    ax.set_xlabel(
        "Reference Utterance Length",
        fontsize=12,
    )

    # -------------------------------------------------------------------------
    # Y axis
    # -------------------------------------------------------------------------

    ax.set_ylabel(
        "Mean Sentence-level WER (%)",
        fontsize=12,
    )

    ax.set_ylim(15, 78)

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        "Whisper-base WER by Utterance Length",
        fontsize=15,
        pad=14,
    )

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
    # Clean frame
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    ax.legend(
        title="Dataset",
        frameon=False,
        loc="upper right",
        fontsize=10,
        title_fontsize=10,
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save PNG
    # -------------------------------------------------------------------------

    png_path = OUTPUT_DIR / "figure_03_wer_by_utterance_length.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # -------------------------------------------------------------------------
    # Save PDF
    # -------------------------------------------------------------------------

    pdf_path = OUTPUT_DIR / "figure_03_wer_by_utterance_length.pdf"

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