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
# VERIFIED RESULTS
# =============================================================================

RESULTS = pd.DataFrame(
    {
        "Dataset": ["MSLT", "CoVoST2"],
        "Corpus_WER": [26.49, 27.96],
        "Sentence_WER": [46.74, 31.71],
    }
)


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    print("=" * 72)
    print("FIGURE 2 — WHISPER-BASE WER ACROSS AGGREGATION LEVELS")
    print("=" * 72)
    print(RESULTS.to_string(index=False))
    print()

    fig, ax = plt.subplots(figsize=(9, 6))

    x = [0, 1]

    # -------------------------------------------------------------------------
    # Plot each dataset
    # -------------------------------------------------------------------------

    for _, row in RESULTS.iterrows():

        line = ax.plot(
            x,
            [row["Corpus_WER"], row["Sentence_WER"]],
            marker="o",
            markersize=10,
            linewidth=2.5,
            label=row["Dataset"],
        )[0]

        color = line.get_color()

        corpus = row["Corpus_WER"]
        sentence = row["Sentence_WER"]

        # ---------------------------------------------------------------------
        # Corpus-level endpoint label
        # ---------------------------------------------------------------------

        if row["Dataset"] == "MSLT":

            # Put 26.49% BELOW the blue point to avoid collision
            ax.annotate(
                f"{corpus:.2f}%",
                xy=(x[0], corpus),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=11,
                color=color,
            )

        else:

            # CoVoST2 label remains ABOVE its point
            ax.annotate(
                f"{corpus:.2f}%",
                xy=(x[0], corpus),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color=color,
            )

        # ---------------------------------------------------------------------
        # Sentence-level endpoint label
        # ---------------------------------------------------------------------

        ax.annotate(
            f"{sentence:.2f}%",
            xy=(x[1], sentence),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            color=color,
        )

        # ---------------------------------------------------------------------
        # Percentage-point change
        # ---------------------------------------------------------------------

       

    # -------------------------------------------------------------------------
    # X axis
    # -------------------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "Corpus-level WER",
            "Mean sentence-level WER",
        ],
        fontsize=11,
    )

    # -------------------------------------------------------------------------
    # Y axis
    # -------------------------------------------------------------------------

    ax.set_ylabel(
        "Word Error Rate (WER, %)",
        fontsize=12,
    )

    ax.set_ylim(22, 51)

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        "Whisper-base WER: Corpus vs. Sentence-level Aggregation",
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
        loc="upper left",
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

    png_path = OUTPUT_DIR / "figure_02_dataset_wer_comparison.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # -------------------------------------------------------------------------
    # Save PDF
    # -------------------------------------------------------------------------

    pdf_path = OUTPUT_DIR / "figure_02_dataset_wer_comparison.pdf"

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