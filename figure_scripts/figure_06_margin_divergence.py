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
    / "first_divergence_margins.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA
# =============================================================================

def load_results():

    df = pd.read_csv(INPUT_FILE)

    print("CSV columns:")
    print(df.columns.tolist())
    print()

    margin_col = "fp32_margin"

    if margin_col not in df.columns:
        raise ValueError(
            f"Expected column '{margin_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    margins = pd.to_numeric(
        df[margin_col],
        errors="coerce",
    ).dropna()

    return margins


# =============================================================================
# ANALYSIS
# =============================================================================

def calculate_bins(margins):

    bins = [
        0,
        0.25,
        0.5,
        1.0,
        2.0,
        float("inf"),
    ]

    labels = [
        "0–0.25",
        "0.25–0.5",
        "0.5–1",
        "1–2",
        ">2",
    ]

    binned = pd.cut(
        margins,
        bins=bins,
        labels=labels,
        right=False,
    )

    counts = (
        binned
        .value_counts(sort=False)
        .reindex(labels, fill_value=0)
    )

    total = counts.sum()

    percentages = counts / total * 100

    return labels, counts, percentages


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    margins = load_results()

    labels, counts, percentages = calculate_bins(margins)

    n = len(margins)

    # -------------------------------------------------------------------------
    # Main finding
    # -------------------------------------------------------------------------

    below_two = counts.iloc[:4].sum()
    below_two_pct = below_two / n * 100

    above_two = counts.iloc[4]
    above_two_pct = above_two / n * 100

    print("=" * 72)
    print("FIGURE 6 — DECODER MARGIN AT FIRST FP32–INT8 DIVERGENCE")
    print("=" * 72)
    print(f"First-divergence cases : {n:,}")
    print()

    for label, count, pct in zip(
        labels,
        counts,
        percentages,
    ):
        print(
            f"{label:>8} logits : "
            f"{count:4d} ({pct:5.1f}%)"
        )

    print()
    print(
        f"Margin < 2 logits     : "
        f"{below_two:,} / {n:,} "
        f"({below_two_pct:.1f}%)"
    )
    print(
        f"Margin >= 2 logits    : "
        f"{above_two:,} / {n:,} "
        f"({above_two_pct:.1f}%)"
    )
    print()

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9.0, 5.8)
    )

    x = list(range(len(labels)))

    # -------------------------------------------------------------------------
    # Highlight low-margin region
    # -------------------------------------------------------------------------

    ax.axvspan(
        -0.5,
        3.5,
        alpha=0.08,
        zorder=0,
    )

    # -------------------------------------------------------------------------
    # Lollipop
    # -------------------------------------------------------------------------

    # One restrained color for the main distribution.
    point_color = "#CC1C3B"

    ax.vlines(
        x,
        0,
        percentages.values,
        linewidth=2.2,
        color=point_color,
        alpha=0.85,
        zorder=2,
    )

    ax.scatter(
        x,
        percentages.values,
        s=105,
        color=point_color,
        zorder=3,
    )

    # -------------------------------------------------------------------------
    # Endpoint labels
    # -------------------------------------------------------------------------

    for xpos, pct, count in zip(
        x,
        percentages.values,
        counts.values,
    ):

        ax.annotate(
            f"{pct:.1f}%",
            xy=(xpos, pct),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        ax.annotate(
            f"n={count:,}",
            xy=(xpos, pct),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            alpha=0.70,
        )

    # -------------------------------------------------------------------------
    # Main finding annotation
    # -------------------------------------------------------------------------

    ax.text(
        1.5,
        max(percentages.values) * 0.78,
        f"{below_two:,} / {n:,} cases ({below_two_pct:.1f}%)\n"
        "occur below a 2-logit margin",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
    )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    ax.set_xlabel(
        "FP32 Top-1/Top-2 Decoding Margin (logits)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Share of First-Divergence Cases (%)",
        fontsize=12,
    )

    ax.set_title(
        "Decoder Margin at First FP32–INT8 Divergence",
        fontsize=15,
        pad=14,
    )

    # -------------------------------------------------------------------------
    # Y-axis
    # -------------------------------------------------------------------------

    upper_limit = max(percentages.values) * 1.28

    ax.set_ylim(
        0,
        upper_limit,
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Clean frame
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Dataset information
    # -------------------------------------------------------------------------

    ax.text(
        0.98,
        0.96,
        f"First-divergence cases: N = {n:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
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
        / "figure_06_margin_divergence.png"
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
        / "figure_06_margin_divergence.pdf"
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    generate_figure()