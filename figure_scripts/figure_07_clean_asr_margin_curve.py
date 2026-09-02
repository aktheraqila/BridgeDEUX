from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_FILE = (
    PROJECT_ROOT
    / "analysis"
    / "level2_margin_tokens_clean.parquet"
)

ASR_FILE = (
    PROJECT_ROOT
    / "analysis"
    / "level2_margin_tokens_asr.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

MARGIN_BINS = [
    0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    np.inf,
]

MARGIN_LABELS = [
    "0–0.25",
    "0.25–0.5",
    "0.5–1",
    "1–2",
    "2–4",
    "4–8",
    ">8",
]


# =============================================================================
# DATA
# =============================================================================

def load_condition(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} data not found:\n{path}"
        )

    df = pd.read_parquet(
        path,
        columns=[
            "margin_fp32",
            "is_flip",
        ],
    )

    df["margin_fp32"] = pd.to_numeric(
        df["margin_fp32"],
        errors="coerce",
    )

    df["is_flip"] = pd.to_numeric(
        df["is_flip"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "margin_fp32",
            "is_flip",
        ]
    ).copy()

    return df


# =============================================================================
# CALCULATION
# =============================================================================

def calculate_curve(df):

    df = df.copy()

    df["margin_bin"] = pd.cut(
        df["margin_fp32"],
        bins=MARGIN_BINS,
        labels=MARGIN_LABELS,
        right=False,
    )

    grouped = (
        df.groupby(
            "margin_bin",
            observed=False,
        )
        .agg(
            flip_probability=("is_flip", "mean"),
            samples=("is_flip", "size"),
            flips=("is_flip", "sum"),
        )
        .reindex(MARGIN_LABELS)
    )

    grouped["flip_probability"] *= 100

    return grouped


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure():

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    clean = load_condition(
        CLEAN_FILE,
        "Clean",
    )

    asr = load_condition(
        ASR_FILE,
        "ASR",
    )

    clean_curve = calculate_curve(clean)
    asr_curve = calculate_curve(asr)

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print("=" * 72)
    print("FIGURE 7 — CLEAN VS ASR MARGIN CURVE")
    print("=" * 72)

    print(
        f"Clean token positions : {len(clean):,}"
    )

    print(
        f"ASR token positions   : {len(asr):,}"
    )

    print()

    print("CLEAN")
    print(clean_curve.to_string())

    print()

    print("ASR")
    print(asr_curve.to_string())

    print()

    # =============================================================================
    # PLOT
    # =============================================================================

    fig, ax = plt.subplots(
        figsize=(9.5, 6.0)
    )

    x = np.arange(len(MARGIN_LABELS))

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------

    clean_color = "#355C7D"
    asr_color = "#2A7F62"

    # -------------------------------------------------------------------------
    # Lines
    # -------------------------------------------------------------------------

    ax.plot(
        x,
        clean_curve["flip_probability"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=clean_color,
        label="Clean input",
    )

    ax.plot(
        x,
        asr_curve["flip_probability"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=asr_color,
        label="ASR input",
    )

    # =============================================================================
    # VALUE LABELS
    # =============================================================================

    clean_values = clean_curve["flip_probability"].values
    asr_values = asr_curve["flip_probability"].values

    # -------------------------------------------------------------------------
    # Clean / BLUE labels
    #
    # Requested positioning:
    #   35.1%  -> further left
    #   10.40% -> lower
    #   1.54%  -> further left + lower
    # -------------------------------------------------------------------------

    clean_offsets = [
        (-28, -10),   # 0–0.25
        (0, -20),     # 0.25–0.5
        (-20, 3),   # 0.5–1
        (0, -10),     # 1–2
        (0, -10),     # 2–4
        (0, -10),     # 4–8
        (0, -10),     # >8
    ]

    for i, value in enumerate(clean_values):

        if pd.notna(value) and value >= 1.0:

            dx, dy = clean_offsets[i]

            ax.annotate(
                f"{value:.2f}%",
                xy=(x[i], value),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=9,
                color=clean_color,
            )

    # -------------------------------------------------------------------------
    # ASR / GREEN labels
    #
    # Requested positioning:
    #   37.63% -> above
    #   14.50% -> further right
    #   3.34%  -> further right
    # -------------------------------------------------------------------------

    asr_offsets = [
        (0, 10),       # 0–0.25
        (22, 10),      # 0.25–0.5
        (22, 10),      # 0.5–1
        (0, 10),       # 1–2
        (0, 10),       # 2–4
        (0, 10),       # 4–8
        (0, 10),       # >8
    ]

    for i, value in enumerate(asr_values):

        if pd.notna(value) and value >= 1.0:

            dx, dy = asr_offsets[i]

            ax.annotate(
                f"{value:.2f}%",
                xy=(x[i], value),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=asr_color,
            )

    # =============================================================================
    # LOW-MARGIN REGION
    # =============================================================================

    ax.axvspan(
        -0.5,
        2.5,
        alpha=0.035,
        zorder=0,
    )

    ax.text(
        1.0,
        0.97,
        "Low-margin region",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        alpha=0.65,
    )

    # =============================================================================
    # AXES
    # =============================================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        MARGIN_LABELS,
        fontsize=10,
    )

    ax.set_xlabel(
        "FP32 Top-1/Top-2 Decoding Margin (logits)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Token Flip Probability (%)",
        fontsize=12,
    )

    ax.set_title(
        "Token Flip Probability Declines with Increasing Decoding Margin",
        fontsize=15,
        pad=14,
    )

    # =============================================================================
    # Y AXIS
    # =============================================================================

    ymax = max(
        clean_curve["flip_probability"].max(),
        asr_curve["flip_probability"].max(),
    )

    ax.set_ylim(
        0,
        ymax * 1.28,
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
    # METADATA
    # =============================================================================

    ax.text(
        0.02,
        0.96,
        (
            f"Clean: N = {len(clean):,}\n"
            f"ASR: N = {len(asr):,}"
        ),
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
        / "figure_07_clean_asr_margin_curve.png"
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
        / "figure_07_clean_asr_margin_curve.pdf"
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