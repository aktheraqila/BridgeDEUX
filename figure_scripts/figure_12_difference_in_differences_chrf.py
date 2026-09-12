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

conditions = [
    "Clean input",
    "ASR input",
]

# chrF++ INT8 effect = INT8 − FP32
clean_effect = -0.062
asr_effect = -0.229

effects = np.array([
    clean_effect,
    asr_effect,
])

# Difference-in-Differences
did = asr_effect - clean_effect


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print("=" * 72)
print("FIGURE — DIFFERENCE-IN-DIFFERENCES: chrF++")
print("=" * 72)

print(f"Clean INT8 effect : {clean_effect:+.3f} chrF++")
print(f"ASR INT8 effect   : {asr_effect:+.3f} chrF++")
print(f"DiD               : {did:+.3f} chrF++")
print()


# =============================================================================
# FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

x = np.arange(len(conditions))

point_color = "#174A5A"


# =============================================================================
# ZERO REFERENCE
# =============================================================================

ax.axhline(
    0,
    linewidth=1.2,
    color="black",
    linestyle="-",
    alpha=0.75,
    zorder=1,
)


# =============================================================================
# EFFECT LINE
# =============================================================================

ax.plot(
    x,
    effects,
    linewidth=2.5,
    color=point_color,
    marker="o",
    markersize=9,
    zorder=3,
)


# =============================================================================
# VALUE LABELS
# =============================================================================

ax.annotate(
    f"{clean_effect:+.3f}",
    xy=(x[0], clean_effect),
    xytext=(0, 12),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=11,
    color=point_color,
)

ax.annotate(
    f"{asr_effect:+.3f}",
    xy=(x[1], asr_effect),
    xytext=(0, -14),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=11,
    color=point_color,
)


# =============================================================================
# DiD ANNOTATION
# =============================================================================

ax.annotate(
    f"DiD = {did:+.3f} chrF++",
    xy=(0.5, np.mean(effects)),
    xytext=(0, 45),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0,
        color="black",
    ),
)


# =============================================================================
# AXES
# =============================================================================

ax.set_xticks(x)

ax.set_xticklabels(
    conditions,
    fontsize=11,
)

ax.set_xlabel(
    "Input Condition",
    fontsize=12,
)

ax.set_ylabel(
    "INT8 Effect (chrF++, INT8 − FP32)",
    fontsize=12,
)


# =============================================================================
# TITLE
# =============================================================================

ax.set_title(
    "Difference-in-Differences: INT8 Effect Under Clean vs ASR Input",
    fontsize=15,
    pad=14,
)


# =============================================================================
# Y-AXIS
# =============================================================================

ax.set_ylim(
    -0.30,
    0.06,
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
# DATASET INFORMATION
# =============================================================================

ax.text(
    0.98,
    0.96,
    "CoVoST2 test set: N = 13,511",
    transform=ax.transAxes,
    ha="right",
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
    / "figure_12_difference_in_differences_chrf.png"
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
    / "figure_12_difference_in_differences_chrf.pdf"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)


# =============================================================================
# CLOSE
# =============================================================================

plt.close(fig)


# =============================================================================
# OUTPUT
# =============================================================================

print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
print()
print("Figure generated successfully.")