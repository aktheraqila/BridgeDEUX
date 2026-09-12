"""
BridgeDEUX — Figure 10
Oracle Routing Ceiling

Generates:
    figures/figure_10_oracle_routing_ceiling.png
    figures/figure_10_oracle_routing_ceiling.pdf

    figures/figure_10_oracle_win_distribution.png
    figures/figure_10_oracle_win_distribution.pdf

Source:
    scripts/00b_oracle_ceiling.py
    BridgeDEUX thesis Section 4.10 — Oracle Routing Ceiling

Frozen oracle results:
    Total samples        = 13,511
    FP32 wins            = 1,882
    INT8 wins            = 1,739
    FP32 win percentage  = 13.93%
    INT8 win percentage  = 12.87%
    Mean FP32 advantage  = 9.085694 chrF++
    Mean INT8 advantage  = 9.353302 chrF++
    Oracle ceiling       = 1.265582 chrF++

Important:
    The oracle has access to the reference translation.
    Therefore this is a theoretical upper bound, not a deployable
    production router.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# =====================================================================
# PROJECT PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# =====================================================================
# FROZEN ORACLE RESULTS
# =====================================================================

TOTAL = 13_511

FP32_WINS = 1_882
INT8_WINS = 1_739

FP32_WIN_PCT = 13.93
INT8_WIN_PCT = 12.87

FP32_ADVANTAGE = 9.085694
INT8_ADVANTAGE = 9.353302

ORACLE_CEILING = 1.265582


# =====================================================================
# VISUAL STYLE
# =====================================================================

NAVY = "#174A5A"
TEAL = "#2F7180"
LIGHT_TEAL = "#DCECEF"
MID_TEAL = "#78AAB5"

TEXT = "#17343D"
MUTED = "#5E7379"

LIGHT_BG = "#F4F7F8"
WHITE = "#FFFFFF"
LINE = "#A9BEC4"

FP32_COLOR = "#2F7180"
INT8_COLOR = "#78AAB5"


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    body=None,
    facecolor=WHITE,
    edgecolor=LINE,
    title_color=TEXT,
    body_color=MUTED,
    title_size=16,
    body_size=11,
):
    """Draw a rounded information box."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )

    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height - 0.12,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        transform=ax.transAxes,
    )

    if body:
        ax.text(
            x + width / 2,
            y + height / 2 - 0.015,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=body_color,
            linespacing=1.35,
            transform=ax.transAxes,
        )

    return patch


def arrow(ax, x1, y1, x2, y2):
    """Draw a directional arrow between pipeline stages."""

    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color=TEAL,
        )
    )


# =====================================================================
# FIGURE 10A — ORACLE ROUTING CEILING
# =====================================================================

fig, ax = plt.subplots(figsize=(13.333, 7.5))

fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)

ax.axis("off")


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

ax.text(
    0.055,
    0.91,
    "ORACLE ROUTING CEILING",
    fontsize=25,
    fontweight="bold",
    color=NAVY,
    transform=ax.transAxes,
)

ax.text(
    0.055,
    0.855,
    "What a perfect sentence-level precision selector could gain",
    fontsize=15,
    fontstyle="italic",
    color=TEAL,
    transform=ax.transAxes,
)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

rounded_box(
    ax,
    0.055,
    0.42,
    0.235,
    0.245,
    "1  COMPARE",
    f"{TOTAL:,} matched samples\n\n"
    "FP32 output\n"
    "vs.\n"
    "INT8 output",
)


rounded_box(
    ax,
    0.382,
    0.42,
    0.235,
    0.245,
    "2  CHOOSE",
    "Reference-aware oracle\n\n"
    "Select the output with the\n"
    "higher sentence-level chrF++",
)


rounded_box(
    ax,
    0.709,
    0.42,
    0.235,
    0.245,
    "3  MEASURE CEILING",
    "Compare oracle performance\n"
    "against the static baseline",
    facecolor=LIGHT_TEAL,
    edgecolor=MID_TEAL,
)


arrow(ax, 0.29, 0.542, 0.378, 0.542)

arrow(ax, 0.617, 0.542, 0.705, 0.542)


# ---------------------------------------------------------------------
# Main oracle result
# ---------------------------------------------------------------------

ax.text(
    0.825,
    0.285,
    f"+{ORACLE_CEILING:.2f}",
    ha="center",
    va="center",
    fontsize=30,
    fontweight="semibold",
    color=NAVY,
    transform=ax.transAxes,
)

ax.text(
    0.825,
    0.225,
    "chrF++",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color=TEAL,
    transform=ax.transAxes,
)

ax.text(
    0.825,
    0.18,
    "theoretical improvement",
    ha="center",
    va="center",
    fontsize=11,
    color=MUTED,
    transform=ax.transAxes,
)


# ---------------------------------------------------------------------
# Oracle preference counts
# ---------------------------------------------------------------------

ax.text(
    0.055,
    0.255,
    "Oracle preference across samples",
    fontsize=13,
    fontweight="bold",
    color=TEXT,
    transform=ax.transAxes,
)

ax.text(
    0.055,
    0.205,
    f"FP32 preferred: {FP32_WINS:,}  ({FP32_WIN_PCT:.2f}%)",
    fontsize=11,
    color=TEXT,
    transform=ax.transAxes,
)

ax.text(
    0.055,
    0.165,
    f"INT8 preferred: {INT8_WINS:,}  ({INT8_WIN_PCT:.2f}%)",
    fontsize=11,
    color=TEXT,
    transform=ax.transAxes,
)


# ---------------------------------------------------------------------
# Upper-bound warning
# ---------------------------------------------------------------------

boundary = FancyBboxPatch(
    (0.055, 0.055),
    0.89,
    0.065,
    boxstyle="round,pad=0.012,rounding_size=0.015",
    linewidth=1.0,
    edgecolor=LINE,
    facecolor=WHITE,
    transform=ax.transAxes,
)

ax.add_patch(boundary)

ax.text(
    0.5,
    0.087,
    "UPPER BOUND  •  Uses reference information  •  Not a validated production router",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=MUTED,
    transform=ax.transAxes,
)


# ---------------------------------------------------------------------
# Save Figure 10A
# ---------------------------------------------------------------------

fig.savefig(
    FIGURES / "figure_10_oracle_routing_ceiling.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

fig.savefig(
    FIGURES / "figure_10_oracle_routing_ceiling.pdf",
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

plt.close(fig)


# =====================================================================
# FIGURE 10B — ORACLE PREFERENCE BALANCE
# =====================================================================

fig, ax = plt.subplots(figsize=(11, 5.8))

fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)


# ---------------------------------------------------------------------
# Calculate preference shares
# ---------------------------------------------------------------------

DIVERGENT = FP32_WINS + INT8_WINS

fp32_share = FP32_WINS / DIVERGENT * 100

int8_share = INT8_WINS / DIVERGENT * 100


# =====================================================================
# HEADER
# =====================================================================

ax.text(
    0.02,
    0.94,
    "ORACLE PREFERENCE IS APPROXIMATELY BALANCED",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
    transform=ax.transAxes,
)

ax.text(
    0.02,
    0.875,
    f"{DIVERGENT:,} samples where FP32 and INT8 differ",
    fontsize=12,
    color=TEAL,
    transform=ax.transAxes,
)


# =====================================================================
# MAIN SPLIT BAR
# =====================================================================

bar_y = 0.52

bar_height = 0.18


# FP32 portion
ax.barh(
    bar_y,
    fp32_share,
    height=bar_height,
    left=0,
    color=FP32_COLOR,
    edgecolor="none",
)


# INT8 portion
ax.barh(
    bar_y,
    int8_share,
    height=bar_height,
    left=fp32_share,
    color=INT8_COLOR,
    edgecolor="none",
)


# =====================================================================
# LABELS INSIDE THE SPLIT BAR
# =====================================================================

ax.text(
    fp32_share / 2,
    bar_y,
    f"FP32\n{FP32_WINS:,}  •  {fp32_share:.1f}%",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="semibold",
    color=WHITE,
)


ax.text(
    fp32_share + int8_share / 2,
    bar_y,
    f"INT8\n{INT8_WINS:,}  •  {int8_share:.1f}%",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="semibold",
    color=WHITE,
)


# =====================================================================
# ORACLE CEILING
# =====================================================================

ax.text(
    0.5,
    0.28,
    f"+{ORACLE_CEILING:.2f} chrF++",
    ha="center",
    va="center",
    fontsize=27,
    fontweight="semibold",
    color=NAVY,
    transform=ax.transAxes,
)

ax.text(
    0.5,
    0.20,
    "theoretical improvement from perfect sentence-level selection",
    ha="center",
    va="center",
    fontsize=11,
    color=MUTED,
    transform=ax.transAxes,
)


# =====================================================================
# MEAN ADVANTAGES
# =====================================================================

ax.text(
    0.02,
    0.08,
    f"FP32 wins: +{FP32_ADVANTAGE:.3f} chrF++ mean advantage",
    fontsize=10.5,
    color=TEXT,
    transform=ax.transAxes,
)

ax.text(
    0.98,
    0.08,
    f"INT8 wins: +{INT8_ADVANTAGE:.3f} chrF++ mean advantage",
    fontsize=10.5,
    color=TEXT,
    ha="right",
    transform=ax.transAxes,
)


# =====================================================================
# FORMATTING
# =====================================================================

ax.set_xlim(0, 100)

ax.set_ylim(0, 1)

ax.set_yticks([])

ax.set_xticks([0, 25, 50, 75, 100])

ax.set_xticklabels(
    ["0%", "25%", "50%", "75%", "100%"],
    fontsize=10,
    color=MUTED,
)


ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

ax.spines["bottom"].set_color(LINE)


ax.tick_params(
    axis="x",
    length=0,
)


fig.tight_layout()


# =====================================================================
# SAVE FIGURE 10B
# =====================================================================

fig.savefig(
    FIGURES / "figure_10_oracle_win_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

fig.savefig(
    FIGURES / "figure_10_oracle_win_distribution.pdf",
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

plt.close(fig)


# =====================================================================
# VERIFICATION OUTPUT
# =====================================================================

print("=" * 72)
print("BRIDGEDEUX — FIGURE 10: ORACLE ROUTING")
print("=" * 72)

print(f"Total samples       : {TOTAL:,}")

print(
    f"FP32 wins            : {FP32_WINS:,} "
    f"({FP32_WIN_PCT:.2f}%)"
)

print(
    f"INT8 wins            : {INT8_WINS:,} "
    f"({INT8_WIN_PCT:.2f}%)"
)

print(f"FP32 mean advantage  : {FP32_ADVANTAGE:.6f} chrF++")

print(f"INT8 mean advantage  : {INT8_ADVANTAGE:.6f} chrF++")

print(f"Oracle ceiling       : +{ORACLE_CEILING:.6f} chrF++")

print()

print(f"FP32 share of wins   : {fp32_share:.2f}%")

print(f"INT8 share of wins   : {int8_share:.2f}%")

print()

print("Saved:")

print(
    "  figures/figure_10_oracle_routing_ceiling.png"
)

print(
    "  figures/figure_10_oracle_routing_ceiling.pdf"
)

print(
    "  figures/figure_10_oracle_win_distribution.png"
)

print(
    "  figures/figure_10_oracle_win_distribution.pdf"
)

print("=" * 72)