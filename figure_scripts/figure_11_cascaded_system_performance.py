"""
BridgeDEUX — Figure 11
Cascaded System Performance

Visualization:
    Paired-dot / dumbbell plot, inspired by presentation-style
    comparison graphics rather than a conventional bar chart or table.

The figure emphasizes the FP32 -> INT8 change in cascaded latency
for each model × dataset × ARM architecture configuration.

Outputs:
    figures/figure_11_cascaded_latency_dumbbell.png
    figures/figure_11_cascaded_latency_dumbbell.pdf

Data reproduced from the supplied thesis table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =====================================================================
# PROJECT PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# =====================================================================
# DATA FROM SUPPLIED TABLE
# =====================================================================

data = [
    {
        "model": "Whisper-base",
        "dataset": "CoVoST2",
        "arch": "ARMv8",
        "fp32": 253249.54,
        "int8": 252915.55,
    },
    {
        "model": "Whisper-base",
        "dataset": "CoVoST2",
        "arch": "ARMv9",
        "fp32": 50974.32,
        "int8": 50860.04,
    },
    {
        "model": "Whisper-base",
        "dataset": "MSLT",
        "arch": "ARMv8",
        "fp32": 217027.57,
        "int8": 216667.26,
    },
    {
        "model": "Whisper-base",
        "dataset": "MSLT",
        "arch": "ARMv9",
        "fp32": 49335.28,
        "int8": 49214.33,
    },
    {
        "model": "Student (W1)",
        "dataset": "CoVoST2",
        "arch": "ARMv8",
        "fp32": 220739.91,
        "int8": 220441.83,
    },
    {
        "model": "Student (W1)",
        "dataset": "CoVoST2",
        "arch": "ARMv9",
        "fp32": 51200.78,
        "int8": 51105.93,
    },
    {
        "model": "Student (W1)",
        "dataset": "MSLT",
        "arch": "ARMv8",
        "fp32": 227696.12,
        "int8": 227351.61,
    },
    {
        "model": "Student (W1)",
        "dataset": "MSLT",
        "arch": "ARMv9",
        "fp32": 50126.14,
        "int8": 49998.59,
    },
]


# =====================================================================
# CALCULATE LATENCY CHANGES
# =====================================================================

for item in data:
    item["change"] = item["int8"] - item["fp32"]
    item["reduction"] = item["fp32"] - item["int8"]
    item["reduction_pct"] = (
        item["reduction"] / item["fp32"] * 100
    )


# =====================================================================
# VISUAL STYLE
# =====================================================================

NAVY = "#174A5A"
TEAL = "#2F7180"
LIGHT_TEAL = "#78AAB5"

TEXT = "#17343D"
MUTED = "#60777D"
GRID = "#D3DEE1"
BG = "#F4F7F8"
WHITE = "#FFFFFF"

FP32_COLOR = "#174A5A"
INT8_COLOR = "#78AAB5"


# =====================================================================
# FIGURE
# =====================================================================

fig, ax = plt.subplots(figsize=(14, 8))

fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# =====================================================================
# TITLE
# =====================================================================

ax.text(
    0.02,
    1.075,
    "Cascaded Latency: FP32 vs INT8",
    transform=ax.transAxes,
    fontsize=25,
    fontweight="bold",
    color=NAVY,
    va="top",
)

ax.text(
    0.02,
    1.025,
    "INT8 produces only a small latency reduction across the tested "
    "models, datasets, and ARM architectures",
    transform=ax.transAxes,
    fontsize=12.5,
    color=TEAL,
    va="top",
)


# =====================================================================
# Y POSITIONS
# =====================================================================

# Order is intentionally grouped by model.
y_positions = list(range(len(data) - 1, -1, -1))


# =====================================================================
# MAIN DUMBBELL PLOT
# =====================================================================

for item, y in zip(data, y_positions):

    fp32 = item["fp32"]
    int8 = item["int8"]

    # Connecting segment.
    ax.plot(
        [fp32, int8],
        [y, y],
        linewidth=3,
        color=LIGHT_TEAL,
        solid_capstyle="round",
        zorder=2,
    )

    # FP32 point.
    ax.scatter(
        fp32,
        y,
        s=95,
        color=FP32_COLOR,
        edgecolor=WHITE,
        linewidth=1.5,
        zorder=4,
    )

    # INT8 point.
    ax.scatter(
        int8,
        y,
        s=95,
        color=INT8_COLOR,
        edgecolor=WHITE,
        linewidth=1.5,
        zorder=4,
    )


# =====================================================================
# Y LABELS
# =====================================================================

y_labels = []

for item in data:
    y_labels.append(
        f"{item['model']}  ·  "
        f"{item['dataset']}  ·  "
        f"{item['arch']}"
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(
    y_labels[::-1],
    fontsize=10.5,
    color=TEXT,
)


# =====================================================================
# DIRECT VALUE LABELS
# =====================================================================

# Use a small horizontal offset so values remain readable.
x_min = min(item["int8"] for item in data)
x_max = max(item["fp32"] for item in data)
x_range = x_max - x_min

offset = x_range * 0.012

for item, y in zip(data, y_positions):

    fp32 = item["fp32"]
    int8 = item["int8"]

    # Only label the two endpoints.
    ax.text(
        fp32 - offset,
        y + 0.18,
        f"{fp32:,.0f}",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=FP32_COLOR,
    )

    ax.text(
        int8 + offset,
        y - 0.18,
        f"{int8:,.0f}",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT,
    )


# =====================================================================
# MODEL GROUP SEPARATOR
# =====================================================================

# Separator between Whisper-base and Student (W1).
ax.axhline(
    3.5,
    color=TEAL,
    linewidth=1.8,
    xmin=0.0,
    xmax=1.0,
)


# =====================================================================
# MODEL GROUP LABELS
# =====================================================================

ax.text(
    0.0,
    0.985,
    "WHISPER-BASE",
    transform=ax.transAxes,
    fontsize=9.5,
    fontweight="bold",
    color=MUTED,
    va="bottom",
)

ax.text(
    0.0,
    0.485,
    "STUDENT (W1)",
    transform=ax.transAxes,
    fontsize=9.5,
    fontweight="bold",
    color=MUTED,
    va="bottom",
)


# =====================================================================
# AXES
# =====================================================================

ax.set_xlabel(
    "Total cascaded latency (ms)",
    fontsize=11.5,
    color=TEXT,
    labelpad=12,
)

ax.set_xlim(
    x_min - x_range * 0.055,
    x_max + x_range * 0.055,
)

ax.set_ylim(
    -0.75,
    len(data) - 0.25,
)


# =====================================================================
# GRID
# =====================================================================

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    color=GRID,
    alpha=0.8,
)

ax.set_axisbelow(True)


# =====================================================================
# SPINES
# =====================================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.spines["bottom"].set_color(GRID)

ax.tick_params(
    axis="y",
    length=0,
)

ax.tick_params(
    axis="x",
    colors=MUTED,
    labelsize=9.5,
)


# =====================================================================
# LEGEND
# =====================================================================

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=FP32_COLOR,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        markersize=8,
        label="FP32",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=INT8_COLOR,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        markersize=8,
        label="INT8",
    ),
]

ax.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=False,
    fontsize=10,
    ncol=2,
)


# =====================================================================
# KEY MESSAGE
# =====================================================================

ax.text(
    0.02,
    -0.115,
    "KEY RESULT",
    transform=ax.transAxes,
    fontsize=10,
    fontweight="bold",
    color=NAVY,
)

ax.text(
    0.02,
    -0.16,
    "The FP32 and INT8 points remain close in every tested configuration.",
    transform=ax.transAxes,
    fontsize=10.5,
    color=MUTED,
)


# =====================================================================
# MEMORY CALLOUT
# =====================================================================

ax.text(
    0.98,
    -0.115,
    "MOBILE MEMORY",
    transform=ax.transAxes,
    fontsize=10,
    fontweight="bold",
    color=NAVY,
    ha="right",
)

ax.text(
    0.98,
    -0.16,
    "≈71% saving on ARMv8  ·  ≈83% on ARMv9",
    transform=ax.transAxes,
    fontsize=10.5,
    color=MUTED,
    ha="right",
)


# =====================================================================
# SAVE
# =====================================================================

png_path = FIGURES / "figure_11_cascaded_latency_dumbbell.png"
pdf_path = FIGURES / "figure_11_cascaded_latency_dumbbell.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

plt.close(fig)


# =====================================================================
# CONSOLE VERIFICATION
# =====================================================================

print("=" * 72)
print("BRIDGEDEUX — FIGURE 11")
print("Cascaded Latency: FP32 vs INT8")
print("=" * 72)

for item in data:
    print(
        f"{item['model']:15s} | "
        f"{item['dataset']:8s} | "
        f"{item['arch']:5s} | "
        f"FP32 {item['fp32']:,.2f} ms | "
        f"INT8 {item['int8']:,.2f} ms | "
        f"Δ {item['change']:,.2f} ms"
    )

print()
print("Saved:")
print(f"  {png_path}")
print(f"  {pdf_path}")
print("=" * 72)
