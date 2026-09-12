from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Figure 4.9 — MSLT Test-Set ASR Performance
# Primary PyTorch / Hugging Face KD evaluation
# ============================================================

models = ["Parakeet", "Whisper-base", "W1", "W2"]
wer = [22.76, 42.55, 31.78, 34.83]

# Output directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    models,
    wer,
    width=0.30,
    color="#176365",
)

# Value labels
for bar, value in zip(bars, wer):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

# Title and labels
ax.set_title(
    "ASR Knowledge Distillation — WER Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "ASR Model",
    fontsize=12,
    labelpad=8
)

ax.set_ylabel(
    "Word Error Rate (WER, %)",
    fontsize=12,
    labelpad=8
)

# Axis and grid
ax.set_ylim(0, 48)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

# Clean appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()

# Save directly into figures/
png_path = OUTPUT_DIR / "figure_09_teacher_student_wer.png"
pdf_path = OUTPUT_DIR / "figure_09_teacher_student_wer.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

# Do NOT display the figure
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")