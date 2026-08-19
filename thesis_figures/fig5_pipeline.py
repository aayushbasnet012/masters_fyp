"""Chapter 5 figure: six-stage preprocessing pipeline flowchart."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT, ACCENT_DARK, GREY, ACCENT_LIGHT, GOLD

stages = [
    ("1. Discovery", "Enumerate EN/JA file pairs via rglob;\ntag each by partition"),
    ("2. Stratified Sampling", "Sample documents by (partition, year)\nstrata to reach the target pair count"),
    ("3. Alignment", "Pair EN/JA sentences on the shared\ntrailing sentence-index key"),
    ("4. Quality Filtering", "Apply Table 5.4 filters; document-level\nyield gate (≥50% aligned)"),
    ("5. Output Writing", "80/10/10 split, MD5 de-duplication,\nwrite MLM + concept-pair files"),
    ("6. Statistics & Data Card", "Compute dataset_stats.json and\nDATA_CARD.md"),
]

fig, ax = plt.subplots(figsize=(6.6, 9.4))
ax.set_xlim(0, 60); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor("white")

n = len(stages)
box_h, gap = 11, 5.2
total_h = n * box_h + (n - 1) * gap
y0 = 100 - (100 - total_h) / 2 - box_h

for i, (title, desc) in enumerate(stages):
    y = y0 - i * (box_h + gap)
    color = ACCENT_DARK if i % 2 == 0 else ACCENT
    b = FancyBboxPatch((5, y), 50, box_h, boxstyle="round,pad=0.35,rounding_size=1.6",
                        linewidth=1.2, edgecolor=color, facecolor=color, zorder=3)
    ax.add_patch(b)
    ax.text(30, y + box_h * 0.68, title, ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold", zorder=4)
    ax.text(30, y + box_h * 0.30, desc, ha="center", va="center", fontsize=7.6,
            color="white", zorder=4)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((30, y), (30, y - gap), arrowstyle="-|>",
                                      mutation_scale=14, color=GREY, linewidth=1.6, zorder=2))

fig.tight_layout(pad=0.3)
fig.savefig("fig5_pipeline.png", dpi=300, facecolor="white")
print("saved")
