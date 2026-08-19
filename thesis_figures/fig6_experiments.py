"""Chapter 6.3 figure: six interpretability experiment modules across the 13 model layers."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT, ACCENT_DARK, GREY, ACCENT_LIGHT, GOLD, EN_COLOR, JA_COLOR

fig, ax = plt.subplots(figsize=(9.6, 7.6))
ax.set_xlim(-2, 100); ax.set_ylim(0, 85); ax.axis("off")
fig.patch.set_facecolor("white")

layers = ["Emb"] + [f"L{i}" for i in range(1, 13)]
n = len(layers)
x0, x1 = 14, 96
xs = [x0 + i * (x1 - x0) / (n - 1) for i in range(n)]

axis_y = 78
ax.plot([x0, x1], [axis_y, axis_y], color=GREY, linewidth=1.4, zorder=2)
for x, lab in zip(xs, layers):
    ax.plot([x, x], [axis_y - 0.8, axis_y + 0.8], color=GREY, linewidth=1.2, zorder=2)
    ax.text(x, axis_y + 2.2, lab, ha="center", va="bottom", fontsize=7.2, color=GREY)
ax.text((x0 + x1) / 2, axis_y + 6.5, "Transformer layer depth", ha="center", fontsize=9.5,
        color=ACCENT_DARK, fontweight="bold")

modules = [
    ("Module 1 – CKA", None, ACCENT_DARK, "full"),
    ("Module 2 – Linear Probe", None, ACCENT, "full"),
    ("Module 4 – Retrieval", None, GOLD, "full"),
    ("Module 3 – SAE", [6, 9, 11], EN_COLOR, "points"),
    ("Module 5 – RAF", [3, 6, 9, 11], JA_COLOR, "points"),
    ("Module 7 – Causal Patching", [3, 6, 9, 11], GREY, "points"),
]

row_gap = 8.6
top_row_y = axis_y - 10
label_x = 11
for i, (name, idxs, color, kind) in enumerate(modules):
    y = top_row_y - i * row_gap
    ax.text(label_x, y, name, ha="right", va="center", fontsize=8.8, color=color, fontweight="bold")
    if kind == "full":
        ax.plot([xs[0], xs[-1]], [y, y], color=color, linewidth=3.2, alpha=0.85,
                 solid_capstyle="round", zorder=3)
        for li in (0, 12):
            ax.scatter([xs[li]], [y], s=30, color=color, zorder=4)
    else:
        pts_x = [xs[li] for li in idxs]
        ax.plot([pts_x[0], pts_x[-1]], [y, y], color=color, linewidth=1.2, linestyle="dotted", zorder=2)
        ax.scatter(pts_x, [y] * len(pts_x), s=65, color=color, zorder=4, marker="D")

last_y = top_row_y - (len(modules) - 1) * row_gap
ax.text((x0 + x1) / 2, top_row_y + row_gap - 1.5,
        "Findings-producing modules (span the full layer sweep, or mark specific probe layers)",
        ha="center", fontsize=8, color=GREY, style="italic")

# Module 6 box, comfortably below the last row
by = last_y - 8
box = FancyBboxPatch((10, by - 5), 78, 8.5, boxstyle="round,pad=0.35,rounding_size=1.6",
                      linewidth=1.2, edgecolor=GREY, facecolor=ACCENT_LIGHT, zorder=3)
ax.add_patch(box)
ax.text(49, by - 0.8,
        "Module 6 – Concept Vocabulary Coverage  (data validation, not a findings module;\n"
        "checks the 597-pair curated evaluation set feeding Modules 1, 2, 3, 4, 5 & 7)",
        ha="center", va="center", fontsize=7.8, color=GREY, fontweight="bold")

fig.tight_layout(pad=0.4)
fig.savefig("fig6_experiments.png", dpi=300, facecolor="white")
print("saved")
