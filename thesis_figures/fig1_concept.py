"""Chapter 1 concept figure: the open question MINT-Pat investigates."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT, ACCENT_DARK, GREY, EN_COLOR, JA_COLOR, ACCENT_LIGHT

fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, color, text, tc="white", fs=9, dashed=False, ec=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                        linewidth=1.3, edgecolor=ec or color, facecolor=color,
                        linestyle="dashed" if dashed else "solid", zorder=3)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold", zorder=4)

def arrow(x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                  color=color, linewidth=1.6, zorder=2))

box(2, 28, 24, 8, EN_COLOR, "EN: “prior art”\n(patent claim text)", fs=8.5)
box(2, 8, 24, 8, JA_COLOR, "JA: “先行技術”\n(patent claim text)", fs=8.5)
arrow(26, 33, 39, 26)
arrow(26, 13, 39, 20)

box(39, 18, 22, 10, ACCENT_DARK, "Domain-Adapted\nMultilingual\nTransformer", fs=8.5)
arrow(61, 23, 70, 23)

ax.text(74, 23, "?", ha="center", va="center", fontsize=30, color=GREY, fontweight="bold", zorder=4)

arrow(78, 25, 88, 32)
arrow(78, 20, 88, 12)

box(66, 30, 32, 9, ACCENT_LIGHT, "Shared concept circuit\n(same representation)",
    tc=ACCENT_DARK, fs=8, dashed=True, ec=ACCENT)
box(66, 6, 32, 9, ACCENT_LIGHT, "Separate, language-specific\ncircuits (surface pattern only)",
    tc=GREY, fs=8, dashed=True, ec=GREY)

ax.text(50, 50, "Do equivalent EN/JA patent concepts activate\nthe same internal circuits, or only look aligned?",
        ha="center", va="top", fontsize=10.5, color=ACCENT_DARK, fontweight="bold")

fig.tight_layout(pad=0.4)
fig.savefig("fig1_concept.png", dpi=300, facecolor="white")
print("saved fig1_concept.png")
