"""
Title-page cover graphic for the MINT-Pat thesis.

A minimalist schematic: parallel English/Japanese patent sentences flow into
a domain-adapted multilingual Transformer, whose representations are then
probed and decomposed (mechanistic interpretability) into shared
cross-lingual concept circuits versus language-specific circuits.

Adjust COLORS / LABELS below and re-run to regenerate.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np

ACCENT = "#2E75B6"       # thesis accent blue (matches document captions/boxes)
ACCENT_DARK = "#1B4F7A"
GREY = "#595959"
LIGHT_GREY = "#EBF4FB"
EN_COLOR = "#C44E52"     # English-specific
JA_COLOR = "#55A868"     # Japanese-specific
SHARED_COLOR = "#2E75B6"  # shared cross-lingual concept

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

def rounded_box(x, y, w, h, color, text, text_color="white", fontsize=8.5, alpha=1.0, fontweight="bold"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.6",
        linewidth=1.1, edgecolor=color, facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight, zorder=4)
    return box

def arrow(x1, y1, x2, y2, color=GREY, lw=1.6, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                         color=color, linewidth=lw, zorder=2)
    ax.add_patch(a)

# --- 1. Source language boxes (left) -----------------------------------
rounded_box(2, 33, 15, 8, EN_COLOR, "English\nPatent Sentences", fontsize=8)
rounded_box(2, 9, 15, 8, JA_COLOR, "Japanese\nPatent Sentences", fontsize=8)

arrow(17, 37, 30, 28, color=EN_COLOR)
arrow(17, 13, 30, 22, color=JA_COLOR)

# --- 2. Transformer stack (middle) --------------------------------------
n_layers = 6
stack_x, stack_w = 30, 20
stack_y0, layer_h, gap = 12, 4.0, 0.9
for i in range(n_layers):
    shade = 0.35 + 0.55 * (i / (n_layers - 1))
    color = plt.cm.Blues(0.35 + 0.5 * (i / (n_layers - 1)))
    y = stack_y0 + i * (layer_h + gap)
    box = FancyBboxPatch((stack_x, y), stack_w, layer_h,
                          boxstyle="round,pad=0.15,rounding_size=0.8",
                          linewidth=0.8, edgecolor=ACCENT_DARK, facecolor=color, zorder=3)
    ax.add_patch(box)

ax.text(stack_x + stack_w / 2, stack_y0 + n_layers * (layer_h + gap) + 2.2,
        "Domain-Adapted Multilingual\nTransformer (XLM-RoBERTa)",
        ha="center", va="bottom", fontsize=8.3, color=ACCENT_DARK, fontweight="bold")

# small circuit nodes dotted on a couple of upper layers, hinting at probed features
rng = np.random.default_rng(7)
for i in [3, 4, 5]:
    y = stack_y0 + i * (layer_h + gap) + layer_h / 2
    xs = np.linspace(stack_x + 2.5, stack_x + stack_w - 2.5, 5)
    for x in xs:
        jitter = rng.uniform(-0.3, 0.3)
        ax.add_patch(Circle((x, y + jitter), 0.35, facecolor="white", edgecolor=ACCENT_DARK,
                             linewidth=0.6, zorder=4, alpha=0.9))

# --- 3. Magnifying-glass "interpretability" motif -----------------------
lens_cx, lens_cy, lens_r = 52.5, 41.5, 3.0
ax.add_patch(Circle((lens_cx, lens_cy), lens_r, facecolor="none", edgecolor=GREY, linewidth=1.8, zorder=5))
handle = FancyArrowPatch((lens_cx + lens_r * 0.7, lens_cy - lens_r * 0.7),
                          (lens_cx + lens_r * 1.6, lens_cy - lens_r * 1.6),
                          arrowstyle="-", linewidth=2.6, color=GREY, zorder=5)
ax.add_patch(handle)

# --- 4. Output: shared vs language-specific circuits (right) -----------
top_y = stack_y0 + n_layers * (layer_h + gap)
arrow(stack_x + stack_w, top_y - 5, 72, 34, color=GREY)
arrow(stack_x + stack_w, top_y - 9, 72, 14, color=GREY)

# shared concept circuit cluster
rounded_box(72, 30, 24, 9, LIGHT_GREY, "", text_color=LIGHT_GREY, alpha=1.0)
ax.add_patch(FancyBboxPatch((72, 30), 24, 9, boxstyle="round,pad=0.3,rounding_size=1.6",
                             linewidth=1.1, edgecolor=SHARED_COLOR, facecolor=LIGHT_GREY, zorder=3))
shared_pts = [(76, 34.5), (80, 36.5), (80, 32.5), (85, 34.5), (89, 34.5)]
for (x, y) in shared_pts:
    ax.add_patch(Circle((x, y), 0.55, facecolor=SHARED_COLOR, edgecolor="white", linewidth=0.6, zorder=4))
for i in range(len(shared_pts) - 1):
    x1, y1 = shared_pts[i]
    x2, y2 = shared_pts[i + 1]
    ax.plot([x1, x2], [y1, y2], color=SHARED_COLOR, linewidth=1.0, zorder=3, alpha=0.7)
ax.text(84, 27.3, "Shared Concept Circuits", ha="center", va="top", fontsize=7.8,
        color=ACCENT_DARK, fontweight="bold")

# language-specific circuit cluster
ax.add_patch(FancyBboxPatch((72, 6), 24, 9, boxstyle="round,pad=0.3,rounding_size=1.6",
                             linewidth=1.1, edgecolor=GREY, facecolor=LIGHT_GREY, zorder=3))
en_pts = [(76, 12.5), (79.5, 13.8)]
ja_pts = [(85, 9), (88.5, 10.3)]
for (x, y) in en_pts:
    ax.add_patch(Circle((x, y), 0.55, facecolor=EN_COLOR, edgecolor="white", linewidth=0.6, zorder=4))
for (x, y) in ja_pts:
    ax.add_patch(Circle((x, y), 0.55, facecolor=JA_COLOR, edgecolor="white", linewidth=0.6, zorder=4))
ax.plot([en_pts[0][0], en_pts[1][0]], [en_pts[0][1], en_pts[1][1]], color=EN_COLOR, linewidth=1.0, alpha=0.7, zorder=3)
ax.plot([ja_pts[0][0], ja_pts[1][0]], [ja_pts[0][1], ja_pts[1][1]], color=JA_COLOR, linewidth=1.0, alpha=0.7, zorder=3)
ax.text(84, 3.3, "Language-Specific Circuits", ha="center", va="top", fontsize=7.8,
        color=GREY, fontweight="bold")

fig.tight_layout(pad=0.4)
OUTPUT_FILE = "cover_image.png"
fig.savefig(OUTPUT_FILE, dpi=300, facecolor="white")
print(f"Saved cover image to {OUTPUT_FILE}")
