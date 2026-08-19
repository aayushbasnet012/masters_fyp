"""Chapter 7 figure: layer-wise CKA similarity heatmap (Table 7.1 data)."""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT_DARK, GREY

layers = ["Emb"] + [f"L{i:02d}" for i in range(1, 13)]
cka = [0.254, 0.256, 0.267, 0.282, 0.294, 0.308, 0.322, 0.333, 0.342, 0.350, 0.330, 0.323, 0.393]
data = np.array(cka).reshape(1, -1)

fig, ax = plt.subplots(figsize=(9.6, 2.6))
im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.20, vmax=0.42)

ax.set_xticks(range(len(layers)))
ax.set_xticklabels(layers, fontsize=9)
ax.set_yticks([])
ax.set_title("Layer-Wise CKA Similarity (Matched EN–JA Concept Pairs, n=597)",
              fontsize=11.5, fontweight="bold", color=ACCENT_DARK, pad=10)

for i, v in enumerate(cka):
    txt_color = "white" if v > 0.33 else GREY
    label = f"{v:.3f}" + ("\n(peak)" if v == max(cka) else "")
    ax.text(i, 0, label, ha="center", va="center", fontsize=8.3, color=txt_color, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
cbar.set_label("CKA", fontsize=8.5, color=GREY)
cbar.ax.tick_params(labelsize=7.5)

for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout(pad=0.5)
fig.savefig("fig7_heatmap.png", dpi=300, facecolor="white")
print("saved")
