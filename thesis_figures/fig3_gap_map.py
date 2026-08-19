"""Chapter 4.8 figure: research-gap map positioning MINT-Pat against prior work."""
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT, ACCENT_DARK, GREY, EN_COLOR, ACCENT_LIGHT

fig, ax = plt.subplots(figsize=(8.6, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
fig.patch.set_facecolor("white")

# quadrant shading: top-right = the gap
ax.axvspan(5.5, 10, ymin=0.55, ymax=1.0, color=ACCENT_LIGHT, zorder=0)
ax.text(5.8, 9.7, "Gap MINT-Pat\naddresses", ha="left", va="top", fontsize=8.5,
        color=ACCENT_DARK, style="italic", fontweight="bold")

points = [
    ("Muller et al. (2021)\nlayer-wise multilingual analysis", 4.6, 6.3, GREY, -0.6, "bottom"),
    ("Brinkmann/Andrylie (2025)\nSAE multilingual interpretability", 4.3, 8.4, GREY, 0.55, "bottom"),
    ("Casper et al. (2024)\nwhite-box audit argument (programmatic)", 2.6, 7.2, GREY, 0.55, "bottom"),
    ("EPO deployment\nEP-AutoCla / AI-PreSearch", 7.6, 2.9, EN_COLOR, 0.55, "bottom"),
    ("USPTO deployment\nAI similarity search", 8.8, 1.2, EN_COLOR, -0.75, "top"),
    ("WIPO Translate\nneural MT for patents", 9.3, 4.0, EN_COLOR, 0.55, "bottom"),
    ("MINT-Pat (this thesis)", 8.9, 8.2, ACCENT_DARK, 0.55, "bottom"),
]

for label, x, y, color, dy, va in points:
    is_mint = "MINT-Pat" in label
    ax.scatter([x], [y], s=260 if is_mint else 140,
               marker="*" if is_mint else "o",
               color=color, edgecolor="white", linewidth=1.0, zorder=5)
    ax.annotate(label, (x, y), xytext=(x, y + dy), ha="center",
                va=va, fontsize=7.6, color=color,
                fontweight="bold" if is_mint else "normal", zorder=6)

ax.axhline(5.5, color="#BBBBBB", linewidth=0.8, zorder=1)
ax.axvline(5.5, color="#BBBBBB", linewidth=0.8, zorder=1)

ax.set_xlabel("Domain-specificity & cross-lingual scope  →", fontsize=10, color=GREY)
ax.set_ylabel("Mechanistic depth of evidence  →", fontsize=10, color=GREY)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color("#CCCCCC")

ax.set_title("Where prior work sits relative to MINT-Pat", fontsize=12,
             fontweight="bold", color=ACCENT_DARK, pad=14)

fig.tight_layout(pad=0.6)
fig.savefig("fig3_gap_map.png", dpi=300, facecolor="white")
print("saved")
