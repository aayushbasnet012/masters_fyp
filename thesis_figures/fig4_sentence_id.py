"""Chapter 5 figure (fills existing Figure 5.1 slot): Sentence ID format breakdown."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT, ACCENT_DARK, GREY, EN_COLOR, JA_COLOR, GOLD

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
fig.patch.set_facecolor("white")

# main annotated example: claim_0032_826,827
parts = [("claim", EN_COLOR, "section"), ("_", None, None), ("0032", ACCENT_DARK, "paragraph (4-digit)"),
         ("_", None, None), ("826,827", GOLD, "sentence index (comma-grouped)")]

x = 8
y_code = 30
widths = {"claim": 18, "_": 4, "0032": 14, "826,827": 22}
xs = {}
cur = x
for token, color, label in parts:
    w = widths.get(token, 4)
    xs[token] = (cur, w)
    if color:
        b = FancyBboxPatch((cur, y_code), w, 7, boxstyle="round,pad=0.15,rounding_size=0.6",
                            linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.85, zorder=3)
        ax.add_patch(b)
        ax.text(cur + w/2, y_code + 3.5, token, ha="center", va="center", fontsize=11,
                color="white", fontweight="bold", family="monospace", zorder=4)
    else:
        ax.text(cur + w/2, y_code + 3.5, token, ha="center", va="center", fontsize=13,
                color=GREY, fontweight="bold", zorder=4)
    cur += w

labels = [("claim", "section:\ntitle / abstract / description /\nclaim / figref", EN_COLOR),
          ("0032", "paragraph:\nzero-padded 4-digit\nblock index", ACCENT_DARK),
          ("826,827", "sentence_index:\nglobal counter, shared EN↔JA;\ncomma list = grouped sentences", GOLD)]

for token, text, color in labels:
    bx, bw = xs[token]
    tx = bx + bw / 2
    ax.add_patch(FancyArrowPatch((tx, y_code), (tx, 20), arrowstyle="-", color=color, linewidth=1.3, zorder=2))
    ax.text(tx, 19, text, ha="center", va="top", fontsize=8, color=color, fontweight="bold")

ax.text(x, 41, "Sentence ID schema:  {section}_{paragraph:04d}_{sentence_index}",
        fontsize=11.5, color=ACCENT_DARK, fontweight="bold")

# secondary examples
examples = [
    ("title_0000_0", "title section, paragraph 0, sentence 0"),
    ("description_0003_7", "description section, paragraph 3, sentence 7"),
]
ey = 8
for code, meaning in examples:
    ax.text(x, ey, code, fontsize=9.5, family="monospace", color=ACCENT_DARK, fontweight="bold")
    ax.text(x + 32, ey, "→  " + meaning, fontsize=9, color=GREY)
    ey -= 4.5

fig.tight_layout(pad=0.3)
fig.savefig("fig4_sentence_id.png", dpi=300, facecolor="white")
print("saved")
