"""Chapter 3 figure: the two-week agile sprint cycle."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style import ACCENT, ACCENT_DARK, GREY, ACCENT_LIGHT

fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.set_xlim(-10, 110); ax.set_ylim(-10, 110); ax.axis("off"); ax.set_aspect("equal")
fig.patch.set_facecolor("white")

cx, cy, R = 50, 50, 36
stages = [
    ("Sprint Plan", "Define the sprint's deliverable\nand review checkpoint"),
    ("Build & Experiment", "Implement the stage/module\nscheduled for this sprint"),
    ("Review vs. RQs\n& Hypotheses", "Check findings against the\nresearch questions/hypotheses"),
    ("Adjust Next Sprint", "Redirect analytical focus\nif an early finding warrants it"),
]
n = len(stages)
angles = [90 - i * (360 / n) for i in range(n)]  # start at top, clockwise

positions = []
for ang in angles:
    rad = np.deg2rad(ang)
    x, y = cx + R * np.cos(rad), cy + R * np.sin(rad)
    positions.append((x, y))

box_w, box_h = 34, 16
for (x, y), (title, sub) in zip(positions, stages):
    b = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                        boxstyle="round,pad=0.3,rounding_size=2", linewidth=1.3,
                        edgecolor=ACCENT_DARK, facecolor=ACCENT, zorder=3)
    ax.add_patch(b)
    ax.text(x, y + 3.2, title, ha="center", va="center", fontsize=9.3, color="white",
            fontweight="bold", zorder=4)
    ax.text(x, y - 3.8, sub, ha="center", va="center", fontsize=6.7, color="white", zorder=4)

# curved arrows between consecutive stages, following the circle
for i in range(n):
    a1 = angles[i]
    a2 = angles[(i + 1) % n]
    # place arrow along the arc, offset inward slightly from box edges
    mid = (a1 + a2) / 2 if a1 > a2 else (a1 + a2 - 360) / 2
    start_ang = a1 - 22
    end_ang = a2 + 22 if a2 < a1 else a2 + 22 - 360
    r1 = np.deg2rad(start_ang); r2 = np.deg2rad(end_ang)
    x1, y1 = cx + R * np.cos(r1), cy + R * np.sin(r1)
    x2, y2 = cx + R * np.cos(r2), cy + R * np.sin(r2)
    arr = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad=-0.3",
                          arrowstyle="-|>", mutation_scale=16, color=GREY, linewidth=1.8, zorder=2)
    ax.add_patch(arr)

ax.text(cx, cy, "Two-Week\nSprint", ha="center", va="center", fontsize=11, color=ACCENT_DARK,
        fontweight="bold", zorder=4)

ax.text(cx, 102, "Repeats every two weeks, from tool familiarization through final synthesis",
        ha="center", va="top", fontsize=9.5, color=GREY, style="italic")

fig.tight_layout(pad=0.3)
fig.savefig("fig2_sprint_cycle.png", dpi=300, facecolor="white")
print("saved")
