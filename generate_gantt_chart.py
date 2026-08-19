"""
Gantt chart generator for the MINT-Pat thesis project
(Mechanistic Interpretability of Cross-Lingual Concept Alignment in a
Domain-Adapted Multilingual Patent Language Model).

Phases below are derived from the project's actual methodology (Section 3.1,
"Desk-Based Agile Research Approach") and technical implementation chapters:
literature review -> environment setup -> corpus curation/preprocessing pipeline
-> model retraining (domain adaptation) -> six-module interpretability
experiments -> synthesis/critical evaluation -> final write-up, with ethical
reflection running continuously throughout (Section 3.5).

Edit PROJECT_START, PROJECT_END, or the PHASES list below and re-run to
regenerate the chart with updated dates/tasks.
"""

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 1. Project timeframe
# ---------------------------------------------------------------------------
PROJECT_START = dt.date(2026, 3, 10)
PROJECT_END = dt.date(2026, 7, 31)
TOTAL_DAYS = (PROJECT_END - PROJECT_START).days

# ---------------------------------------------------------------------------
# 2. Sequential phases, each given a relative weight (in "weeks") that is
#    scaled proportionally to fill the exact PROJECT_START -> PROJECT_END span.
#    Adjust the weights (or add/remove phases) to change relative durations.
# ---------------------------------------------------------------------------
SEQUENTIAL_PHASES = [
    ("Literature Review & Research Design", 3.0, "#4C72B0"),
    ("Methodology Design & Environment Setup", 1.5, "#55A868"),
    ("Corpus Curation & Preprocessing Pipeline", 4.0, "#C44E52"),
    ("Model Retraining (Domain Adaptation)", 2.0, "#8172B2"),
    ("Interpretability Experiments\n(CKA, SAE, Probe, Retrieval, RAF, Patching)", 4.0, "#CCB974"),
    ("Results Synthesis & Critical Evaluation", 2.0, "#64B5CD"),
    ("Final Thesis Writing & Submission", 2.0, "#B07AA1"),
]

# A cross-cutting task that runs for the full project duration, matching the
# thesis's explicit statement that ethical reflection "accompanies every
# sprint rather than being deferred" (Section 3.5).
CONTINUOUS_PHASES = [
    ("Ethical Reflection & Reproducibility Documentation", "#999999"),
]

# ---------------------------------------------------------------------------
# 3. Compute start/end dates for each sequential phase
# ---------------------------------------------------------------------------
total_weight = sum(w for _, w, _ in SEQUENTIAL_PHASES)
scale = TOTAL_DAYS / total_weight

rows = []  # (label, start_date, end_date, color)
cursor = PROJECT_START
for label, weight, color in SEQUENTIAL_PHASES:
    duration_days = round(weight * scale)
    phase_start = cursor
    phase_end = phase_start + dt.timedelta(days=duration_days)
    rows.append((label, phase_start, phase_end, color))
    cursor = phase_end

# Make sure the last phase ends exactly on PROJECT_END (rounding correction)
last_label, last_start, _, last_color = rows[-1]
rows[-1] = (last_label, last_start, PROJECT_END, last_color)

# Append the continuous phase(s) spanning the whole project
for label, color in CONTINUOUS_PHASES:
    rows.append((label, PROJECT_START, PROJECT_END, color))

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.5))

y_positions = range(len(rows))
for i, (label, start, end, color) in enumerate(rows):
    duration = (end - start).days
    ax.barh(
        i, duration, left=mdates.date2num(start), height=0.55,
        color=color, edgecolor="black", linewidth=0.6, zorder=3,
        alpha=0.95 if "Ethical" not in label else 0.55,
    )
    # duration label centered in the bar
    mid = mdates.date2num(start) + duration / 2
    ax.text(
        mid, i, f"{duration}d", ha="center", va="center",
        fontsize=8, color="white" if "Ethical" not in label else "black",
        fontweight="bold", zorder=4,
    )

ax.set_yticks(list(y_positions))
ax.set_yticklabels([label for label, *_ in rows], fontsize=9.5)
ax.invert_yaxis()

# x-axis as dates
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))  # weekly gridlines (Mondays)
ax.grid(axis="x", which="major", color="#888888", linewidth=0.8, alpha=0.6, zorder=0)
ax.grid(axis="x", which="minor", color="#dddddd", linewidth=0.5, alpha=0.6, zorder=0)

ax.set_xlim(mdates.date2num(PROJECT_START) - 3, mdates.date2num(PROJECT_END) + 3)

# "today" marker
today = dt.date.today()
if PROJECT_START <= today <= PROJECT_END:
    ax.axvline(mdates.date2num(today), color="red", linestyle="--", linewidth=1.2, zorder=5)
    ax.text(mdates.date2num(today), -0.9, "Today", color="red", fontsize=8,
            ha="center", fontweight="bold")

ax.set_title(
    "MINT-Pat Thesis Project Timeline\n"
    f"{PROJECT_START:%d %b %Y} – {PROJECT_END:%d %b %Y}",
    fontsize=13, fontweight="bold", pad=14,
)
ax.set_xlabel("Timeline", fontsize=10)

legend_handles = [Patch(facecolor="#999999", alpha=0.55, edgecolor="black", label="Continuous task")]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

fig.tight_layout()
OUTPUT_FILE = "gantt_chart.png"
fig.savefig(OUTPUT_FILE, dpi=200)
print(f"Saved Gantt chart to {OUTPUT_FILE}")

# Print a text summary of the schedule for quick reference
print("\nPhase schedule:")
for label, start, end, _ in rows:
    print(f"  {start:%Y-%m-%d} -> {end:%Y-%m-%d}  ({(end-start).days:>3}d)  {label.splitlines()[0]}")
