#!/usr/bin/env python3
import json, statistics
from collections import defaultdict

with open(r"D:/en-ja/level2_results.json", encoding="utf-8") as f:
    d = json.load(f)
samples = d["samples_summary"]

for corp in ["jp-us", "jp-x-us", "pct", "us-jp"]:
    vals = [s["approx_align_pct"] for s in samples if s["corpus"] == corp]
    print(f"{corp}: mean={statistics.mean(vals):.1f} min={min(vals)} max={max(vals)}")

by_y = defaultdict(list)
for s in samples:
    by_y[s["year"]].append(s["approx_align_pct"])
for y in sorted(by_y):
    v = by_y[y]
    print(f"year {y}: mean={statistics.mean(v):.1f} range={min(v)}-{max(v)}")

all_secs = set()
for s in samples:
    all_secs |= set(s["en_sections"]) | set(s["ja_sections"])
print("sections:", sorted(all_secs))
print("figref in JA:", sum(1 for s in samples if "figref" in s["ja_sections"]))
