import json
import importlib.util
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("l3", Path(r"D:/en-ja/level3_alignment.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

with open(r"D:/en-ja/level3_results.json", encoding="utf-8") as f:
    d = json.load(f)
with open(r"D:/en-ja/level2_results.json", encoding="utf-8") as f:
    l2 = json.load(f)

fr = d["file_results"]
for k in ["pairs", "1to1", "one_to_many_ja", "many_to_one_en", "cross_section", "many_to_many"]:
    print(k, sum(r[k] for r in fr))

maxlen = 0
maxlen_sections = Counter()
for s in l2["samples_summary"]:
    en = m.ROOT / s["corpus"] / s["corpus"] / s["year"] / f"{s['stem']}.en"
    ja = en.with_suffix(".ja")
    pairs, _ = m.align_patent_pair(en, ja)
    for x in pairs:
        if m.word_count_en(x["en_text"]) > 350 or m.ja_char_count(x["ja_text"]) > 1800:
            maxlen += 1
            maxlen_sections[m.section_from_id(x["en_id"] or x["ja_id"])] += 1
print("max_length", maxlen, dict(maxlen_sections))

raw = 13503
fm = d["failure_modes"]
for name, val in fm.items():
    print(name, val, f"{100*val/raw:.1f}%")

# extrapolation
factor = d["extrapolation_factor"]
for term in ["embodiment", "invention", "method", "semiconductor", "prior art"]:
    c = d["concept_counts"][term]
    print(f"extrap {term} both:", int(c["both"] * factor))
