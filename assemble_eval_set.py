"""
assemble_eval_set.py — Merge all curated KEEP=yes pairs into output/eval_set.tsv

Reads the four curation CSVs from output/curation/, keeps rows where KEEP=yes,
deduplicates by (en_text, ja_text), and writes output/eval_set.tsv.

Run:
    python assemble_eval_set.py

Output:
    output/eval_set.tsv   — final evaluation set for mint_pat_validate_experiments.py
"""

import csv
import os
from collections import Counter, defaultdict

CURATION_DIR = "output/curation"
OUT_TSV      = "output/eval_set.tsv"

SOURCES = [
    ("curation_candidates.csv",    "candidates"),
    ("curation_prior_art.csv",     "prior_art"),
    ("curation_novelty.csv",       "novelty"),
    ("curation_mixed_generic.csv", "mixed_generic"),
]

# concept_ja lookup for concepts not available in mixed_generic
CONCEPT_JA = {
    "semiconductor": "半導体",
    "prior art":     "先行技術",
    "claim":         "請求項",
    "controller":    "制御装置",
    "embodiment":    "実施形態",
    "novelty":       "新規性",
    "invention":     "発明",
    "patent":        "特許",
    "signal":        "信号",
    "device":        "装置",
    "method":        "方法",
    "system":        "システム",
}

TECHNICAL_CONCEPTS = {"semiconductor", "signal", "controller", "device", "system", "method"}
LEGAL_CONCEPTS     = {"claim", "prior art", "novelty", "embodiment", "patent", "invention"}

def concept_category(concept_en):
    if concept_en in TECHNICAL_CONCEPTS:
        return "technical"
    elif concept_en in LEGAL_CONCEPTS:
        return "legal-procedural"
    return "other"


rows_out = []
seen     = set()   # dedup by (en_text, ja_text)

for filename, source_tag in SOURCES:
    path = os.path.join(CURATION_DIR, filename)
    if not os.path.isfile(path):
        print(f"  SKIP (not found): {path}")
        continue

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("KEEP", "").strip().lower() != "yes":
                continue

            en_text    = row.get("en_text", "").strip()
            ja_text    = row.get("ja_text", "").strip()
            concept_en = row.get("concept_en", "").strip().lower()
            section    = row.get("section", "").strip()
            category   = row.get("category", "") or concept_category(concept_en)
            concept_ja = row.get("concept_ja", "") or CONCEPT_JA.get(concept_en, "")

            if not en_text or not ja_text:
                continue

            key = (en_text, ja_text)
            if key in seen:
                continue
            seen.add(key)

            rows_out.append({
                "concept_en": concept_en,
                "concept_ja": concept_ja,
                "category":   category,
                "section":    section,
                "en_text":    en_text,
                "ja_text":    ja_text,
                "source":     source_tag,
            })

# Write TSV
fieldnames = ["concept_en", "concept_ja", "category", "section", "en_text", "ja_text", "source"]
with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows_out)

# Summary
print(f"\neval_set.tsv written: {len(rows_out)} pairs  →  {OUT_TSV}")
print()

concept_counts   = Counter(r["concept_en"] for r in rows_out)
category_counts  = Counter(r["category"]   for r in rows_out)
source_counts    = Counter(r["source"]     for r in rows_out)
section_counts   = Counter(r["section"]    for r in rows_out)

print(f"{'Concept':<20}  {'n':>5}  {'category'}")
print("-" * 50)
for concept, n in sorted(concept_counts.items(), key=lambda x: concept_category(x[0])):
    print(f"  {concept:<18}  {n:>5}  {concept_category(concept)}")

print()
print(f"By category:")
for cat, n in sorted(category_counts.items()):
    print(f"  {cat:<20}: {n}")

print()
print(f"By source:")
for src, n in sorted(source_counts.items()):
    print(f"  {src:<20}: {n}")

print()
print(f"By section:")
for sec, n in sorted(section_counts.items()):
    print(f"  {sec:<20}: {n}")

print()
print("Next step:")
print("  Upload output/eval_set.tsv to Google Drive as 'eval_set.tsv'")
print("  In mint_pat_validate_experiments.py, set:")
print("    CONCEPT_FILE = '/content/drive/MyDrive/eval_set.tsv'")
print("    N_PAIRS = 300   # or higher — eval_set has", len(rows_out), "total pairs")
