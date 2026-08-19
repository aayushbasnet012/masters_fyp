"""
build_eval_set.py — Generate curation CSVs for the 500-pair manual evaluation set.

Outputs (all in output/curation/):
  1. curation_candidates.csv     — scored + sorted concept_candidates.tsv (you fill KEEP column)
  2. curation_prior_art.csv      — mined prior art pairs from concept_pairs_eval.tsv
  3. curation_novelty.csv        — mined novelty pairs from concept_pairs_eval.tsv
  4. curation_mixed_generic.csv  — random sample of general pairs (you pick 150)

Run:
  python build_eval_set.py
"""

import csv
import re
import random
import os
import unicodedata

random.seed(42)

CANDIDATES_TSV   = "output/concept_candidates.tsv"
EVAL_TSV         = "output/concept_pairs_eval.tsv"
OUT_DIR          = "output/curation"

os.makedirs(OUT_DIR, exist_ok=True)

# Japanese terms per concept (from SKILLS.MD)
JA_TERMS = {
    "semiconductor": ["半導体"],
    "prior art":     ["先行技術"],
    "claim":         ["請求項"],
    "controller":    ["制御装置", "コントローラ"],
    "embodiment":    ["実施形態"],
    "novelty":       ["新規性"],
    "invention":     ["発明"],
    "patent":        ["特許"],
    "signal":        ["信号"],
    "device":        ["装置"],
    "method":        ["方法"],
    "system":        ["システム"],
}

TECHNICAL_CONCEPTS      = {"semiconductor", "signal", "controller", "device", "system", "method"}
LEGAL_CONCEPTS          = {"claim", "prior art", "novelty", "embodiment", "patent", "invention"}

def cjk_density(text):
    """Fraction of characters that are CJK/hiragana/katakana."""
    if not text:
        return 0.0
    cjk = sum(
        1 for ch in text
        if '぀' <= ch <= 'ヿ'   # hiragana + katakana
        or '㐀' <= ch <= '鿿'   # CJK unified + extension A
        or '豈' <= ch <= '﫿'   # CJK compatibility
    )
    return cjk / len(text)

def en_word_count(text):
    return len(text.split())

def length_ratio(en_text, ja_text):
    """Rough EN-word to JA-char ratio. Japanese ~5.5 chars per English word on average."""
    en_words = en_word_count(en_text)
    ja_chars = len(ja_text)
    if ja_chars == 0:
        return 0.0
    return (en_words * 5.5) / ja_chars

def ja_term_present(concept_en, ja_text):
    """Check that at least one Japanese term for the concept appears in ja_text."""
    terms = JA_TERMS.get(concept_en, [])
    return any(t in ja_text for t in terms)

def en_term_present(concept_en, en_text):
    return concept_en.lower() in en_text.lower()

def score_pair(concept_en, en_text, ja_text):
    """Return (score 0-5, dict of individual check results)."""
    checks = {
        "ja_term":     ja_term_present(concept_en, ja_text),
        "cjk_ok":      cjk_density(ja_text) > 0.20,
        "length_ok":   0.30 <= length_ratio(en_text, ja_text) <= 4.5,
        "en_term":     en_term_present(concept_en, en_text),
        "en_length":   en_word_count(en_text) >= 5,
    }
    score = sum(checks.values())
    return score, checks

def suggest_keep(score):
    if score == 5:
        return "yes"
    elif score == 4:
        return "?"
    else:
        return "no"

def concept_category(concept_en):
    if concept_en in TECHNICAL_CONCEPTS:
        return "technical"
    elif concept_en in LEGAL_CONCEPTS:
        return "legal-procedural"
    return "unknown"


# ── Step 1: Score concept_candidates.tsv ────────────────────────────────────

print("Scoring concept_candidates.tsv ...")

rows = []
with open(CANDIDATES_TSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        concept_en = row['concept_en'].strip()
        en_text    = row['en_text'].strip()
        ja_text    = row['ja_text'].strip()

        score, checks = score_pair(concept_en, en_text, ja_text)

        rows.append({
            "concept_en":    concept_en,
            "concept_ja":    row['concept_ja'].strip(),
            "category":      concept_category(concept_en),
            "section":       row['section'].strip(),
            "rank_score":    row['rank_score'].strip(),
            "quality_score": score,
            "ja_term_ok":    "Y" if checks["ja_term"]   else "N",
            "cjk_ok":        "Y" if checks["cjk_ok"]    else "N",
            "length_ok":     "Y" if checks["length_ok"] else "N",
            "en_term_ok":    "Y" if checks["en_term"]   else "N",
            "en_length_ok":  "Y" if checks["en_length"] else "N",
            "KEEP":          suggest_keep(score),
            "en_text":       en_text,
            "ja_text":       ja_text,
            "source_file":   row['source_file'].strip(),
        })

# Sort: by category, then quality_score desc, then rank_score desc
rows.sort(key=lambda r: (
    r['category'],
    -int(r['quality_score']),
    -float(r['rank_score'])
))

curation_path = os.path.join(OUT_DIR, "curation_candidates.csv")
fieldnames = [
    "concept_en", "concept_ja", "category", "section", "rank_score",
    "quality_score", "ja_term_ok", "cjk_ok", "length_ok", "en_term_ok", "en_length_ok",
    "KEEP", "en_text", "ja_text", "source_file"
]
with open(curation_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Print per-concept summary
from collections import Counter
by_concept = {}
for r in rows:
    c = r['concept_en']
    if c not in by_concept:
        by_concept[c] = {"yes": 0, "?": 0, "no": 0}
    by_concept[c][r['KEEP']] += 1

print(f"\n{'Concept':<20} {'total':>5} {'auto-yes':>8} {'maybe':>6} {'auto-no':>8} {'category'}")
print("-" * 70)
for concept, counts in sorted(by_concept.items(), key=lambda x: concept_category(x[0])):
    total = sum(counts.values())
    print(f"{concept:<20} {total:>5} {counts['yes']:>8} {counts['?']:>6} {counts['no']:>8}  {concept_category(concept)}")
print(f"\nWrote {len(rows)} rows → {curation_path}")


# ── Step 2: Mine prior art + novelty from concept_pairs_eval.tsv ─────────────

print("\nMining prior art and novelty from concept_pairs_eval.tsv ...")

PA_JA_TERMS     = ["先行技術"]
NOVELTY_JA_TERMS = ["新規性"]
# Also check English side
PA_EN_PATTERNS      = [r'\bprior art\b', r'\bprior-art\b']
NOVELTY_EN_PATTERNS = [r'\bnovelty\b', r'\bnovel\b']

prior_art_rows = []
novelty_rows   = []

with open(EVAL_TSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        en_text = row['en_text'].strip()
        ja_text = row['ja_text'].strip()
        section = row['section'].strip()
        source  = row['source_file'].strip()

        # Prior art
        pa_ja = any(t in ja_text for t in PA_JA_TERMS)
        pa_en = any(re.search(p, en_text, re.IGNORECASE) for p in PA_EN_PATTERNS)
        if pa_ja or pa_en:
            score, checks = score_pair("prior art", en_text, ja_text)
            # Override ja_term check with our specific search
            ja_hit = pa_ja
            prior_art_rows.append({
                "concept_en":    "prior art",
                "concept_ja":    "先行技術",
                "category":      "legal-procedural",
                "section":       section,
                "quality_score": score,
                "ja_term_ok":    "Y" if ja_hit  else "N",
                "cjk_ok":        "Y" if checks["cjk_ok"]    else "N",
                "length_ok":     "Y" if checks["length_ok"] else "N",
                "en_term_ok":    "Y" if pa_en               else "N",
                "en_length_ok":  "Y" if checks["en_length"] else "N",
                "KEEP":          suggest_keep(score),
                "en_text":       en_text,
                "ja_text":       ja_text,
                "source_file":   source,
            })

        # Novelty
        nv_ja = any(t in ja_text for t in NOVELTY_JA_TERMS)
        nv_en = any(re.search(p, en_text, re.IGNORECASE) for p in NOVELTY_EN_PATTERNS)
        if nv_ja or nv_en:
            score, checks = score_pair("novelty", en_text, ja_text)
            ja_hit = nv_ja
            novelty_rows.append({
                "concept_en":    "novelty",
                "concept_ja":    "新規性",
                "category":      "legal-procedural",
                "section":       section,
                "quality_score": score,
                "ja_term_ok":    "Y" if ja_hit  else "N",
                "cjk_ok":        "Y" if checks["cjk_ok"]    else "N",
                "length_ok":     "Y" if checks["length_ok"] else "N",
                "en_term_ok":    "Y" if nv_en               else "N",
                "en_length_ok":  "Y" if checks["en_length"] else "N",
                "KEEP":          suggest_keep(score),
                "en_text":       en_text,
                "ja_text":       ja_text,
                "source_file":   source,
            })

mine_fields = [
    "concept_en", "concept_ja", "category", "section",
    "quality_score", "ja_term_ok", "cjk_ok", "length_ok", "en_term_ok", "en_length_ok",
    "KEEP", "en_text", "ja_text", "source_file"
]

prior_art_rows.sort(key=lambda r: -r['quality_score'])
pa_path = os.path.join(OUT_DIR, "curation_prior_art.csv")
with open(pa_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=mine_fields)
    writer.writeheader()
    writer.writerows(prior_art_rows)
print(f"Prior art: found {len(prior_art_rows)} candidates → {pa_path}")

novelty_rows.sort(key=lambda r: -r['quality_score'])
nv_path = os.path.join(OUT_DIR, "curation_novelty.csv")
with open(nv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=mine_fields)
    writer.writeheader()
    writer.writerows(novelty_rows)
print(f"Novelty:   found {len(novelty_rows)} candidates → {nv_path}")


# ── Step 3: Sample mixed/generic pairs ──────────────────────────────────────

print("\nSampling mixed/generic pairs from concept_pairs_eval.tsv ...")

# Mixed/generic = pairs that pass quality filters but are sampled across
# all concepts to represent general patent language. Sample 250 so the user
# can pick 150 after reading.
MIXED_SAMPLE_TARGET = 250

all_eval_rows = []
with open(EVAL_TSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        en_text    = row['en_text'].strip()
        ja_text    = row['ja_text'].strip()
        concept_en = row['concept_en'].strip()
        section    = row['section'].strip()
        source     = row['source_file'].strip()

        score, checks = score_pair(concept_en, en_text, ja_text)
        if score >= 4:  # only sample quality pairs
            all_eval_rows.append({
                "concept_en":    concept_en,
                "category":      "mixed-generic",
                "section":       section,
                "quality_score": score,
                "ja_term_ok":    "Y" if checks["ja_term"]   else "N",
                "cjk_ok":        "Y" if checks["cjk_ok"]    else "N",
                "length_ok":     "Y" if checks["length_ok"] else "N",
                "en_term_ok":    "Y" if checks["en_term"]   else "N",
                "en_length_ok":  "Y" if checks["en_length"] else "N",
                "KEEP":          "?",
                "en_text":       en_text,
                "ja_text":       ja_text,
                "source_file":   source,
            })

# Stratified sample across concepts so no single concept dominates
by_concept_pool = {}
for r in all_eval_rows:
    c = r['concept_en']
    by_concept_pool.setdefault(c, []).append(r)

# Take up to 25 per concept (12 concepts × 25 = 300 max, trim to 250)
mixed_sample = []
per_concept = max(1, MIXED_SAMPLE_TARGET // len(by_concept_pool))
for concept, pool in sorted(by_concept_pool.items()):
    random.shuffle(pool)
    mixed_sample.extend(pool[:per_concept])

random.shuffle(mixed_sample)
mixed_sample = mixed_sample[:MIXED_SAMPLE_TARGET]

mixed_fields = [
    "concept_en", "category", "section",
    "quality_score", "ja_term_ok", "cjk_ok", "length_ok", "en_term_ok", "en_length_ok",
    "KEEP", "en_text", "ja_text", "source_file"
]

mx_path = os.path.join(OUT_DIR, "curation_mixed_generic.csv")
with open(mx_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=mixed_fields)
    writer.writeheader()
    writer.writerows(mixed_sample)

print(f"Mixed/generic: sampled {len(mixed_sample)} pairs → {mx_path}")

print("\n── Summary ──────────────────────────────────────────────────────────────")
print(f"  curation_candidates.csv   : {len(rows):>4} rows  (target: keep 200 technical + 113 legal)")
print(f"  curation_prior_art.csv    : {len(prior_art_rows):>4} rows  (target: keep ~25)")
print(f"  curation_novelty.csv      : {len(novelty_rows):>4} rows  (target: keep ~12)")
print(f"  curation_mixed_generic.csv: {len(mixed_sample):>4} rows  (target: keep 150)")
print()
print("Next step: open each CSV in a spreadsheet, read the en_text column,")
print("and change KEEP from 'yes'/'?'/'no' to 'yes' or 'no' as appropriate.")
print("Then run: python assemble_eval_set.py  (to be written after curation)")
