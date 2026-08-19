#!/usr/bin/env python3
"""MINT-Pat JaParaPat sample preprocessing analysis."""
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EN_PATH = Path(r"d:\en-ja\jp-us\jp-us\2020\JP2020000023-US20210105929.en")
JA_PATH = Path(r"d:\en-ja\jp-us\jp-us\2020\JP2020000023-US20210105929.ja")


def load_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split("\t")
                sid, text = parts[0], parts[1] if len(parts) > 1 else ""
            else:
                sid, text = parts[0], parts[1]
            rows.append({"id": sid, "text": text})
    return rows


def section_from_id(sid):
    return sid.split("_")[0]


def trailing_index_part(sid):
    """Return the part after last underscore (may be comma-separated)."""
    return sid.rsplit("_", 1)[-1]


def parse_indices(sid):
    """Parse trailing sentence index(es) as integers."""
    part = trailing_index_part(sid)
    return [int(x) for x in part.split(",")]


def truncate(s, n=80):
    s = s.replace("\t", " ")
    return s[:n] + ("..." if len(s) > n else "")


def main():
    en_rows = load_file(EN_PATH)
    ja_rows = load_file(JA_PATH)

  # --- Step 1 ---
    print("=" * 80)
    print("STEP 1: LOAD AND INSPECT")
    print("=" * 80)
    for label, rows in [("EN", en_rows), ("JA", ja_rows)]:
        print(f"\n--- {label} ---")
        print(f"Total lines: {len(rows)}")
        print("\nFirst 10 lines (ID | first 80 chars):")
        for r in rows[:10]:
            print(f"  {r['id']}\t{truncate(r['text'], 80)}")
        print("\nLast 5 lines:")
        for r in rows[-5:]:
            print(f"  {r['id']}\t{truncate(r['text'], 80)}")
        sections = sorted({section_from_id(r["id"]) for r in rows})
        print(f"\nUnique section prefixes: {sections}")

    # --- Step 2 ---
    print("\n" + "=" * 80)
    print("STEP 2: SENTENCE ID STRUCTURE")
    print("=" * 80)

    def index_sets(rows):
        all_idx = set()
        row_to_indices = {}
        comma_ids = []
        for r in rows:
            idxs = parse_indices(r["id"])
            row_to_indices[r["id"]] = idxs
            for i in idxs:
                all_idx.add(i)
            if "," in trailing_index_part(r["id"]):
                comma_ids.append(r)
        return all_idx, row_to_indices, comma_ids

    en_idx, en_map, en_comma = index_sets(en_rows)
    ja_idx, ja_map, ja_comma = index_sets(ja_rows)

    inter = en_idx & ja_idx
    en_only = en_idx - ja_idx
    ja_only = ja_idx - en_idx

    print(f"\nEN unique sentence indices: {len(en_idx)} (range {min(en_idx)}-{max(en_idx)})")
    print(f"JA unique sentence indices: {len(ja_idx)} (range {min(ja_idx)}-{max(ja_idx)})")
    print(f"Intersection: {len(inter)}")
    print(f"EN-only indices: {len(en_only)} -> {sorted(en_only)[:30]}{'...' if len(en_only)>30 else ''}")
    print(f"JA-only indices: {len(ja_only)} -> {sorted(ja_only)[:30]}{'...' if len(ja_only)>30 else ''}")

    print(f"\nComma-separated IDs in EN: {len(en_comma)}")
    for r in en_comma[:5]:
        print(f"  {r['id']}\t{truncate(r['text'], 60)}")
    if len(en_comma) > 5:
        print(f"  ... and {len(en_comma)-5} more")
    print(f"Comma-separated IDs in JA: {len(ja_comma)}")
    for r in ja_comma[:5]:
        print(f"  {r['id']}\t{truncate(r['text'], 60)}")

    print("\nSentences per section:")
    for label, rows in [("EN", en_rows), ("JA", ja_rows)]:
        c = Counter(section_from_id(r["id"]) for r in rows)
        print(f"  {label}: {dict(sorted(c.items()))}")

    # --- Step 3: Alignment ---
    print("\n" + "=" * 80)
    print("STEP 3: ALIGNMENT VIA SENTENCE INDEX")
    print("=" * 80)

    # index -> list of row dicts
    en_by_idx = defaultdict(list)
    ja_by_idx = defaultdict(list)
    for r in en_rows:
        for i in parse_indices(r["id"]):
            en_by_idx[i].append(r)
    for r in ja_rows:
        for i in parse_indices(r["id"]):
            ja_by_idx[i].append(r)

    matched_indices = sorted(en_idx & ja_idx)
    pairs_1to1 = []
    pairs_1tomany = []
    pairs_manyto1 = []
    examples_shown = 0

    for idx in matched_indices:
        el = en_by_idx[idx]
        jl = ja_by_idx[idx]
        en_ids = [r["id"] for r in el]
        ja_ids = [r["id"] for r in jl]
        if len(el) == 1 and len(jl) == 1:
            pairs_1to1.append((idx, el[0], jl[0]))
        elif len(el) == 1 and len(jl) > 1:
            pairs_1tomany.append((idx, el, jl))
        elif len(el) > 1 and len(jl) == 1:
            pairs_manyto1.append((idx, el, jl))
        else:
            pairs_manyto1.append((idx, el, jl))  # many-many at index level

    print(f"\nMatched indices (shared counter): {len(matched_indices)}")
    print(f"1-to-1 aligned pairs: {len(pairs_1to1)}")
    print(f"1-to-many (one EN -> multiple JA at same index): {len(pairs_1tomany)}")
    print(f"many-to-1 (multiple EN -> one JA at same index): {len(pairs_manyto1)}")

    en_unmatched_rows = [r for r in en_rows if not any(i in ja_idx for i in parse_indices(r["id"]))]
    ja_unmatched_rows = [r for r in ja_rows if not any(i in en_idx for i in parse_indices(r["id"]))]

    print(f"EN sentences with NO matching index in JA: {len(en_unmatched_rows)}")
    print(f"JA sentences with NO matching index in EN: {len(ja_unmatched_rows)}")

    print("\nSample matched pairs (first 15):")
    for idx, er, jr in pairs_1to1[:15]:
        print(f"  idx={idx} | 1-to-1 | EN:{er['id']} | JA:{jr['id']}")
        print(f"    EN: {truncate(er['text'], 100)}")
        print(f"    JA: {truncate(jr['text'], 100)}")

    if pairs_1tomany:
        print(f"\nSample 1-to-many (first 5 of {len(pairs_1tomany)}):")
        for idx, el, jl in pairs_1tomany[:5]:
            print(f"  idx={idx} | EN:{el[0]['id']} -> {len(jl)} JA lines")
            for j in jl:
                print(f"    JA {j['id']}: {truncate(j['text'], 80)}")

    if pairs_manyto1:
        print(f"\nSample many-to-1 (first 5 of {len(pairs_manyto1)}):")
        for idx, el, jl in pairs_manyto1[:5]:
            print(f"  idx={idx} | {len(el)} EN -> JA:{jl[0]['id']}")
            for e in el:
                print(f"    EN {e['id']}: {truncate(e['text'], 80)}")

    # --- Step 4: Section-level ---
    print("\n" + "=" * 80)
    print("STEP 4: SECTION-LEVEL ANALYSIS")
    print("=" * 80)

    def section_stats(rows, other_idx_set, by_idx_other):
        stats = {}
        for sec in ["title", "abstract", "description", "claim"]:
            sec_rows = [r for r in rows if section_from_id(r["id"]) == sec]
            aligned = 0
            patterns = Counter()
            for r in sec_rows:
                idxs = parse_indices(r["id"])
                matched = [i for i in idxs if i in other_idx_set]
                if matched:
                    aligned += 1
                    # count partners at first matched index
                    partners = sum(len(by_idx_other[i]) for i in matched)
                    if partners == 1:
                        patterns["1-to-1"] += 1
                    elif partners > 1:
                        patterns["1-to-many_or_many"] += 1
            stats[sec] = {
                "count": len(sec_rows),
                "aligned": aligned,
                "patterns": patterns,
            }
        return stats

    en_by_idx_full = defaultdict(list)
    ja_by_idx_full = defaultdict(list)
    for r in en_rows:
        for i in parse_indices(r["id"]):
            en_by_idx_full[i].append(r)
    for r in ja_rows:
        for i in parse_indices(r["id"]):
            ja_by_idx_full[i].append(r)

    en_sec = section_stats(en_rows, ja_idx, ja_by_idx_full)
    ja_sec = section_stats(ja_rows, en_idx, en_by_idx_full)

    print(f"{'Section':<12} {'EN sents':>8} {'JA sents':>8} {'1-to-1 pairs':>12} {'EN aligned':>10} {'JA aligned':>10} {'Common pattern':>20}")
    for sec in ["title", "abstract", "description", "claim"]:
        en_c = en_sec[sec]["count"]
        ja_c = ja_sec[sec]["count"]
        # 1-to-1 pairs in this section
        sec_1to1 = sum(
            1
            for idx, er, jr in pairs_1to1
            if section_from_id(er["id"]) == sec and section_from_id(jr["id"]) == sec
        )
        en_al = en_sec[sec]["aligned"]
        ja_al = ja_sec[sec]["aligned"]
        pat = en_sec[sec]["patterns"].most_common(1)
        pat_str = pat[0][0] if pat else "n/a"
        print(f"{sec:<12} {en_c:>8} {ja_c:>8} {sec_1to1:>12} {en_al:>10} {ja_al:>10} {pat_str:>20}")

    figref = [r for r in ja_rows if section_from_id(r["id"]) == "figref"]
    print(f"\nfigref (JA only): {len(figref)} lines")
    print("Sample figref lines:")
    for r in figref[:5]:
        print(f"  {r['id']}\t{truncate(r['text'], 100)}")
    if len(figref) > 5:
        print(f"  ... {len(figref)-5} more")

    # --- Step 5: Text quality ---
    print("\n" + "=" * 80)
    print("STEP 5: TEXT QUALITY (1-to-1 pairs)")
    print("=" * 80)

    problems = {
        "en_short": [],
        "ja_short": [],
        "xml": [],
        "ja_suspicious": [],
    }
    for idx, er, jr in pairs_1to1:
        if len(er["text"].strip()) < 5:
            problems["en_short"].append((idx, er, jr))
        if len(jr["text"].strip()) < 5:
            problems["ja_short"].append((idx, er, jr))
        if "<" in er["text"] or ">" in er["text"] or "<" in jr["text"] or ">" in jr["text"]:
            problems["xml"].append((idx, er, jr))
        en_wc = len(er["text"].split())
        ja_len = len(jr["text"])
        if en_wc > 20 and ja_len < en_wc * 0.3:
            problems["ja_suspicious"].append((idx, er, jr, en_wc, ja_len))

    for k, v in problems.items():
        print(f"\n{k}: {len(v)}")
        for item in v[:5]:
            if k == "ja_suspicious":
                idx, er, jr, ew, jl = item
                print(f"  idx={idx} EN_words={ew} JA_chars={jl}")
            else:
                idx, er, jr = item
                print(f"  idx={idx}")
            print(f"    EN [{er['id']}]: {truncate(er['text'], 100)}")
            print(f"    JA [{jr['id']}]: {truncate(jr['text'], 100)}")

    # --- Step 6: Token length ---
    print("\n" + "=" * 80)
    print("STEP 6: TOKEN LENGTH DISTRIBUTION (1-to-1 pairs)")
    print("=" * 80)

    en_words = [len(er["text"].split()) for _, er, jr in pairs_1to1]
    ja_chars = [len(jr["text"]) for _, er, jr in pairs_1to1]

    def stats_report(vals, name):
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
        }

    en_st = stats_report(en_words, "EN")
    ja_st = stats_report(ja_chars, "JA")
    en_over_400 = sum(1 for w in en_words if w > 400)
    ja_over_1000 = sum(1 for c in ja_chars if c > 1000)

    # rough 512 subword estimate: ~1.3 tokens per EN word, ~0.5 tokens per JA char (conservative)
    likely_over_512 = sum(
        1
        for _, er, jr in pairs_1to1
        if len(er["text"].split()) * 1.3 + len(jr["text"]) * 0.5 > 512
    )

    print(f"English word count (whitespace split): n={len(en_words)}")
    print(f"  min={en_st['min']} max={en_st['max']} mean={en_st['mean']:.1f} median={en_st['median']}")
    print(f"  sentences > 400 words: {en_over_400}")
    print(f"Japanese char count: n={len(ja_chars)}")
    print(f"  min={ja_st['min']} max={ja_st['max']} mean={ja_st['mean']:.1f} median={ja_st['median']}")
    print(f"  sentences > 1000 chars: {ja_over_1000}")
    print(f"Pairs likely exceeding 512 subword tokens (EN+JA concat, rough est.): {likely_over_512}")

    # --- Step 7: Vocabulary ---
    print("\n" + "=" * 80)
    print("STEP 7: CONCEPT VOCABULARY SPOT CHECK")
    print("=" * 80)

    terms = [
        ("semiconductor", ["半導体"]),
        ("controller", ["制御装置", "コントローラ"]),
        ("prior art", ["先行技術"]),
        ("claim", ["請求項"]),
        ("steering", ["操舵", "ステアリング"]),
    ]

    for en_term, ja_terms in terms:
        en_hits = [(idx, er, jr) for idx, er, jr in pairs_1to1 if en_term.lower() in er["text"].lower()]
        ja_hits = [
            (idx, er, jr)
            for idx, er, jr in pairs_1to1
            if any(t in jr["text"] for t in ja_terms)
        ]
        both = [
            (idx, er, jr)
            for idx, er, jr in pairs_1to1
            if en_term.lower() in er["text"].lower()
            and any(t in jr["text"] for t in ja_terms)
        ]
        print(f"\n{en_term} / {ja_terms}")
        print(f"  EN sentences with term: {len(en_hits)}")
        print(f"  JA sentences with term: {len(ja_hits)}")
        print(f"  Both in aligned pair: {len(both)}")
        for i, (idx, er, jr) in enumerate(both[:2]):
            print(f"  Example {i+1} idx={idx}:")
            print(f"    EN: {truncate(er['text'], 100)}")
            print(f"    JA: {truncate(jr['text'], 100)}")
        if not both and en_hits:
            print("  (No both-match; showing EN-only example:)")
            idx, er, jr = en_hits[0]
            print(f"    EN: {truncate(er['text'], 100)}")
            print(f"    JA: {truncate(jr['text'], 100)}")

    # --- Step 8 summary numbers ---
    print("\n" + "=" * 80)
    print("STEP 8: YIELD SUMMARY NUMBERS")
    print("=" * 80)
    print(f"Clean 1-to-1 pairs from this document: {len(pairs_1to1)}")
    print(f"Estimated across 1.4M documents: {len(pairs_1to1) * 1_400_000:,}")


if __name__ == "__main__":
    main()
