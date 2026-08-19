#!/usr/bin/env python3
"""MINT-Pat Level 3: alignment, quality filters, concept analysis."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:/en-ja")
LEVEL2_JSON = ROOT / "level2_results.json"
FULL_CORPUS_PAIRS = 370_835

JA_CJK = re.compile(r"[\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff]")

CONCEPTS = [
    ("semiconductor", ["半導体"]),
    ("prior art", ["先行技術"]),
    ("claim", ["請求項"]),
    ("controller", ["制御装置", "コントローラ"]),
    ("embodiment", ["実施形態"]),
    ("novelty", ["新規性"]),
    ("invention", ["発明"]),
    ("patent", ["特許"]),
    ("signal", ["信号"]),
    ("device", ["装置"]),
    ("method", ["方法"]),
    ("system", ["システム"]),
]

# Content-proxy domains (IPC not on disk)
DOMAIN_KEYWORDS = {
    "electrical": ["semiconductor", "circuit", "signal", "processor", "electronic", "voltage", "半導体", "回路", "信号"],
    "chemistry": ["compound", "polymer", "reaction", "molecule", "NMR", "formula", "化合物", "ポリマー", "アゾール"],
    "mechanical": ["mechanism", "apparatus", "vehicle", "motor", "shaft", "装置", "機構", "車両"],
}


def load_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if "\t" in line:
                sid, text = line.split("\t", 1)
            else:
                sid, text = line, ""
            rows.append({"id": sid, "text": text})
    return rows


def section_from_id(sid: str) -> str:
    return sid.split("_")[0] if "_" in sid else sid


def parse_indices(sid: str) -> list[int]:
    if "_" not in sid:
        return []
    part = sid.rsplit("_", 1)[-1]
    out = []
    for x in part.split(","):
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def primary_index(sid: str) -> int | None:
    idxs = parse_indices(sid)
    return idxs[0] if idxs else None


def align_patent_pair(en_path: Path, ja_path: Path) -> tuple[list[dict], dict]:
    en_rows = load_tsv(en_path)
    ja_rows = load_tsv(ja_path)

    en_by_idx: dict[int, list[dict]] = defaultdict(list)
    ja_by_idx: dict[int, list[dict]] = defaultdict(list)

    for r in en_rows:
        for i in parse_indices(r["id"]):
            en_by_idx[i].append(r)
    for r in ja_rows:
        for i in parse_indices(r["id"]):
            ja_by_idx[i].append(r)

    en_idx_set = set(en_by_idx)
    ja_idx_set = set(ja_by_idx)
    shared = en_idx_set & ja_idx_set

    pairs: list[dict] = []
    used_en_ids: set[str] = set()
    used_ja_ids: set[str] = set()

    def make_pair(atype, er, jr, idx):
        sec = section_from_id(er["id"]) if er else section_from_id(jr["id"])
        return {
            "alignment_type": atype,
            "en_id": er["id"] if er else "",
            "ja_id": jr["id"] if jr else "",
            "en_text": er["text"] if er else "",
            "ja_text": jr["text"] if jr else "",
            "section": sec,
            "en_sentence_index": idx,
            "ja_sentence_index": idx,
        }

    for idx in sorted(shared):
        el = en_by_idx[idx]
        jl = ja_by_idx[idx]
        # dedupe same physical line listed once per index
        el_unique = list({r["id"]: r for r in el}.values())
        jl_unique = list({r["id"]: r for r in jl}.values())

        if len(el_unique) == 1 and len(jl_unique) == 1:
            er, jr = el_unique[0], jl_unique[0]
            if section_from_id(er["id"]) == section_from_id(jr["id"]):
                atype = "1to1"
            else:
                atype = "cross_section"
            pairs.append(make_pair(atype, er, jr, idx))
            used_en_ids.add(er["id"])
            used_ja_ids.add(jr["id"])
        elif len(el_unique) == 1 and len(jl_unique) > 1:
            er = el_unique[0]
            ja_text = " ".join(j["text"] for j in jl_unique)
            ja_id = ",".join(j["id"] for j in jl_unique)
            pairs.append(
                {
                    "alignment_type": "one_to_many_ja",
                    "en_id": er["id"],
                    "ja_id": ja_id,
                    "en_text": er["text"],
                    "ja_text": ja_text,
                    "section": section_from_id(er["id"]),
                    "en_sentence_index": idx,
                    "ja_sentence_index": idx,
                }
            )
            used_en_ids.add(er["id"])
            for j in jl_unique:
                used_ja_ids.add(j["id"])
        elif len(el_unique) > 1 and len(jl_unique) == 1:
            jr = jl_unique[0]
            en_text = " ".join(e["text"] for e in el_unique)
            en_id = ",".join(e["id"] for e in el_unique)
            pairs.append(
                {
                    "alignment_type": "many_to_one_en",
                    "en_id": en_id,
                    "ja_id": jr["id"],
                    "en_text": en_text,
                    "ja_text": jr["text"],
                    "section": section_from_id(jr["id"]),
                    "en_sentence_index": idx,
                    "ja_sentence_index": idx,
                }
            )
            for e in el_unique:
                used_en_ids.add(e["id"])
            used_ja_ids.add(jr["id"])
        else:
            # many-to-many: concatenate both sides in stable order
            en_text = " ".join(e["text"] for e in el_unique)
            ja_text = " ".join(j["text"] for j in jl_unique)
            pairs.append(
                {
                    "alignment_type": "many_to_many",
                    "en_id": ",".join(e["id"] for e in el_unique),
                    "ja_id": ",".join(j["id"] for j in jl_unique),
                    "en_text": en_text,
                    "ja_text": ja_text,
                    "section": section_from_id(el_unique[0]["id"]),
                    "en_sentence_index": idx,
                    "ja_sentence_index": idx,
                }
            )
            for e in el_unique:
                used_en_ids.add(e["id"])
            for j in jl_unique:
                used_ja_ids.add(j["id"])

    unmatched_en = [(r["id"], r["text"]) for r in en_rows if r["id"] not in used_en_ids]
    unmatched_ja = []
    for r in ja_rows:
        if r["id"] in used_ja_ids:
            continue
        if section_from_id(r["id"]) == "figref":
            continue  # discard JA-only figref
        unmatched_ja.append((r["id"], r["text"]))

    return pairs, {"unmatched_en": unmatched_en, "unmatched_ja": unmatched_ja}


def word_count_en(text: str) -> int:
    return len(text.split())


def ja_char_count(text: str) -> int:
    return len(text)


def ja_cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(JA_CJK.findall(text))
    return cjk / len(text)


def length_ratio(en_text: str, ja_text: str) -> float:
    el, jl = len(en_text), len(ja_text)
    if not el or not jl:
        return float("inf")
    return max(el, jl) / min(el, jl)


def apply_filters(pairs: list[dict]) -> tuple[list[dict], dict]:
    """Return kept pairs and per-filter removal counts."""
    stats = Counter()
    removed_examples = defaultdict(list)
    kept = []

    for p in pairs:
        en_t, ja_t = p["en_text"], p["ja_text"]
        reasons = []

        lr = length_ratio(en_t, ja_t)
        if lr > 5.0:
            reasons.append("length_ratio")

        if word_count_en(en_t) < 5 or ja_char_count(ja_t) < 10:
            reasons.append("min_length")

        if ja_cjk_ratio(ja_t) < 0.2:
            reasons.append("ja_density")

        if word_count_en(en_t) > 350 or ja_char_count(ja_t) > 1800:
            reasons.append("max_length")

        if reasons:
            for r in reasons:
                stats[r] += 1
                if len(removed_examples[r]) < 5:
                    removed_examples[r].append(p)
        else:
            kept.append(p)

    return kept, {"counts": dict(stats), "examples": dict(removed_examples)}


def classify_domain(en_text: str, ja_text: str) -> str:
    blob = (en_text + " " + ja_text).lower()
    scores = {}
    for dom, kws in DOMAIN_KEYWORDS.items():
        scores[dom] = sum(1 for k in kws if k.lower() in blob)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


def run_level3():
    with open(LEVEL2_JSON, encoding="utf-8") as f:
        l2 = json.load(f)

    file_results = []
    all_pairs: list[dict] = []
    stress_cases = [
        ("clean", "jp-us", "2016", "JP2016220164-US20160353320"),
        ("ambiguous", "jp-x-us", "2018", "JP2018163362-US20190212575"),
        ("problematic", "pct", "2017", "JP2017099237-US20190031628"),
    ]

    for s in l2["samples_summary"]:
        en_path = ROOT / s["corpus"] / s["corpus"] / s["year"] / f"{s['stem']}.en"
        ja_path = en_path.with_suffix(".ja")
        pairs, unmatched = align_patent_pair(en_path, ja_path)
        all_pairs.extend(pairs)

        n_en, n_ja = s["en_lines"], s["ja_lines"]
        counts = Counter(p["alignment_type"] for p in pairs)
        yield_pct = 100 * len(pairs) / max(n_en, n_ja) if max(n_en, n_ja) else 0

        file_results.append(
            {
                "corpus": s["corpus"],
                "year": s["year"],
                "stem": s["stem"],
                "en_lines": n_en,
                "ja_lines": n_ja,
                "pairs": len(pairs),
                "1to1": counts.get("1to1", 0),
                "one_to_many_ja": counts.get("one_to_many_ja", 0),
                "many_to_one_en": counts.get("many_to_one_en", 0),
                "cross_section": counts.get("cross_section", 0),
                "many_to_many": counts.get("many_to_many", 0),
                "unmatched_en": len(unmatched["unmatched_en"]),
                "unmatched_ja": len(unmatched["unmatched_ja"]),
                "yield_pct": round(yield_pct, 1),
            }
        )

    # Section-level yield
    section_stats = defaultdict(lambda: {"pairs": 0, "lines": 0, "unmatched": 0, "en_words": [], "ja_chars": []})
    for s in l2["samples_summary"]:
        en_path = ROOT / s["corpus"] / s["corpus"] / s["year"] / f"{s['stem']}.en"
        ja_path = en_path.with_suffix(".ja")
        pairs, unmatched = align_patent_pair(en_path, ja_path)
        en_rows = load_tsv(en_path)
        for sec in ["title", "abstract", "description", "claim"]:
            sec_lines = sum(1 for r in en_rows if section_from_id(r["id"]) == sec)
            sec_pairs = [p for p in pairs if p["section"] == sec or section_from_id(p.get("en_id", "")) == sec]
            section_stats[sec]["lines"] += sec_lines
            section_stats[sec]["pairs"] += len([p for p in pairs if section_from_id(p["en_id"] or p["ja_id"]) == sec])
            for p in pairs:
                if section_from_id(p["en_id"] or p["ja_id"]) == sec:
                    section_stats[sec]["en_words"].append(word_count_en(p["en_text"]))
                    section_stats[sec]["ja_chars"].append(ja_char_count(p["ja_text"]))

    # Filters
    kept, filter_info = apply_filters(all_pairs)
    combined_removed = len(all_pairs) - len(kept)
    raw_lines = sum(s["en_lines"] for s in l2["samples_summary"])
    net_yield = 100 * len(kept) / raw_lines if raw_lines else 0
    pre_yield = 100 * len(all_pairs) / raw_lines if raw_lines else 0

    # Duplicates within files
    dup_en = dup_ja = 0
    for s in l2["samples_summary"]:
        en_path = ROOT / s["corpus"] / s["corpus"] / s["year"] / f"{s['stem']}.en"
        ja_path = en_path.with_suffix(".ja")
        for rows, attr in [(load_tsv(en_path), "en"), (load_tsv(ja_path), "ja")]:
            texts = [r["text"] for r in rows]
            c = Counter(texts)
            dups = sum(v - 1 for v in c.values() if v > 1)
            if attr == "en":
                dup_en += dups
            else:
                dup_ja += dups

    # Concepts
    concept_counts = {}
    for en_term, ja_terms in CONCEPTS:
        en_only = ja_only = both = 0
        for p in all_pairs:
            has_en = en_term.lower() in p["en_text"].lower()
            has_ja = any(t in p["ja_text"] for t in ja_terms)
            if has_en and has_ja:
                both += 1
            elif has_en:
                en_only += 1
            elif has_ja:
                ja_only += 1
        en_total = sum(1 for p in all_pairs if en_term.lower() in p["en_text"].lower())
        ja_total = sum(1 for p in all_pairs if any(t in p["ja_text"] for t in ja_terms))
        concept_counts[en_term] = {
            "en_only": en_only,
            "ja_only": ja_only,
            "both": both,
            "en_total": en_total,
            "ja_total": ja_total,
            "both_pct_en": round(100 * both / en_total, 2) if en_total else 0,
            "both_pct_ja": round(100 * both / ja_total, 2) if ja_total else 0,
        }

    # Domain proxy per file
    domain_files = defaultdict(list)
    for s in l2["samples_summary"]:
        en_path = ROOT / s["corpus"] / s["corpus"] / s["year"] / f"{s['stem']}.en"
        ja_path = en_path.with_suffix(".ja")
        pairs, _ = align_patent_pair(en_path, ja_path)
        blob_en = " ".join(load_tsv(en_path)[i]["text"] for i in range(min(50, len(load_tsv(en_path)))))
        blob_ja = " ".join(load_tsv(ja_path)[i]["text"] for i in range(min(50, len(load_tsv(ja_path)))))
        dom = classify_domain(blob_en, blob_ja)
        domain_files[dom].append(
            {
                "yield": len(pairs) / max(s["en_lines"], 1) * 100,
                "pairs": len(pairs),
                "lines": s["en_lines"],
            }
        )

    # Stress tests
    stress_output = []
    for label, corp, year, stem in stress_cases:
        en_path = ROOT / corp / corp / year / f"{stem}.en"
        ja_path = en_path.with_suffix(".ja")
        pairs, unmatched = align_patent_pair(en_path, ja_path)
        stress_output.append(
            {
                "label": label,
                "stem": stem,
                "pairs_count": len(pairs),
                "sample_pairs": pairs[:8],
                "unmatched_en_sample": unmatched["unmatched_en"][:5],
                "unmatched_ja_sample": unmatched["unmatched_ja"][:5],
            }
        )

    # Failure mode aggregation
    total_unmatched_en = sum(r["unmatched_en"] for r in file_results)
    total_unmatched_ja = sum(r["unmatched_ja"] for r in file_results)
    total_lines = raw_lines
    failure_modes = {
        "unmatched_en": total_unmatched_en,
        "unmatched_ja": total_unmatched_ja,
        "non_1to1_aligned": sum(
            r["one_to_many_ja"] + r["many_to_one_en"] + r["cross_section"] + r["many_to_many"]
            for r in file_results
        ),
        "filtered_quality": combined_removed,
    }

    out = {
        "file_results": file_results,
        "section_stats": {
            k: {
                "yield_pct": round(100 * v["pairs"] / v["lines"], 1) if v["lines"] else 0,
                "pairs": v["pairs"],
                "lines": v["lines"],
                "avg_en_words": round(statistics.mean(v["en_words"]), 1) if v["en_words"] else 0,
                "avg_ja_chars": round(statistics.mean(v["ja_chars"]), 1) if v["ja_chars"] else 0,
            }
            for k, v in section_stats.items()
        },
        "yield": {
            "raw_lines": raw_lines,
            "pairs_before_filter": len(all_pairs),
            "pairs_after_filter": len(kept),
            "pre_filter_yield_pct": round(pre_yield, 1),
            "net_yield_pct": round(net_yield, 1),
        },
        "filter_info": filter_info,
        "duplicates": {"en": dup_en, "ja": dup_ja},
        "concept_counts": concept_counts,
        "domain_files": {k: {"n": len(v), "avg_yield": round(statistics.mean(x["yield"] for x in v), 1) if v else 0} for k, v in domain_files.items()},
        "stress_output": stress_output,
        "failure_modes": failure_modes,
        "extrapolation_factor": FULL_CORPUS_PAIRS / 40,
    }

    out_path = ROOT / "level3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps(out, indent=2, ensure_ascii=False)[:12000])
    return out


if __name__ == "__main__":
    run_level3()
