#!/usr/bin/env python3
"""MINT-Pat Level 2: Stratified sampling and format analysis."""
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
ROOT = Path(r"D:/en-ja")
CORPORA = ["jp-us", "jp-x-us", "pct", "us-jp"]
YEARS_FIXED = ["2016", "2018", "2020"]  # earliest, middle, most recent
JA_RE = re.compile(r"[\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff]")
ID_PART = re.compile(r"^([a-zA-Z_]+)_(\d+)_([\d,]+)$")


def load_tsv(path):
    rows = []
    malformed = {"0_col": 0, "1_col": 0, "3plus_col": 0, "no_tab": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n\r")
            if not raw:
                continue
            if "\t" not in raw:
                malformed["no_tab"] += 1
                rows.append({"id": raw, "text": "", "nf": 1})
                continue
            parts = raw.split("\t")
            nf = len(parts)
            if nf == 2:
                sid, text = parts[0], parts[1]
            elif nf == 1:
                malformed["1_col"] += 1
                sid, text = parts[0], ""
            elif nf > 2:
                malformed["3plus_col"] += 1
                sid, text = parts[0], "\t".join(parts[1:])
            else:
                malformed["0_col"] += 1
                sid, text = "", ""
            rows.append({"id": sid, "text": text, "nf": nf})
    return rows, malformed


def section_from_id(sid):
    if "_" in sid:
        return sid.split("_")[0]
    return sid


def trailing_indices(sid):
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


def analyze_pair(en_path, ja_path, meta):
    en_rows, en_mal = load_tsv(en_path)
    ja_rows, ja_mal = load_tsv(ja_path)

    en_ids = [r["id"] for r in en_rows]
    ja_ids = [r["id"] for r in ja_rows]
    en_sections = Counter(section_from_id(i) for i in en_ids)
    ja_sections = Counter(section_from_id(i) for i in ja_ids)

    en_trailing = set()
    ja_trailing = set()
    en_comma = ja_comma = 0
    for sid in en_ids:
        if "," in sid.rsplit("_", 1)[-1]:
            en_comma += 1
        en_trailing.update(trailing_indices(sid))
    for sid in ja_ids:
        if "," in sid.rsplit("_", 1)[-1]:
            ja_comma += 1
        ja_trailing.update(trailing_indices(sid))

    id_overlap = len(set(en_ids) & set(ja_ids))
    en_idx_pct = 100 * len(en_trailing & ja_trailing) / len(en_trailing) if en_trailing else 0
    ja_idx_pct = 100 * len(en_trailing & ja_trailing) / len(ja_trailing) if ja_trailing else 0

    def spot(rows, lang):
        empty = xml = num_only = wrong_script = 0
        for r in rows:
            t = r["text"].strip()
            if not t:
                empty += 1
            if "<" in t or ">" in t:
                xml += 1
            if t and re.fullmatch(r"[\d\s\.,;:+\-*/=()\[\]{}%°]+", t):
                num_only += 1
            has_ja = bool(JA_RE.search(t))
            if lang == "ja" and not has_ja and t:
                wrong_script += 1
            if lang == "en" and has_ja:
                wrong_script += 1
        return {
            "empty": empty,
            "xml": xml,
            "num_only": num_only,
            "wrong_script": wrong_script,
        }

    en_spot = spot(en_rows, "en")
    ja_spot = spot(ja_rows, "ja")

    id_format_ok = all(ID_PART.match(i) or "_" in i for i in en_ids[:50])

    section_ratios = {}
    all_secs = set(en_sections) | set(ja_sections)
    for s in sorted(all_secs):
        ec, jc = en_sections.get(s, 0), ja_sections.get(s, 0)
        section_ratios[s] = round(ec / jc, 3) if jc else float("inf")

    line_ratio = len(en_rows) / len(ja_rows) if ja_rows else 0
    approx_align = round((en_idx_pct + ja_idx_pct) / 2, 1)

    anomalies = []
    if len(en_rows) != len(ja_rows):
        anomalies.append(f"line_count_mismatch:{len(en_rows)}vs{len(ja_rows)}")
    if en_mal["no_tab"] or ja_mal["no_tab"] or en_mal["3plus_col"] or ja_mal["3plus_col"]:
        anomalies.append("malformed_columns")
    if en_spot["xml"] or ja_spot["xml"]:
        anomalies.append(f"xml_tags:en{en_spot['xml']}/ja{ja_spot['xml']}")
    if en_spot["wrong_script"] or ja_spot["wrong_script"]:
        anomalies.append(f"script_mix:en{en_spot['wrong_script']}/ja{ja_spot['wrong_script']}")
    if abs(line_ratio - 1) > 0.15:
        anomalies.append(f"line_ratio:{line_ratio:.2f}")
    if approx_align < 70:
        anomalies.append(f"low_index_overlap:{approx_align}%")

    return {
        **meta,
        "en_path": str(en_path),
        "ja_path": str(ja_path),
        "en_bytes": en_path.stat().st_size,
        "ja_bytes": ja_path.stat().st_size,
        "en_lines": len(en_rows),
        "ja_lines": len(ja_rows),
        "lines_match": len(en_rows) == len(ja_rows),
        "en_malformed": en_mal,
        "ja_malformed": ja_mal,
        "en_sections": dict(en_sections),
        "ja_sections": dict(ja_sections),
        "section_ratios": section_ratios,
        "en_comma_ids": en_comma,
        "ja_comma_ids": ja_comma,
        "en_trailing_min": min(en_trailing) if en_trailing else None,
        "en_trailing_max": max(en_trailing) if en_trailing else None,
        "ja_trailing_min": min(ja_trailing) if ja_trailing else None,
        "ja_trailing_max": max(ja_trailing) if ja_trailing else None,
        "en_trailing_count": len(en_trailing),
        "ja_trailing_count": len(ja_trailing),
        "en_idx_overlap_pct": round(en_idx_pct, 1),
        "ja_idx_overlap_pct": round(ja_idx_pct, 1),
        "approx_align_pct": approx_align,
        "identical_id_count": id_overlap,
        "id_format_ok": id_format_ok,
        "en_spot": en_spot,
        "ja_spot": ja_spot,
        "line_ratio": round(line_ratio, 3),
        "anomalies": anomalies,
        "en_rows": en_rows,
        "ja_rows": ja_rows,
    }


def sample_corpus(corpus):
    base = ROOT / corpus / corpus
    rng = random.Random(SEED + hash(corpus) % 1000)
    extra_years = rng.sample(["2017", "2019"], 2)
    years = YEARS_FIXED + extra_years
    pairs = []
    for year in years:
        ydir = base / year
        if not ydir.is_dir():
            continue
        en_files = sorted(ydir.glob("*.en"))
        if len(en_files) < 2:
            chosen = en_files
        else:
            chosen = rng.sample(en_files, 2)
        for en_f in chosen:
            ja_f = en_f.with_suffix(".ja")
            if ja_f.exists():
                pairs.append(
                    {
                        "corpus": corpus,
                        "year": year,
                        "stem": en_f.stem,
                        "en_file": en_f.name,
                    }
                )
    return pairs, years


def main():
    rng = random.Random(SEED)
    all_samples = []
    sampling_log = {}

    for corp in CORPORA:
        pairs, years = sample_corpus(corp)
        sampling_log[corp] = {
            "years_sampled": years,
            "method": "2 random .en per year from {2016,2018,2020} + 2 extra years from {2017,2019}",
            "n_pairs": len(pairs),
        }
        for p in pairs:
            en_path = ROOT / corp / corp / p["year"] / p["en_file"]
            ja_path = en_path.with_suffix(".ja")
            all_samples.append(analyze_pair(en_path, ja_path, p))

    # Aggregate stats
    section_global = Counter()
    for s in all_samples:
        for k, v in s["en_sections"].items():
            section_global[k] += v

    align_pcts = [s["approx_align_pct"] for s in all_samples]
    line_ratios = [s["line_ratio"] for s in all_samples]

    # Pick 3 deep-dive: clean, ambiguous, problematic
    clean = min(all_samples, key=lambda x: (len(x["anomalies"]), abs(x["line_ratio"] - 1)))
    problematic = max(
        all_samples,
        key=lambda x: (len(x["anomalies"]), abs(x["line_ratio"] - 1), -x["approx_align_pct"]),
    )
    ambiguous = sorted(
        all_samples,
        key=lambda x: abs(x["line_ratio"] - 1) + (100 - x["approx_align_pct"]) / 100,
    )[len(all_samples) // 2]

    deep_dives = [
        ("clean", clean),
        ("ambiguous", ambiguous),
        ("problematic", problematic),
    ]

    out = {
        "seed": SEED,
        "sampling_log": sampling_log,
        "n_pairs": len(all_samples),
        "samples_summary": [
            {
                k: s[k]
                for k in [
                    "corpus",
                    "year",
                    "stem",
                    "en_lines",
                    "ja_lines",
                    "lines_match",
                    "en_sections",
                    "ja_sections",
                    "approx_align_pct",
                    "line_ratio",
                    "en_comma_ids",
                    "ja_comma_ids",
                    "identical_id_count",
                    "anomalies",
                    "section_ratios",
                    "en_spot",
                    "ja_spot",
                    "en_malformed",
                    "ja_malformed",
                ]
            }
            for s in all_samples
        ],
        "align_pct_range": [min(align_pcts), max(align_pcts)],
        "line_ratio_range": [min(line_ratios), max(line_ratios)],
        "section_global": dict(section_global),
        "deep_dive_stems": [(t, s["corpus"], s["year"], s["stem"]) for t, s in deep_dives],
    }

    # Write JSON (without full rows for size)
    out_path = ROOT / "level2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Write deep dive text
    dd_path = ROOT / "level2_deep_dives.txt"
    with open(dd_path, "w", encoding="utf-8") as f:
        f.write(f"Level 2 Deep Dives (seed={SEED})\n\n")
        for label, s in deep_dives:
            f.write("=" * 80 + "\n")
            f.write(f"{label.upper()}: {s['corpus']}/{s['year']}/{s['stem']}\n")
            f.write(f"Anomalies: {s['anomalies']}\n")
            f.write(f"Align%: {s['approx_align_pct']}  Line ratio: {s['line_ratio']}\n\n")
            for lang, rows in [("EN", s["en_rows"]), ("JA", s["ja_rows"])]:
                f.write(f"--- First 15 {lang} ---\n")
                for r in rows[:15]:
                    f.write(f"{r['id']}\t{r['text']}\n")
                f.write(f"\n--- Last 5 {lang} ---\n")
                for r in rows[-5:]:
                    f.write(f"{r['id']}\t{r['text']}\n")
                f.write("\n")
            # anomalous lines
            f.write("--- Flagged lines ---\n")
            for lang, rows in [("EN", s["en_rows"]), ("JA", s["ja_rows"])]:
                for r in rows:
                    t = r["text"].strip()
                    flags = []
                    if not t:
                        flags.append("empty")
                    if "<" in t or ">" in t:
                        flags.append("xml")
                    if lang == "EN" and JA_RE.search(t):
                        flags.append("ja_in_en")
                    if lang == "JA" and t and not JA_RE.search(t):
                        flags.append("no_ja")
                    if flags:
                        f.write(f"[{lang}:{','.join(flags)}] {r['id']}\t{t[:120]}\n")
            f.write("\n")

    print(json.dumps(out, indent=2, ensure_ascii=False)[:8000])
    print(f"\n... full results: {out_path}")
    print(f"... deep dives: {dd_path}")


if __name__ == "__main__":
    main()
