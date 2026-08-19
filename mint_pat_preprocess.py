#!/usr/bin/env python3
"""
MINT-Pat production preprocessing pipeline (JaParaPat EN-JA patent pairs).

Stages: discovery -> stratified sampling -> alignment -> quality filter -> output -> stats.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants (Level 1-3 findings)
# ---------------------------------------------------------------------------

VALID_SECTIONS = {"title", "abstract", "description", "claim", "figref"}
SKIP_PARTITIONS: set[str] = set()  # none excluded; low-yield handled per-file

JA_CJK = re.compile(r"[\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff]")
FIGURE_ONLY_JA = re.compile(r"^(図|Ｆｉｇ\.?|FIG\.?)\s*[\d０-９]+", re.I)
NMR_MARKERS = ("NMR", "1H-", "13C-", "δ:", "CDCl3", "DMSO-d")

CONCEPTS: list[tuple[str, list[str]]] = [
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

# IPC letter -> preferred JaParaPat partitions (no IPC on disk; partition proxy)
IPC_PARTITION_MAP = {
    "G": ["jp-us", "jp-x-us", "pct"],
    "H": ["jp-us", "jp-x-us", "pct"],
    "C": ["pct"],
    "F": ["jp-us", "us-jp"],
    "B": ["jp-us", "us-jp"],
}

DOC_YIELD_MIN = 0.50  # skip documents below this alignment yield

_WIN_DRIVE_PATH = re.compile(r"^([A-Za-z]):[/\\](.*)$")


def resolve_user_path(path_str: str) -> Path:
    """
    Resolve a user-supplied path, converting Windows drive paths when running under WSL/Linux.

    Examples:
      D:\\en-ja  -> /mnt/d/en-ja  (on WSL)
      D:/en-ja   -> /mnt/d/en-ja  (on WSL)
    """
    raw = path_str.strip().strip("'\"")
    p = Path(raw)

    # Already valid
    if p.exists():
        return p.resolve()

    # Windows drive letter path on Linux/WSL: D:\foo -> /mnt/d/foo
    m = _WIN_DRIVE_PATH.match(raw.replace("\\", "/") if "\\" in raw else raw)
    if m is None and "\\" in raw:
        m = _WIN_DRIVE_PATH.match(raw.replace("\\", "/"))
    if m and platform.system() != "Windows":
        drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
        wsl = Path(f"/mnt/{drive}") / rest
        if wsl.exists():
            return wsl.resolve()

    # Forward-slash variant of Windows path without regex match
    if platform.system() != "Windows" and len(raw) >= 2 and raw[1] == ":":
        drive, rest = raw[0].lower(), raw[2:].lstrip("/\\").replace("\\", "/")
        wsl = Path(f"/mnt/{drive}") / rest
        if wsl.exists():
            return wsl.resolve()

    return p.resolve()


# ---------------------------------------------------------------------------
# Alignment (Level 3)
# ---------------------------------------------------------------------------


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

    shared = set(en_by_idx) & set(ja_by_idx)
    pairs: list[dict] = []
    used_en: set[str] = set()
    used_ja: set[str] = set()

    def base_pair(atype, er, jr, idx):
        return {
            "alignment_type": atype,
            "en_id": er["id"],
            "ja_id": jr["id"],
            "en_text": er["text"],
            "ja_text": jr["text"],
            "section": section_from_id(er["id"]),
            "en_sentence_index": idx,
            "ja_sentence_index": idx,
        }

    for idx in sorted(shared):
        el = list({r["id"]: r for r in en_by_idx[idx]}.values())
        jl = list({r["id"]: r for r in ja_by_idx[idx]}.values())

        if len(el) == 1 and len(jl) == 1:
            er, jr = el[0], jl[0]
            atype = "1to1" if section_from_id(er["id"]) == section_from_id(jr["id"]) else "cross_section"
            pairs.append(base_pair(atype, er, jr, idx))
            used_en.add(er["id"])
            used_ja.add(jr["id"])
        elif len(el) == 1 and len(jl) > 1:
            er = el[0]
            pairs.append(
                {
                    "alignment_type": "one_to_many_ja",
                    "en_id": er["id"],
                    "ja_id": ",".join(j["id"] for j in jl),
                    "en_text": er["text"],
                    "ja_text": " ".join(j["text"] for j in jl),
                    "section": section_from_id(er["id"]),
                    "en_sentence_index": idx,
                    "ja_sentence_index": idx,
                }
            )
            used_en.add(er["id"])
            used_ja.update(j["id"] for j in jl)
        elif len(el) > 1 and len(jl) == 1:
            jr = jl[0]
            pairs.append(
                {
                    "alignment_type": "many_to_one_en",
                    "en_id": ",".join(e["id"] for e in el),
                    "ja_id": jr["id"],
                    "en_text": " ".join(e["text"] for e in el),
                    "ja_text": jr["text"],
                    "section": section_from_id(jr["id"]),
                    "en_sentence_index": idx,
                    "ja_sentence_index": idx,
                }
            )
            used_en.update(e["id"] for e in el)
            used_ja.add(jr["id"])
        else:
            pairs.append(
                {
                    "alignment_type": "many_to_many",
                    "en_id": ",".join(e["id"] for e in el),
                    "ja_id": ",".join(j["id"] for j in jl),
                    "en_text": " ".join(e["text"] for e in el),
                    "ja_text": " ".join(j["text"] for j in jl),
                    "section": section_from_id(el[0]["id"]),
                    "en_sentence_index": idx,
                    "ja_sentence_index": idx,
                }
            )
            used_en.update(e["id"] for e in el)
            used_ja.update(j["id"] for j in jl)

    unmatched_en = [(r["id"], r["text"]) for r in en_rows if r["id"] not in used_en]
    unmatched_ja = [
        (r["id"], r["text"])
        for r in ja_rows
        if r["id"] not in used_ja and section_from_id(r["id"]) != "figref"
    ]
    return pairs, {"unmatched_en": unmatched_en, "unmatched_ja": unmatched_ja}


# ---------------------------------------------------------------------------
# Quality filters (Level 3 production recommendations)
# ---------------------------------------------------------------------------


def word_count_en(text: str) -> int:
    return len(text.split())


def ja_char_count(text: str) -> int:
    return len(text)


def ja_cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(JA_CJK.findall(text)) / len(text)


def is_figure_only_ja(text: str) -> bool:
    t = text.strip()
    if len(t) <= 12 and FIGURE_ONLY_JA.match(t):
        return True
    if t in ("図",) or re.fullmatch(r"[図Ｆｉｇ\.?\s\d]+", t):
        return True
    return False


def is_nmr_or_chemistry_ja(text: str) -> bool:
    return any(m in text for m in NMR_MARKERS)


def filter_pair(pair: dict) -> tuple[bool, str | None]:
    """Return (keep, reject_reason)."""
    en_t, ja_t = pair["en_text"], pair["ja_text"]
    sec = pair.get("section") or section_from_id(pair.get("en_id", ""))

    if pair["alignment_type"] == "cross_section":
        return False, "cross_section"
    if pair["alignment_type"] == "many_to_many":
        return False, "many_to_many"

    if is_figure_only_ja(ja_t):
        return False, "figure_only_ja"

    if sec != "title":
        if word_count_en(en_t) < 5:
            return False, "min_length_en"
        if ja_char_count(ja_t) < 10:
            return False, "min_length_ja"

    if not is_nmr_or_chemistry_ja(ja_t) and ja_cjk_ratio(ja_t) < 0.2:
        return False, "ja_density"

    if word_count_en(en_t) > 350 or ja_char_count(ja_t) > 1800:
        return False, "max_length"

    # Level 3: do NOT apply cross-language character length_ratio (omitted intentionally)
    return True, None


def apply_quality_filters(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, rejected = [], []
    for p in pairs:
        ok, reason = filter_pair(p)
        if ok:
            kept.append(p)
        else:
            rejected.append({**p, "reject_reason": reason})
    return kept, rejected


def pair_has_concept(pair: dict) -> tuple[str, str] | None:
    for en_term, ja_terms in CONCEPTS:
        if en_term.lower() in pair["en_text"].lower() and any(t in pair["ja_text"] for t in ja_terms):
            return en_term, ja_terms[0]
    return None


# ---------------------------------------------------------------------------
# Stage 1: Discovery
# ---------------------------------------------------------------------------


@dataclass
class FilePair:
    en_path: Path
    ja_path: Path
    partition: str
    year: str
    stem: str

    @property
    def key(self) -> str:
        return str(self.en_path)


def discover_pairs(data_root: Path, domain_ipc: str | None = None) -> list[FilePair]:
    pairs: list[FilePair] = []
    allowed_partitions: set[str] | None = None
    if domain_ipc:
        letter = domain_ipc.upper()[:1]
        allowed_partitions = set(IPC_PARTITION_MAP.get(letter, []))

    for en_path in data_root.rglob("*.en"):
        if en_path.suffix != ".en":
            continue
        ja_path = en_path.with_suffix(".ja")
        if not ja_path.is_file():
            continue
        parts = en_path.relative_to(data_root).parts
        partition = parts[0] if parts else "unknown"
        if partition in SKIP_PARTITIONS:
            continue
        if allowed_partitions and partition not in allowed_partitions:
            continue
        year = parts[2] if len(parts) >= 3 and parts[1] == partition else "unknown"
        pairs.append(
            FilePair(
                en_path=en_path,
                ja_path=ja_path,
                partition=partition,
                year=year,
                stem=en_path.stem,
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Stage 2: Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    all_pairs: list[FilePair],
    target_parallel_pairs: int,
    seed: int,
    dry_run_max_files: int | None = None,
) -> list[FilePair]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[FilePair]] = defaultdict(list)
    for fp in all_pairs:
        by_stratum[(fp.partition, fp.year)].append(fp)

    for key in by_stratum:
        rng.shuffle(by_stratum[key])

    strata = list(by_stratum.keys())
    rng.shuffle(strata)
    selected: list[FilePair] = []
    # Round-robin across (partition, year) for equal stratum weight
    indices = {k: 0 for k in strata}
    est_pairs_per_doc = 230  # Level 3 filtered average
    max_files = dry_run_max_files
    if max_files is None:
        max_files = max(1, int(target_parallel_pairs / est_pairs_per_doc) + 500)
    else:
        max_files = min(max_files, len(all_pairs))

    while len(selected) < max_files and strata:
        progressed = False
        for s in strata:
            idx = indices[s]
            bucket = by_stratum[s]
            if idx < len(bucket):
                selected.append(bucket[idx])
                indices[s] += 1
                progressed = True
                if len(selected) >= max_files:
                    break
        if not progressed:
            break
    return selected


# ---------------------------------------------------------------------------
# Stage 5-6: Output + stats
# ---------------------------------------------------------------------------


@dataclass
class PipelineStats:
    files_discovered: int = 0
    files_sampled: int = 0
    files_processed: int = 0
    files_skipped_resume: int = 0
    files_skipped_low_yield: int = 0
    files_error: int = 0
    raw_lines: int = 0
    aligned_pairs: int = 0
    filtered_pairs: int = 0
    rejected_pairs: int = 0
    concept_pairs: int = 0
    mlm_lines: int = 0
    filter_reasons: Counter = field(default_factory=Counter)
    en_word_lengths: list[int] = field(default_factory=list)
    ja_char_lengths: list[int] = field(default_factory=list)
    concept_counts: Counter = field(default_factory=Counter)


def assign_splits(stems: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    mapping = {}
    for i, s in enumerate(shuffled):
        if i < n_train:
            mapping[s] = "train"
        elif i < n_train + n_valid:
            mapping[s] = "valid"
        else:
            mapping[s] = "test"
    return mapping


def write_mlm_lines(path: Path, lines: list[str], mode: str = "a") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\t", " ").replace("\n", " ") + "\n")


def process_pipeline(args: argparse.Namespace) -> PipelineStats:
    data_root = resolve_user_path(args.data_root)
    output_dir = resolve_user_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        print(
            f"ERROR: data_root does not exist: {data_root}\n"
            f"  (from --data_root {args.data_root!r})\n"
            "  On WSL, use: --data_root /mnt/d/en-ja",
            file=sys.stderr,
        )
        sys.exit(1)

    sample_en = list(data_root.glob("*/*/*.en"))[:1]
    if not sample_en and not list(data_root.rglob("*.en"))[:1]:
        print(
            f"ERROR: no .en/.ja pairs found under {data_root}\n"
            "  Expected layout: {root}/jp-us/jp-us/2016/*.en",
            file=sys.stderr,
        )
        sys.exit(1)

    log_path = output_dir / "processed_files.log"
    error_log = output_dir / "errors.log"
    processed = set()
    if log_path.is_file():
        processed = {ln.strip() for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()}

    stats = PipelineStats()
    all_files = discover_pairs(data_root, args.domain)
    stats.files_discovered = len(all_files)

    if args.dry_run:
        dry_max = 1000
    elif args.max_files:
        dry_max = args.max_files
    else:
        dry_max = None
    sampled = stratified_sample(all_files, args.target_pairs, args.seed, dry_max)
    stats.files_sampled = len(sampled)

    unique_stems = [fp.stem for fp in sampled]
    split_map = assign_splits(unique_stems, args.seed)

    mlm_paths = {
        "train": output_dir / "train.txt",
        "valid": output_dir / "valid.txt",
        "test": output_dir / "test.txt",
    }
    if not args.dry_run and not processed:
        for p in mlm_paths.values():
            p.write_text("", encoding="utf-8")

    concept_train_path = output_dir / "concept_pairs_train.tsv"
    concept_eval_path = output_dir / "concept_pairs_eval.tsv"
    concept_header = ["en_text", "ja_text", "section", "alignment_type", "concept_en", "source_file"]

    seen_pair_en: set[str] = set()
    seen_pair_ja: set[str] = set()
    seen_mlm_hashes: set[str] = set()
    concept_rows: list[dict] = []

    t0 = time.perf_counter()

    for i, fp in enumerate(sampled, 1):
        if fp.key in processed:
            stats.files_skipped_resume += 1
            continue
        try:
            en_rows = load_tsv(fp.en_path)
            stats.raw_lines += len(en_rows)
            aligned, _unm = align_patent_pair(fp.en_path, fp.ja_path)
            stats.aligned_pairs += len(aligned)

            yield_rate = len(aligned) / max(len(en_rows), 1)
            if yield_rate < DOC_YIELD_MIN:
                stats.files_skipped_low_yield += 1
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(fp.key + "\n")
                continue

            kept, rejected = apply_quality_filters(aligned)
            stats.filtered_pairs += len(kept)
            stats.rejected_pairs += len(rejected)
            for r in rejected:
                stats.filter_reasons[r.get("reject_reason", "unknown")] += 1

            split = split_map.get(fp.stem, "train")
            mlm_batch: list[str] = []

            for p in kept:
                en_t, ja_t = p["en_text"].strip(), p["ja_text"].strip()
                if not en_t or not ja_t:
                    continue
                if en_t in seen_pair_en and ja_t in seen_pair_ja:
                    continue
                seen_pair_en.add(en_t)
                seen_pair_ja.add(ja_t)
                stats.en_word_lengths.append(word_count_en(en_t))
                stats.ja_char_lengths.append(ja_char_count(ja_t))
                for line in (en_t, ja_t):
                    h = hashlib.md5(line.encode("utf-8")).hexdigest()
                    if h not in seen_mlm_hashes:
                        seen_mlm_hashes.add(h)
                        mlm_batch.append(line)

                if p["alignment_type"] == "1to1":
                    concept = pair_has_concept(p)
                    if concept:
                        en_c, _ja_c = concept
                        stats.concept_counts[en_c] += 1
                        concept_rows.append(
                            {
                                "en_text": en_t,
                                "ja_text": ja_t,
                                "section": p["section"],
                                "alignment_type": p["alignment_type"],
                                "concept_en": en_c,
                                "source_file": fp.key,
                            }
                        )

            stats.mlm_lines += len(mlm_batch)
            if not args.dry_run and mlm_batch:
                write_mlm_lines(mlm_paths[split], mlm_batch, "a")

            stats.files_processed += 1
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(fp.key + "\n")

        except Exception as e:
            stats.files_error += 1
            with open(error_log, "a", encoding="utf-8") as ef:
                ef.write(f"{fp.key}\t{type(e).__name__}: {e}\n")
                ef.write(traceback.format_exc() + "\n")

        if i % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"[{i}/{len(sampled)}] processed={stats.files_processed} "
                f"pairs={stats.filtered_pairs} errors={stats.files_error} "
                f"({rate:.1f} files/s)",
                flush=True,
            )

    stats.concept_pairs = len(concept_rows)
    if not args.dry_run and concept_rows:
        rng = random.Random(args.seed)
        rng.shuffle(concept_rows)
        n = len(concept_rows)
        n_train = int(n * 0.8)
        n_eval = max(1, n - n_train) if n > 1 else n
        with open(concept_train_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=concept_header, delimiter="\t")
            w.writeheader()
            w.writerows(concept_rows[:n_train])
        with open(concept_eval_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=concept_header, delimiter="\t")
            w.writeheader()
            w.writerows(concept_rows[n_train:])

    # Data card + stats JSON
    write_data_card(output_dir, stats, args)
    return stats


def write_data_card(output_dir: Path, stats: PipelineStats, args: argparse.Namespace) -> None:
    import statistics as stat_mod

    pre_yield = 100 * stats.aligned_pairs / stats.raw_lines if stats.raw_lines else 0
    net_yield = 100 * stats.filtered_pairs / stats.raw_lines if stats.raw_lines else 0
    avg_en = stat_mod.mean(stats.en_word_lengths) if stats.en_word_lengths else 0
    std_en = stat_mod.stdev(stats.en_word_lengths) if len(stats.en_word_lengths) > 1 else 0
    avg_ja = stat_mod.mean(stats.ja_char_lengths) if stats.ja_char_lengths else 0
    std_ja = stat_mod.stdev(stats.ja_char_lengths) if len(stats.ja_char_lengths) > 1 else 0

    card = {
        "dataset_name": "MINT-Pat Patent Pretraining Corpus v1.0",
        "source_corpus": "JaParaPat (Nagata et al., LREC-COLING 2024)",
        "languages": ["en", "ja"],
        "year_range": "2016-2020",
        "partitions": ["jp-us", "jp-x-us", "pct", "us-jp"],
        "alignment_method": "Trailing sentence-index matching with 1to1/cross_section/merge handling",
        "quality_filters": [
            "exclude cross_section and many_to_many",
            "exclude JA figure-only captions",
            "section-aware min length (EN>=5 words, JA>=10 chars; title exempt)",
            "JA CJK density >= 0.2 (NMR/chemistry whitelist)",
            "max EN 350 words / JA 1800 chars",
            "document yield gate >= 50%",
            "exact duplicate removal on EN and JA text",
            "NOTE: cross-language char length_ratio NOT applied (Level 3)",
        ],
        "alignment_yield_pct": round(pre_yield, 2),
        "net_yield_pct": round(net_yield, 2),
        "mlm_lines": stats.mlm_lines,
        "parallel_pairs_kept": stats.filtered_pairs // 1,  # pairs not lines
        "concept_pairs": stats.concept_pairs,
        "split_ratio": "80/10/10 by document",
        "avg_en_words": round(avg_en, 2),
        "std_en_words": round(std_en, 2),
        "avg_ja_chars": round(avg_ja, 2),
        "std_ja_chars": round(std_ja, 2),
        "concept_counts": dict(stats.concept_counts),
        "filter_rejections": dict(stats.filter_reasons),
        "files_processed": stats.files_processed,
        "files_skipped_low_yield": stats.files_skipped_low_yield,
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# MINT-Pat Patent Pretraining Corpus v1.0 — Data Card

## Description
Parallel English–Japanese patent sentence pairs derived from JaParaPat (2016–2020 release),
aligned by shared sentence index, filtered for MLM continued pre-training of XLM-RoBERTa-base
and concept-focused evaluation TSVs for CKA/SAE analysis.

## Source
- **Corpus:** JaParaPat (Nagata et al., LREC-COLING 2024)
- **Original document pairs:** ~370,835 (four partitions)
- **Languages:** English, Japanese
- **Domain:** JPO/USPTO patent applications
- **Year range:** 2016–2020

## Processing
- **Alignment:** `align_patent_pair()` — index-based matching (Level 3)
- **Alignment yield:** {pre_yield:.1f}%
- **Net yield after filtering:** {net_yield:.1f}%
- **Filters:** see `dataset_stats.json`

## Final statistics (this run)
- **MLM lines (EN+JA):** {stats.mlm_lines:,}
- **Parallel pairs kept:** {stats.filtered_pairs:,}
- **Concept pair rows:** {stats.concept_pairs:,}
- **Train/valid/test:** 80/10/10 by document
- **Avg EN length:** {avg_en:.1f} words (σ={std_en:.1f})
- **Avg JA length:** {avg_ja:.1f} chars (σ={std_ja:.1f})

## Known limitations
- IPC codes not in corpus files; partition used as weak domain proxy
- Index alignment can mismatch paragraph offsets (semantically wrong 1to1)
- Chemistry patents may have low document yield
- `prior art` / `novelty` rare in aligned sample

## Intended use
Research-only continued pre-training for MINT-Pat mechanistic interpretability (NTT JaParaPat license).

## License
JaParaPat NTT Terms of Use — research/non-enjoyment only; no commercial translator sale.
"""
    (output_dir / "DATA_CARD.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Concept candidate extraction (Part C2)
# ---------------------------------------------------------------------------


def extract_concept_pairs(
    concept_eval_tsv: Path,
    output_path: Path,
    max_per_concept: int = 50,
) -> int:
    """Extract ranked concept evaluation candidates from pipeline TSV output."""
    if not concept_eval_tsv.is_file():
        raise FileNotFoundError(concept_eval_tsv)

    rows = []
    with open(concept_eval_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    candidates: list[dict] = []
    for en_term, ja_terms in CONCEPTS:
        pool = []
        for row in rows:
            if row.get("alignment_type") != "1to1":
                continue
            en_t, ja_t = row["en_text"], row["ja_text"]
            if en_term.lower() not in en_t.lower():
                continue
            if not any(t in ja_t for t in ja_terms):
                continue
            wc = word_count_en(en_t)
            sec = row.get("section", "")
            sec_bonus = 2.0 if sec == "description" else (1.0 if sec == "abstract" else 0.5)
            len_score = 1.0 if 15 <= wc <= 100 else (0.5 if 5 <= wc < 15 else 0.2)
            score = sec_bonus * len_score
            pool.append(
                {
                    "concept_en": en_term,
                    "concept_ja": ja_terms[0],
                    "section": sec,
                    "en_text": en_t,
                    "ja_text": ja_t,
                    "source_file": row.get("source_file", ""),
                    "alignment_type": row["alignment_type"],
                    "rank_score": round(score, 3),
                }
            )
        pool.sort(key=lambda x: -x["rank_score"])
        candidates.extend(pool[:max_per_concept])

    fields = [
        "concept_en",
        "concept_ja",
        "section",
        "en_text",
        "ja_text",
        "source_file",
        "alignment_type",
        "rank_score",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(candidates)
    return len(candidates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MINT-Pat JaParaPat preprocessing pipeline")
    p.add_argument("--data_root", type=str, default=r"D:\en-ja", help="Corpus root")
    p.add_argument("--output_dir", type=str, default=r"D:\en-ja\mint_pat_output", help="Output directory")
    p.add_argument("--target_pairs", type=int, default=1_000_000, help="Target parallel pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--domain", type=str, default=None, help="IPC section letter proxy (G,H,C,F,B)")
    p.add_argument("--dry_run", action="store_true", help="Process at most 1000 files; stats only, no MLM writes")
    p.add_argument("--max_files", type=int, default=None, help="Cap number of files to process")
    p.add_argument(
        "--extract_concepts",
        action="store_true",
        help="Only run extract_concept_pairs on existing concept_pairs_eval.tsv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract_concepts:
        out = resolve_user_path(args.output_dir) / "concept_candidates.tsv"
        n = extract_concept_pairs(
            resolve_user_path(args.output_dir) / "concept_pairs_eval.tsv",
            out,
        )
        print(f"Wrote {n} concept candidates to {out}")
        return

    print("MINT-Pat preprocess starting...", flush=True)
    data_root = resolve_user_path(args.data_root)
    output_dir = resolve_user_path(args.output_dir)
    print(f"  data_root:  {data_root}", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    stats = process_pipeline(args)
    print("\n=== Pipeline complete ===")
    print(f"Discovered: {stats.files_discovered}")
    print(f"Sampled: {stats.files_sampled}")
    print(f"Processed: {stats.files_processed}")
    print(f"Skipped (resume): {stats.files_skipped_resume}")
    print(f"Skipped (low yield): {stats.files_skipped_low_yield}")
    print(f"Errors: {stats.files_error}")
    print(f"Aligned pairs: {stats.aligned_pairs}")
    print(f"After filter: {stats.filtered_pairs}")
    print(f"MLM lines: {stats.mlm_lines}")
    print(f"Concept rows: {stats.concept_pairs}")
    if stats.files_discovered == 0:
        print("\nNo files discovered — nothing to do.", file=sys.stderr)
        sys.exit(1)
    if not args.dry_run:
        eval_tsv = resolve_user_path(args.output_dir) / "concept_pairs_eval.tsv"
        if eval_tsv.is_file():
            n = extract_concept_pairs(eval_tsv, resolve_user_path(args.output_dir) / "concept_candidates.tsv")
            print(f"Wrote {n} concept candidates to concept_candidates.tsv")
        else:
            print("Skipping concept_candidates.tsv (no concept_pairs_eval.tsv produced).")


if __name__ == "__main__":
    main()
