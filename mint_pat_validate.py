#!/usr/bin/env python3
"""Validation suite for MINT-Pat preprocessing pipeline outputs."""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

JA_CJK = re.compile(r"[\u3000-\u9fff]")
ASCII_ONLY = re.compile(r"^[\x00-\x7F]+$")
VALID_SECTIONS = {"title", "abstract", "description", "claim", "figref"}

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


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n\r") for ln in f]


def check_v1(output_dir: Path) -> CheckResult:
  train = output_dir / "train.txt"
  valid = output_dir / "valid.txt"
  test = output_dir / "test.txt"
  if not all(p.is_file() for p in (train, valid, test)):
    return CheckResult("V1", False, "Missing train/valid/test.txt")
  lines = {k: read_lines(p) for k, p in [("train", train), ("valid", valid), ("test", test)]}
  empty = []
  tabs = []
  for name, ln in lines.items():
    for i, line in enumerate(ln):
      if not line.strip():
        empty.append(f"{name}:{i+1}")
      if "\t" in line:
        tabs.append(f"{name}:{i+1}")
  total = sum(len(v) for v in lines.values())
  if empty or tabs:
    return CheckResult(
      "V1",
      False,
      f"total_lines={total} empty={len(empty)} tabs={len(tabs)} examples={empty[:3]}{tabs[:3]}",
    )
  return CheckResult("V1", True, f"total_lines={total} (train={len(lines['train'])}, valid={len(lines['valid'])}, test={len(lines['test'])})")


def check_v2(output_dir: Path, sample_n: int = 1000) -> CheckResult:
    train = output_dir / "train.txt"
    if not train.is_file():
        return CheckResult("V2", False, "train.txt missing")
    lines = read_lines(train)
    if not lines:
        return CheckResult("V2", False, "train.txt empty")
    rng = random.Random(42)
    sample = rng.sample(lines, min(sample_n, len(lines)))
    ja = sum(1 for ln in sample if JA_CJK.search(ln))
    en = sum(1 for ln in sample if ASCII_ONLY.match(ln.strip()) and ln.strip())
    other = len(sample) - ja - en
    ja_pct = 100 * ja / len(sample)
    en_pct = 100 * en / len(sample)
    ok = 40 <= ja_pct <= 60 and 40 <= en_pct <= 60
    return CheckResult(
      "V2",
      ok,
      f"sample={len(sample)} ja_like={ja_pct:.1f}% ascii_only={en_pct:.1f}% other={other}",
    )


def check_v3(output_dir: Path) -> CheckResult:
    path = output_dir / "concept_pairs_eval.tsv"
    if not path.is_file():
        return CheckResult("V3", False, "concept_pairs_eval.tsv missing")
    failures = []
    with open(path, encoding="utf-8") as f:
      header = f.readline().rstrip("\n").split("\t")
      for i, line in enumerate(f, 2):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
          failures.append(f"row{i}:cols")
          continue
        row = dict(zip(header, parts))
        en_t, ja_t, sec = row.get("en_text", ""), row.get("ja_text", ""), row.get("section", "")
        if not en_t or not ja_t:
          failures.append(f"row{i}:empty")
        if sec and sec not in VALID_SECTIONS:
          failures.append(f"row{i}:section={sec}")
        matched = False
        for en_term, ja_terms in CONCEPTS:
          if en_term.lower() in en_t.lower() and any(t in ja_t for t in ja_terms):
            matched = True
            break
        if not matched:
          failures.append(f"row{i}:no_concept")
        if len(failures) >= 10:
          break
    return CheckResult("V3", len(failures) == 0, f"failures={len(failures)} {failures[:5]}")


def check_v4(output_dir: Path) -> CheckResult:
    issues = []
    train = output_dir / "train.txt"
    if train.is_file():
      lines = read_lines(train)
      c = {}
      for ln in lines:
        c[ln] = c.get(ln, 0) + 1
      dups = sum(v - 1 for v in c.values() if v > 1)
      if dups:
        issues.append(f"train.txt dups={dups}")
    cep = output_dir / "concept_pairs_eval.tsv"
    if cep.is_file():
      rows = []
      with open(cep, encoding="utf-8") as f:
        next(f)
        for line in f:
          rows.append(line)
      c = Counter(rows)
      dups = sum(v - 1 for v in c.values() if v > 1)
      if dups:
        issues.append(f"concept_pairs_eval dups={dups}")
    return CheckResult("V4", not issues, "; ".join(issues) or "no exact duplicates")


def check_v5(output_dir: Path) -> CheckResult:
    def hashes(path: Path) -> set[str]:
        return {hashlib.md5(ln.encode("utf-8")).hexdigest() for ln in read_lines(path)}

    train_h = hashes(output_dir / "train.txt")
    leaks = []
    for name in ("valid.txt", "test.txt"):
        p = output_dir / name
        if not p.is_file():
            continue
        for ln in read_lines(p):
            h = hashlib.md5(ln.encode("utf-8")).hexdigest()
            if h in train_h:
                leaks.append(f"{name}")
                break
    return CheckResult("V5", not leaks, f"leakage_in={leaks}" if leaks else "no split leakage")


def check_v6(output_dir: Path) -> CheckResult:
    bad = []
    for name in ("train.txt", "valid.txt", "test.txt", "concept_pairs_eval.tsv", "concept_pairs_train.tsv"):
        p = output_dir / name
        if not p.is_file():
            continue
        try:
            p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            bad.append(f"{name}:{e}")
    return CheckResult("V6", not bad, "; ".join(bad) or "all UTF-8 valid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=r"D:\en-ja\mint_pat_output")
    args = ap.parse_args()
    out = Path(args.output_dir)

    checks = [
        check_v1(out),
        check_v2(out),
        check_v3(out),
        check_v4(out),
        check_v5(out),
        check_v6(out),
    ]
    print("MINT-Pat validation report")
    print("=" * 60)
    all_ok = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.name}: {c.detail}")
        all_ok = all_ok and c.passed
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
