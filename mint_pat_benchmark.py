#!/usr/bin/env python3
"""Benchmark alignment speed and memory for scaling estimates."""
import time
import tracemalloc
from pathlib import Path

from mint_pat_preprocess import align_patent_pair, apply_quality_filters, discover_pairs, load_tsv

ROOT = Path(r"D:/en-ja")


def main():
    files = discover_pairs(ROOT)[:100]
    t0 = time.perf_counter()
    tracemalloc.start()
    peak = 0
    total_pairs = 0
    for fp in files:
        pairs, _ = align_patent_pair(fp.en_path, fp.ja_path)
        kept, _ = apply_quality_filters(pairs)
        total_pairs += len(kept)
        peak = max(peak, tracemalloc.get_traced_memory()[1])
    elapsed = time.perf_counter() - t0
    tracemalloc.stop()
    per_file = elapsed / len(files)
    print(f"files={len(files)} elapsed={elapsed:.2f}s per_file={per_file*1000:.1f}ms")
    print(f"pairs_kept={total_pairs} peak_tracemalloc={peak/1024/1024:.2f} MB")
    print(f"1000 files 1 core: {per_file*1000/60:.1f} min")
    print(f"1.4M files 1 core: {per_file*1_400_000/3600:.1f} hours")
    print(f"1.4M files 4 cores: {per_file*1_400_000/3600/4:.1f} hours")
    print(f"1.4M files 8 cores: {per_file*1_400_000/3600/8:.1f} hours")
    workers_16gb = max(1, int(16 / (peak / 1024 / 1024 / 512 * 1024 + 50)))
    print(f"suggested_workers_16GB~{min(8, workers_16gb)}")


if __name__ == "__main__":
    main()
