"""
Monte Carlo estimate of the expected nDCG@10 under random ranking for the
MINT-Pat cross-lingual retrieval experiment (Module 4).

Relevance structure taken from eval_set.tsv: 597 queries; for each query the
candidate pool is the full set of 597 target-language sentences, and exactly
ONE candidate (the query's own aligned translation) is relevant. This mirrors
run_retrieval()/ndcg_at_k() in mint_pat_validate_experiments.py, where
correct_idx = qi and relevance is binary.

Method: for each query, draw 1,000 independent uniform-random permutations of
the 597 candidates and score nDCG@10 with the SAME dcg/ndcg implementation as
the thesis experiment script. Also computes the closed-form expectation as a
cross-check:  E[nDCG@10] = (1/N) * sum_{i=1..10} 1/log2(i+1)   for N=597.
"""
import csv, math, sys
import numpy as np

EVAL = "/sessions/modest-focused-mccarthy/mnt/FYP/output/eval_set.tsv"
K = 10
N_SHUFFLES = 1000
SEED = 42

# --- identical metric implementation to mint_pat_validate_experiments.py ---
def dcg_at_k(relevances, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))

def ndcg_at_k(rankings, correct_idx, k=10):
    rel = [1 if r == correct_idx else 0 for r in rankings[:k]]
    idcg = dcg_at_k([1] + [0] * (k - 1), k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0

# --- load the actual relevance structure -----------------------------------
with open(EVAL, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
n = len(rows)
ja_texts = [r["ja_text"] for r in rows]
en_texts = [r["en_text"] for r in rows]
dup_ja = n - len(set(ja_texts))
dup_en = n - len(set(en_texts))
print(f"queries (rows in eval_set.tsv): {n}")
print(f"candidate pool per query:       {n}")
print(f"relevant items per query:       1 (the aligned translation; "
      f"correct_idx = query index, as in the experiment script)")
print(f"duplicate ja_text rows: {dup_ja} | duplicate en_text rows: {dup_en}")

# --- Monte Carlo ------------------------------------------------------------
rng = np.random.default_rng(SEED)
per_query_mean = np.empty(n)
all_scores = np.empty((n, N_SHUFFLES))
for qi in range(n):
    scores = np.empty(N_SHUFFLES)
    for s in range(N_SHUFFLES):
        perm = rng.permutation(n)          # random ranking of all 597 candidates
        scores[s] = ndcg_at_k(perm.tolist(), qi, k=K)
    per_query_mean[qi] = scores.mean()
    all_scores[qi] = scores

mc_mean = per_query_mean.mean()            # mean over 597 queries x 1000 shuffles
mc_se = all_scores.flatten().std(ddof=1) / math.sqrt(all_scores.size)

# --- closed-form cross-check -------------------------------------------------
analytic = sum(1.0 / math.log2(i + 1) for i in range(1, K + 1)) / n

print(f"\nMonte Carlo E[nDCG@10] over {n} queries x {N_SHUFFLES} shuffles "
      f"({n*N_SHUFFLES:,} samples): {mc_mean:.6f}  (SE {mc_se:.6f})")
print(f"Closed-form expectation (1/597)*sum_{{i=1..10}} 1/log2(i+1): {analytic:.6f}")
print(f"Sum_{{i=1..10}} 1/log2(i+1) = "
      f"{sum(1.0/math.log2(i+1) for i in range(1, K+1)):.6f}")

peak = 0.3608
print(f"\npeak layer-8 nDCG@10 = {peak}")
print(f"ratio vs MC baseline:        {peak/mc_mean:.2f}x")
print(f"ratio vs analytic baseline:  {peak/analytic:.2f}x")
print(f"rounded baseline for text:   {mc_mean:.4f}")
print(f"old (incorrect) baseline 0.1084 corresponds to 1/log2(598) = "
      f"{1.0/math.log2(598):.6f} - i.e. an un-truncated DCG position weight, "
      f"not an expected nDCG@10")
