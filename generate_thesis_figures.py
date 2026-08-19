# -*- coding: utf-8 -*-
"""generate_thesis_figures.py

Generates all supplementary figures for the MINT-Pat thesis into thesis_figures/.
Every number plotted here is taken verbatim from dataset_stats.json or from the
result tables already reported in the thesis (Tables 6.4, 6.5, 6.8, 6.9, 7.1-7.8)
and the retrained-run output (validation_results_retrained.json). No value is
estimated or invented.

Run:  python generate_thesis_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 200,
})

TECH = "#2c5f8a"   # technical concepts
LEGAL = "#a04040"  # legal-procedural concepts
GREY = "#666666"
ACCENT = "#e0a030"

LAYERS13 = ["Emb", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12"]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------
# Figure 3.1 - End-to-end research pipeline (schematic)
# ----------------------------------------------------------------------------
def fig_3_1():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f8", ec=TECH, fs=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=GREY, lw=1.2))

    # row 1
    box(0.2, 6.0, 3.2, 1.6, "JaParaPat 2016–2020\n370,835 document pairs\n(JPO × USPTO)")
    box(4.4, 6.0, 3.2, 1.6, "Preprocessing pipeline\nalign → filter → split\n(seed 42)")
    box(8.6, 6.0, 3.2, 1.6, "MINT-Pat Corpus v1.0\n1,050,514 pairs\n1,973,910 MLM lines")
    # row 2
    box(0.2, 3.4, 3.2, 1.6, "Retrained model\nperplexity 7.47 → 3.39\n(−54.6%)")
    box(4.4, 3.4, 3.2, 1.6, "Continued pre-training\nXLM-RoBERTa-base, MLM\n3 epochs, A100 (Colab)")
    box(8.6, 3.4, 3.2, 1.6, "Curated eval set\n597 verified pairs\n(12 concepts)", ec=LEGAL, fc="#f8efee")
    # row 3
    box(0.2, 0.4, 5.6, 1.8, "Six-module interpretability analysis\n(Colab GPU)\nCKA · probe · SAEs · retrieval · RAF · patching",
        fc="#f4f0e6", ec=ACCENT)
    box(6.4, 0.4, 5.4, 1.8, "Findings & guidance\nlayer-specific alignment · 97% shared features\nlayer-8 retrieval peak · practitioner guidance",
        fc="#eef6ee", ec="#3f7a3f")

    arrow(3.4, 6.8, 4.4, 6.8)          # corpus -> preprocessing
    arrow(7.6, 6.8, 8.6, 6.8)          # preprocessing -> corpus v1.0
    arrow(9.4, 6.0, 6.9, 5.0)          # corpus v1.0 -> pretraining
    arrow(10.9, 6.0, 10.9, 5.0)        # corpus v1.0 -> eval set (concept TSVs)
    arrow(4.4, 4.2, 3.4, 4.2)          # pretraining -> retrained model
    arrow(1.8, 3.4, 1.8, 2.2)          # retrained model -> experiments
    arrow(9.5, 3.4, 4.6, 2.2)          # eval set -> experiments
    arrow(5.8, 1.3, 6.4, 1.3)          # experiments -> findings
    save(fig, "fig_3_1_research_pipeline.png")


# ----------------------------------------------------------------------------
# Figure 4.1 - SAE decomposition schematic
# ----------------------------------------------------------------------------
def fig_4_1():
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 6); ax.axis("off")

    # dense polysemantic vector (left)
    ax.text(1.15, 5.6, "Dense activation (d=768)\npolysemantic neurons", ha="center", fontsize=8.5)
    vals = rng.uniform(0.2, 1.0, 24)
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((0.55, 0.7 + i * 0.19), 1.2, 0.16,
                               fc=plt.cm.viridis(v), ec="none", alpha=0.9))
    ax.add_patch(Rectangle((0.55, 0.7), 1.2, 24 * 0.19, fc="none", ec=GREY, lw=1))

    # SAE box (middle)
    ax.add_patch(FancyBboxPatch((3.2, 1.7), 2.8, 2.6, boxstyle="round,pad=0.1",
                                fc="#f4f0e6", ec=ACCENT, lw=1.4))
    ax.text(4.6, 3.85, "Sparse autoencoder", ha="center", fontsize=9.5, weight="bold")
    ax.text(4.6, 2.85, "4× expansion\n(3,072 features)\nL1 sparsity penalty\ntied decoder,\nunit-norm columns",
            ha="center", va="center", fontsize=7.8)
    ax.add_patch(FancyArrowPatch((1.9, 3.0), (3.1, 3.0), arrowstyle="-|>", mutation_scale=15, color=GREY, lw=1.4))
    ax.add_patch(FancyArrowPatch((6.1, 3.0), (7.2, 3.0), arrowstyle="-|>", mutation_scale=15, color=GREY, lw=1.4))

    # sparse feature dictionary (right)
    ax.text(8.35, 5.6, "Sparse features (~1–2% active)\nat layer 9: 97% shared EN↔JA", ha="center", fontsize=8.5)
    n = 24
    active_shared = {3, 9, 15}
    active_lang = {20}
    for i in range(n):
        y = 0.7 + i * 0.19
        if i in active_shared:
            fc = TECH
        elif i in active_lang:
            fc = LEGAL
        else:
            fc = "#e6e6e6"
        ax.add_patch(Rectangle((7.75, y), 1.2, 0.16, fc=fc, ec="none"))
    ax.add_patch(Rectangle((7.75, 0.7), 1.2, n * 0.19, fc="none", ec=GREY, lw=1))
    ax.add_patch(Rectangle((7.75, 0.05), 0.25, 0.16, fc=TECH))
    ax.text(8.08, 0.13, "shared concept feature", fontsize=7.5, va="center")
    ax.add_patch(Rectangle((7.75, -0.32), 0.25, 0.16, fc=LEGAL))
    ax.text(8.08, -0.24, "language-specific feature", fontsize=7.5, va="center")
    save(fig, "fig_4_1_sae_schematic.png")


# ----------------------------------------------------------------------------
# Figure 5.1 - Preprocessing funnel (dataset_stats.json)
# ----------------------------------------------------------------------------
def fig_5_1():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(-0.9, 6.4); ax.axis("off")

    stages = [
        ("Documents available", "370,835", 4.9),
        ("Stratified sample (4 partitions × 5 years)", "4,847", 3.7),
        ("Passed ≥50% yield gate (239 excluded)", "4,608", 2.5),
        ("Parallel pairs kept (64,748 filter rejections)", "1,050,514", 1.3),
        ("MLM lines (deduplicated, 80/10/10 split)", "1,973,910", 0.1),
    ]
    widths = [9.0, 7.4, 5.8, 4.6, 3.4]
    for (label, val, y), w in zip(stages, widths):
        x = (10 - w) / 2
        ax.add_patch(FancyBboxPatch((x, y), w, 1.0, boxstyle="round,pad=0.06",
                                    fc="#eef3f8", ec=TECH, lw=1.2))
        ax.text(5, y + 0.68, label, ha="center", va="center", fontsize=8.3)
        ax.text(5, y + 0.28, val, ha="center", va="center", fontsize=10, weight="bold")
    for y in (4.9, 3.7, 2.5, 1.3):
        ax.add_patch(FancyArrowPatch((5, y), (5, y - 0.2), arrowstyle="-|>",
                                     mutation_scale=13, color=GREY))
    ax.text(5, -0.55, "In parallel: concept-pair TSVs, 119,389 rows (train 95,511 / eval 23,878 / candidates 513)",
            ha="center", fontsize=7.8, style="italic")
    save(fig, "fig_5_1_preprocessing_funnel.png")


# ----------------------------------------------------------------------------
# Figure 6.1 - Filter rejections (filter_rejections, dataset_stats.json)
# ----------------------------------------------------------------------------
def fig_6_1():
    reasons = ["cross_section", "ja_density", "min_length_en", "min_length_ja", "figure_only_ja", "max_length"]
    counts = [55833, 3523, 3520, 1457, 399, 16]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    y = np.arange(len(reasons))[::-1]
    ax.barh(y, counts, color=TECH, alpha=0.85, log=True)
    ax.set_yticks(y); ax.set_yticklabels(reasons, fontsize=9)
    ax.set_xlabel("pairs removed (log scale)")
    for yi, c in zip(y, counts):
        ax.text(c * 1.15, yi, f"{c:,}", va="center", fontsize=8.5)
    ax.set_xlim(10, 3e5)
    save(fig, "fig_6_1_filter_rejections.png")


# ----------------------------------------------------------------------------
# Figure 6.2 - Production concept co-occurrence (concept_counts)
# ----------------------------------------------------------------------------
def fig_6_2():
    data = [("embodiment", 33630, "legal"), ("device", 21236, "tech"), ("invention", 13332, "legal"),
            ("system", 13289, "tech"), ("signal", 12503, "tech"), ("method", 9177, "tech"),
            ("claim", 8241, "legal"), ("controller", 3505, "tech"), ("semiconductor", 2807, "tech"),
            ("patent", 1586, "legal"), ("prior art", 81, "legal"), ("novelty", 2, "legal")]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(data))[::-1]
    counts = [d[1] for d in data]
    colors = [TECH if d[2] == "tech" else LEGAL for d in data]
    ax.barh(y, counts, color=colors, alpha=0.88, log=True)
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=9)
    ax.set_xlabel("co-occurring pairs in kept corpus (log scale)")
    for yi, c in zip(y, counts):
        ax.text(c * 1.18, yi, f"{c:,}", va="center", fontsize=8)
    ax.set_xlim(1, 3e5)
    ax.barh([], [], color=TECH, label="technical")
    ax.barh([], [], color=LEGAL, label="legal-procedural")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "fig_6_2_concept_counts.png")


# ----------------------------------------------------------------------------
# Figure 6.3 - Training losses + perplexity comparison (Tables 6.8, 6.9)
# ----------------------------------------------------------------------------
def fig_6_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [3, 2]})
    epochs = [1, 2, 3]
    ax1.plot(epochs, [1.377, 1.288, 1.251], "o-", color=TECH, label="train (per-token)")
    ax1.plot(epochs, [1.310, 1.244, 1.219], "s--", color=LEGAL, label="validation")
    ax1.set_xticks(epochs); ax1.set_xlabel("epoch"); ax1.set_ylabel("MLM loss")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("Continued pre-training losses", fontsize=9.5)

    models = ["general\nXLM-R-base", "MINT-Pat\nretrained"]
    ppl = [7.47, 3.39]
    bars = ax2.bar(models, ppl, color=[GREY, TECH], width=0.55, alpha=0.9)
    for b, v in zip(bars, ppl):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", fontsize=9)
    ax2.set_ylabel("patent-corpus perplexity")
    ax2.set_title("Perplexity (−54.6%)", fontsize=9.5)
    ax2.set_ylim(0, 8.6)
    fig.tight_layout()
    save(fig, "fig_6_3_training_and_perplexity.png")


# ----------------------------------------------------------------------------
# Figure 6.4 - Curated evaluation set composition (eval_set.tsv)
# ----------------------------------------------------------------------------
def fig_6_4():
    data = [("claim", 65, "legal"), ("controller", 63, "tech"), ("patent", 63, "legal"),
            ("embodiment", 62, "legal"), ("signal", 60, "tech"), ("device", 56, "tech"),
            ("semiconductor", 56, "tech"), ("system", 55, "tech"), ("invention", 52, "legal"),
            ("method", 52, "tech"), ("prior art", 12, "legal"), ("novelty", 1, "legal")]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(data))[::-1]
    counts = [d[1] for d in data]
    colors = [TECH if d[2] == "tech" else LEGAL for d in data]
    ax.barh(y, counts, color=colors, alpha=0.88)
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=9)
    ax.set_xlabel("verified pairs in evaluation set (total 597)")
    for yi, c in zip(y, counts):
        ax.text(c + 0.8, yi, str(c), va="center", fontsize=8.5)
    ax.set_xlim(0, 75)
    ax.barh([], [], color=TECH, label="technical")
    ax.barh([], [], color=LEGAL, label="legal-procedural")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "fig_6_4_eval_set_composition.png")


# ----------------------------------------------------------------------------
# Figure 7.1 - CKA layer profile (Table 7.1)
# ----------------------------------------------------------------------------
def fig_7_1():
    cka = [0.254, 0.256, 0.267, 0.282, 0.294, 0.308, 0.322, 0.333, 0.342, 0.350, 0.330, 0.323, 0.393]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(13)
    ax.axvspan(8.5, 12.5, color=ACCENT, alpha=0.10)
    ax.plot(x, cka, "o-", color=TECH, label="matched pairs (n=597)")
    ax.axhline(0.031, color=LEGAL, ls="--", lw=1.2, label="mismatched, upper-layer mean (0.031)")
    ax.annotate("peak 0.393", (12, 0.393), textcoords="offset points", xytext=(-38, 8), fontsize=8.5)
    ax.annotate("upper layers: 11.2× matched/mismatched", (10.4, 0.13), fontsize=8.2, ha="center")
    ax.set_xticks(x); ax.set_xticklabels(LAYERS13, fontsize=8)
    ax.set_ylabel("linear CKA (mean-pool)"); ax.set_ylim(0, 0.45)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "fig_7_1_cka_profile.png")


# ----------------------------------------------------------------------------
# Figure 7.2 - Language-identity probe (Table 7.2)
# ----------------------------------------------------------------------------
def fig_7_2():
    acc = [0.498, 0.998] + [1.000] * 11
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(13)
    ax.plot(x, acc, "o-", color=TECH)
    ax.axhline(0.5, color=GREY, ls=":", lw=1.2)
    ax.text(11.6, 0.53, "chance (0.5)", fontsize=8, color=GREY)
    ax.annotate("CLS at layer 0 is a constant\nembedding → chance accuracy",
                (0, 0.498), textcoords="offset points", xytext=(14, -34), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels(LAYERS13, fontsize=8)
    ax.set_ylabel("5-fold CV accuracy"); ax.set_ylim(0.4, 1.06)
    save(fig, "fig_7_2_probe_accuracy.png")


# ----------------------------------------------------------------------------
# Figure 7.3 - SAE shared % and concept-specific features (Table 7.3)
# ----------------------------------------------------------------------------
def fig_7_3():
    layers = ["L6", "L9", "L11"]
    shared = [94.0, 97.0, 88.5]
    spec = [3, 16, 5]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(3)
    ax.bar(x - 0.18, shared, width=0.36, color=TECH, alpha=0.88, label="shared features (%)")
    ax.set_ylim(0, 108); ax.set_ylabel("shared EN↔JA active features (%)", color=TECH)
    for xi, v in zip(x, shared):
        ax.text(xi - 0.18, v + 2, f"{v:.1f}%", ha="center", fontsize=8.5, color=TECH)
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines.top.set_visible(False)
    ax2.bar(x + 0.18, spec, width=0.36, color=ACCENT, alpha=0.9, label="concept-specific (count)")
    for xi, v in zip(x, spec):
        ax2.text(xi + 0.18, v + 0.4, str(v), ha="center", fontsize=8.5, color="#8a6510")
    ax2.set_ylabel("concept-specific features (>3× mean)", color="#8a6510")
    ax2.set_ylim(0, 20)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.annotate("2,936 / 3,027 active\nfeatures shared at L9", (1, 60), ha="center", fontsize=8)
    save(fig, "fig_7_3_sae_features.png")


# ----------------------------------------------------------------------------
# Figure 7.4 - Retrieval layer sweep (Table 7.4)
# ----------------------------------------------------------------------------
def fig_7_4():
    ndcg = [0.0076, 0.0174, 0.0225, 0.1044, 0.1227, 0.1341, 0.1634, 0.2710, 0.3608, 0.2069, 0.1215, 0.1782, 0.3100]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(13)
    colors = [TECH] * 13; colors[8] = ACCENT
    ax.bar(x, ndcg, color=colors, alpha=0.9)
    ax.axhline(0.0076, color=LEGAL, ls="--", lw=1.2)
    ax.text(0.1, 0.395, "Monte Carlo random baseline 0.0076 (dashed)", fontsize=7.8, color=LEGAL)
    ax.annotate("peak: layer 8 (0.3608)", (8, 0.3608), textcoords="offset points",
                xytext=(-70, 6), fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(LAYERS13, fontsize=8)
    ax.set_ylabel("nDCG@10 (EN→JA, n=597)"); ax.set_ylim(0, 0.42)
    save(fig, "fig_7_4_retrieval_sweep.png")


# ----------------------------------------------------------------------------
# Figure 7.5 - Per-concept retrieval at layer 8 (Table 7.6)
# ----------------------------------------------------------------------------
def fig_7_5():
    data = [("prior art", 0.8968, 12, "legal"), ("patent", 0.7653, 63, "legal"),
            ("invention", 0.5572, 52, "legal"), ("method", 0.5032, 52, "tech"),
            ("claim", 0.4566, 65, "legal"), ("semiconductor", 0.4501, 56, "tech"),
            ("system", 0.4145, 55, "tech"), ("embodiment", 0.4083, 62, "legal"),
            ("device", 0.3718, 56, "tech"), ("controller", 0.3653, 63, "tech"),
            ("signal", 0.3271, 60, "tech")]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(data))[::-1]
    vals = [d[1] for d in data]
    colors = [TECH if d[3] == "tech" else LEGAL for d in data]
    hatches = ["///" if d[2] < 20 else "" for d in data]
    for yi, v, c, h in zip(y, vals, colors, hatches):
        ax.barh(yi, v, color=c, alpha=0.88, hatch=h, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d[0]} (n={d[2]})" + (" ⚠" if d[2] < 20 else "") for d in data], fontsize=8.5)
    ax.axvline(0.4619, color=GREY, ls="--", lw=1.1)
    ax.text(0.467, 10.4, "mean 0.4619\n(excl. n<20)", fontsize=7.8, color=GREY)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("nDCG@10 at layer 8 (EN→JA)"); ax.set_xlim(0, 1.02)
    save(fig, "fig_7_5_per_concept_ndcg.png")


# ----------------------------------------------------------------------------
# Figure 7.6 - Cosine trajectory + RAF (Table 7.7)
# ----------------------------------------------------------------------------
def fig_7_6():
    cos = [0.1527, 0.6799, 0.9165, 0.9538, 0.9543, 0.9543, 0.9557, 0.9586, 0.9597, 0.9671, 0.9705, 0.9710, 0.9942]
    raf_layers = {3: 0.048, 6: 0.046, 9: 0.032, 11: 0.028}
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(13)
    ax.plot(x, cos, "o-", color=TECH, label="mean EN↔JA cosine (n=100)")
    ax.axvspan(-0.5, 2.5, color=ACCENT, alpha=0.10)
    ax.text(1.0, 0.30, "95% of total gain\nwithin first 2 blocks", ha="center", fontsize=8.2)
    for l, r in raf_layers.items():
        ax.annotate(f"RAF={r:.3f}", (l, cos[l]), textcoords="offset points",
                    xytext=(0, -16), fontsize=7.6, ha="center", color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(LAYERS13, fontsize=8)
    ax.set_ylabel("cosine similarity"); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, "fig_7_6_raf_trajectory.png")


# ----------------------------------------------------------------------------
# Figure 7.7 - Activation patching flip rates (Table 7.8)
# ----------------------------------------------------------------------------
def fig_7_7():
    layers = ["L3", "L6", "L9", "L11"]
    cls_flip = [0.0, 0.0, 0.0, 0.0]
    all_flip = [0.0, 0.0, 0.0, 0.26]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(4)
    ax.bar(x - 0.18, cls_flip, width=0.36, color=TECH, alpha=0.9, label="CLS-only patch")
    ax.bar(x + 0.18, all_flip, width=0.36, color=ACCENT, alpha=0.9, label="all-positions patch")
    ax.axhline(0.0, color=GREY, lw=1)
    ax.annotate("0.260 at L11\n(matched = mismatched)", (3.18, 0.26), textcoords="offset points",
                xytext=(-96, -4), fontsize=8.2)
    for xi in x:
        ax.text(xi - 0.18, 0.006, "0", ha="center", fontsize=7.5, color=GREY)
    for xi, v in zip(x[:-1], all_flip[:-1]):
        ax.text(xi + 0.18, 0.006, "0", ha="center", fontsize=7.5, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("language-ID flip rate (n=100)")
    ax.set_ylim(0, 0.32)
    ax.set_title("Unpatched baseline misclassification rate: 0.000", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "fig_7_7_patching_flips.png")


if __name__ == "__main__":
    fig_3_1(); fig_4_1(); fig_5_1(); fig_6_1(); fig_6_2(); fig_6_3(); fig_6_4()
    fig_7_1(); fig_7_2(); fig_7_3(); fig_7_4(); fig_7_5(); fig_7_6(); fig_7_7()
    print("All figures written to", OUT)
