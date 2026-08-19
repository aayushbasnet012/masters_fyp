---
name: mint-pat
description: >
  Use this skill for any task related to Aayush Basnet's MSc thesis MINT-Pat
  (Mechanistic Interpretability for Patent Transformers). Triggers include:
  working with JaParaPat corpus files (.en/.ja patent pairs), the preprocessing
  pipeline output in ~/FYP/output, training or evaluating XLM-RoBERTa-base on
  patent data, any CKA/SAE/activation-patching interpretability experiment,
  concept evaluation set curation, editing Thesis.docx, or anything referencing
  ~/FYP, dataset_stats.json, DATA_CARD.md, or the MINT-Pat thesis chapters.
  Read this file before writing or editing any code for this project — it
  encodes decisions already made and mistakes already debugged, so the agent
  does not re-derive or re-break them.
---

# MINT-Pat — Coding Agent Skill

## What This Project Is

MINT-Pat investigates whether a multilingual Transformer (XLM-RoBERTa-base)
develops shared, language-agnostic internal representations for technical and
legal concepts ("semiconductor"/"半導体", "prior art"/"先行技術") after being
continued-pretrained on parallel English-Japanese patent text — or whether
apparent cross-lingual capability is superficial statistical correlation.

Three research-question-defining outputs:
- **H1** — CKA > 0.7 for matched concept pairs in upper layers after retraining
- **H2** — SAEs find monosemantic features shared across EN/JA for the same concept
- **RQ2** — Ethical risks of deploying mechanistic interpretability for AI auditing

Every coding task in this project ultimately serves one of: (a) preparing
training data, (b) retraining the model, or (c) running an interpretability
experiment against H1/H2. If a task doesn't map to one of these, ask before
building it.

---

## Working Environment — Current Active Folder

```
~/FYP/                          ← active thesis work folder, all paths below are relative to this
├── Thesis*.docx                 ← live thesis document — multiple revision copies
│                                  routinely coexist (Thesis.docx, _revised, _v2,
│                                  etc.); DO NOT overwrite any of them without
│                                  explicit confirmation, and do not assume which
│                                  one is canonical from the filename alone — see
│                                  "Thesis Document Conventions" below before
│                                  touching any of them
│
├── output/                     ← canonical preprocessing pipeline output (flat,
│   ├── train.txt               ← MLM training lines (1,575,499 lines)
│   ├── valid.txt               ← MLM validation lines (196,571 lines)
│   ├── test.txt                ← MLM test lines (201,840 lines)
│   ├── concept_pairs_train.tsv ← concept-tagged 1to1 pairs, train split
│   ├── concept_pairs_eval.tsv  ← concept-tagged 1to1 pairs, eval split
│   ├── concept_candidates.tsv  ← auto-ranked candidates for manual 500-pair
│   │                              evaluation set curation
│   ├── dataset_stats.json      ← CANONICAL numeric source for all thesis tables
│   ├── DATA_CARD.md            ← human-readable data card (Ch.5 §5.1.8 source)
│   └── processed_files.log     ← exact list of the 4,608 documents included
│
├── mint_pat_outputs/           ← baseline interpretability experiment results
│   ├── validation_results.json ← CKA/SAE/probe/patching results on general
│   │                              xlm-roberta-base (pre-retraining baseline)
│   ├── cka_layer_profile.png
│   ├── language_probe.png
│   ├── retrieval_similarity.png
│   └── sae_loss_curve.png
│
├── -- PIPELINE SCRIPTS (copied from D:\en-ja — raw corpus NOT on this machine) --
├── analyze_japarapat.py        ← JaParaPat sample preprocessing analysis
├── level2_format_analysis.py   ← Level 2: stratified sampling & format analysis
├── level2_summary_table.py     ← Level 2: summary statistics table from results
├── level3_alignment.py         ← Level 3: alignment, quality filters, concept analysis
├── level3_extra_stats.py       ← Level 3: extra statistics from alignment output
├── mint_pat_preprocess.py      ← Main preprocessing pipeline (full corpus run)
├── mint_pat_benchmark.py       ← Alignment speed/memory benchmarking
├── mint_pat_validate.py        ← Validation suite for pipeline outputs
├── mint_pat_validate_experiments.py ← Local interpretability experiment runner
│                                  (CKA, SAE, probe, patching, retrieval).
│                                  Generates mint_pat_outputs/. Switch MODEL_PATH
│                                  from xlm-roberta-base to retrained checkpoint
│                                  to get thesis results.
│
├── -- COLAB TRAINING SCRIPTS (run on Google Colab Pro+, not locally) --
├── final_train.py              ← Full MLM training: 3 epochs, eff. batch 256,
│                                  A100 40GB, ~6–9 hrs total
└── mint_pat_smoke_test.py      ← Smoke-test: same pipeline, 50k lines, 1 epoch,
                                   T4-friendly. Run before final_train.py.
```

**This machine no longer has the raw JaParaPat corpus.** The original raw
corpus (370,835 document pairs across jp-us/jp-x-us/pct/us-jp partitions,
previously at `D:\en-ja` on a Windows/WSL machine) is NOT present here —
only the already-processed output of the pipeline run is. This means:
- Preprocessing pipeline code (discovery, alignment, filtering stages) is
  documented in this skill for reference and thesis-writing purposes, but
  **cannot be re-run from this folder** without either copying the raw
  corpus here or running on the original machine.
- Any task here works with `output/*.txt` and `output/*.tsv` as fixed
  inputs — treat them as ground truth, not as something to regenerate.
- If a task requires reprocessing or resampling the raw corpus, flag this
  explicitly rather than assuming the data is reachable.

---

## Canonical Dataset Numbers — Cite These, Verify Against `dataset_stats.json`

```
Documents sampled:        4,847   (seed=42, target_pairs=1,000,000)
Documents processed:      4,608   (passed ≥50% yield gate)
Documents skipped:        239     (low alignment yield, mostly chemistry)
Parallel pairs kept:      1,050,514
MLM lines (deduplicated): 1,973,910
Split:  train.txt=1,575,499 / valid.txt=196,571 / test.txt=201,840
Concept pairs: concept_pairs_train.tsv=95,511 / concept_pairs_eval.tsv=23,878
Concept candidates: concept_candidates.tsv=513
```

Before citing any number in the thesis or in generated code/output, **open
`output/dataset_stats.json` and confirm the value matches** — this file is
the single source of truth, the numbers above are a convenience cache and
could drift if the pipeline is ever re-run with different parameters.

---

## Resolved Issues (Historical Context — Don't Re-Debug These)

**Nested output path bug (RESOLVED at this location).** An earlier WSL run
with `--output_dir "D:\en-ja\mint_pat_output"` wrote to a doubly-nested path
(`D:\en-ja\D:\en-ja\mint_pat_output\`) due to a path resolution issue in
`resolve_user_path()`. The files now at `~/FYP/output/` are the correct,
flat, already-fixed output — confirmed by the line counts matching the
canonical numbers above. Do not apply any nested-path workaround logic here;
it is not needed in this location.

**torchaudio/transformers version conflict (RESOLVED).** Recent
`transformers` versions pull in `torchaudio` (for RNNT loss support), which
can crash with `undefined symbol: _ZNK5torch8autograd4Node4nameEv` if
torch/torchaudio versions mismatch in the venv. Fixed by pinning
`pip install "transformers==4.44.2"`. If a fresh venv is set up on this
machine, apply this pin proactively rather than waiting for the crash.

---

## File Format — JaParaPat .en/.ja Files (Reference Only — Raw Files Not on This Machine)

UTF-8, tab-separated, **2 columns, no header row**: `{sentence_id}\t{text}`

Sentence ID grammar: `{section}_{paragraph:04d}_{sentence_index}`

```
title_0000_0            title, paragraph 0, sentence 0
description_0003_7      description, paragraph 3, sentence 7
claim_0032_826,827       claim, paragraph 32, MERGED sentences 826 and 827
```

- Sections: `title`, `abstract`, `description`, `claim` (both EN/JA),
  plus `figref` (**JA-only** — figure captions like 図１, has no EN counterpart)
- `sentence_index` may be a **comma-separated list** — this is the most common
  parsing bug. Always split on `,` and expand to multiple int keys before
  using the index for anything.
- The trailing `sentence_index` (not the full ID string) is the alignment
  key — it is a document-global counter intended to be shared between the
  EN and JA file. Full-ID string matches are rare (paragraph numbering
  frequently diverges between languages even when the trailing index matches).
- EN and JA files for the same document always have **identical line counts**
  (confirmed on full corpus) — but identical line count does NOT mean good
  alignment. Use index overlap, not line count, to judge alignment quality.

This section is retained for thesis methodology writing and in case the
raw corpus is later mounted on this machine — the already-processed
`output/` files are what actual coding tasks should use day-to-day.

---

## Alignment Algorithm — `align_patent_pair()`

Reference implementation logic (in `mint_pat_preprocess.py`, now copied to
`~/FYP/mint_pat_preprocess.py`). If asked to modify or reimplement it, preserve
this exact logic:

1. Parse both files into `{id, text}` rows.
2. Expand comma-separated trailing indices into individual int keys; build
   `en_by_idx[i]` and `ja_by_idx[i]` dicts (dedupe by full ID first).
3. For each index present in both `en_by_idx` and `ja_by_idx`:
   - 1 EN + 1 JA, same section → `1to1` (the only type used for training/eval)
   - 1 EN + 1 JA, different section → `cross_section` (**discard** — known
     failure mode, e.g. EN description vs JA figref at the same index)
   - 1 EN + N JA → `one_to_many_ja` (join JA text with space)
   - N EN + 1 JA → `many_to_one_en` (join EN text with space, preserve order)
   - N EN + N JA (N>1 both sides) → `many_to_many` (**discard** — too noisy)
4. JA `figref` lines with no shared index are silently dropped (not logged
   as unmatched — they're a known JA-only artefact, not an error).
5. **Document-level yield gate**: if `len(aligned_pairs) / len(en_rows) < 0.50`,
   discard the entire document. Don't try to salvage partial yield below
   this — chemistry patents with renumbered example sections produce this
   pattern and partial salvage adds more noise than signal.

**Known unresolved limitation** (do not silently "fix" this without flagging
it to the user first): index matching at the *trailing sentence index* level
does not verify that the EN and JA *paragraph* field also matches. When
paragraph numbering diverges between languages, this produces a
superficially valid `1to1` pair that is semantically wrong. Estimated to
affect 10-20% of description-section `1to1` pairs. This is a stated
limitation in the thesis (Ch.5 §5.1.3, Deviation 2), not something to patch
retroactively without discussion — doing so would invalidate the already-
documented dataset statistics.

---

## Quality Filters — Exact Thresholds, Don't Recalibrate Casually

| Filter | Threshold | Applied to |
|---|---|---|
| Exclude alignment type | `cross_section`, `many_to_many` | All |
| Figure-only JA | regex `^(図\|Ｆｉｇ\.?\|FIG\.?)\s*[\d０-９]+`, len ≤12 | All |
| Min length EN | < 5 words | `description`, `claim` only (NOT `title`) |
| Min length JA | < 10 chars | `description`, `claim` only (NOT `title`) |
| JA CJK density | CJK chars / len(text) < 0.20 | `description`, `claim`, EXCEPT if text contains an NMR/chemistry marker (`NMR`, `1H-`, `13C-`, `δ:`, `CDCl3`, `DMSO-d`) — those are legitimate, not encoding errors |
| Max length | EN > 350 words OR JA > 1800 chars | All |
| Exact duplicate | identical EN text seen before, AND identical JA text seen before | Global, across whole corpus run |

**Explicitly NOT applied: cross-language character length ratio.**
This was tested (ratio > 5.0 threshold) and rejected — it flagged 17.4% of
otherwise-valid pairs as false positives, because Japanese frequently
expresses the same content in far fewer characters than English. If you
see this filter re-proposed, push back with this finding before implementing it.

The output in `~/FYP/output/` already reflects these filters applied —
don't re-filter `train.txt`/`valid.txt`/`test.txt` under the assumption
they're raw.

---

## Concept Vocabulary — Fixed List, Don't Add Terms Without Checking

```python
CONCEPTS = [
    ("semiconductor", ["半導体"]),
    ("prior art",     ["先行技術"]),
    ("claim",         ["請求項"]),
    ("controller",    ["制御装置", "コントローラ"]),
    ("embodiment",    ["実施形態"]),
    ("novelty",       ["新規性"]),
    ("invention",     ["発明"]),
    ("patent",        ["特許"]),
    ("signal",        ["信号"]),
    ("device",        ["装置"]),
    ("method",        ["方法"]),
    ("system",        ["システム"]),
]
```

Category split for the 500-pair manual evaluation set:
- **Technical** (200 target): semiconductor, signal, controller, device, system, method
- **Legal-procedural** (150 target): claim, prior art, novelty, embodiment, patent, invention
- **Mixed/generic** (150 target): boundary cases, general description/abstract sentences

**`prior art` and `novelty` have ZERO auto-aligned co-occurrences** in the
production run (`concept_pairs_eval.tsv`). This is a confirmed corpus
property, not a bug to fix. These two concepts need **manual supplementary
collection** — keyword grep across `output/concept_pairs_eval.tsv` plus
hand-verification, not pipeline changes. This is the next concrete task:
work from `output/concept_candidates.tsv` (513 rows) toward the 500-pair
manually verified set.

---

## Model & Training Config — Locked Decisions

```python
BASE_MODEL = "xlm-roberta-base"   # NOT mBERT — see rationale below if asked
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
NUM_EPOCHS = 3                    # 3-5 range; stop early if eval_loss plateaus
BATCH_SIZE = 16                   # per-device, A100 40GB
GRAD_ACCUM_STEPS = 16             # effective batch = 256
SEED = 42
MLM_PROBABILITY = 0.15            # standard, dynamic masking via DataCollator
FP16 = True
```

**Why XLM-RoBERTa-base over mBERT:** SentencePiece tokenizer handles Japanese
without whitespace dependency (mBERT's WordPiece systematically
under-segments Japanese compounds); broader/cleaner pretraining data
(Common Crawl vs Wikipedia-only); no NSP objective (cleaner attention
patterns for circuit analysis); better TransformerLens/SAELens support.
Don't suggest switching base models without flagging this rationale.

**Train on Colab Pro+ (A100), not local hardware** (~30-40 hrs/epoch on a
consumer GPU vs ~2-3 hrs/epoch on A100 for the full 1.97M-line dataset).
Training input is `output/train.txt` and `output/valid.txt` directly —
upload these two files to Drive, nothing else from this folder is needed
for the training step. Local hardware is for inference, activation
extraction, CKA, SAE training, and patching only, once the trained
checkpoint is downloaded back to this machine.

**Environment conflict already debugged** — see "Resolved Issues" above.
Apply the `transformers==4.44.2` pin proactively in any new venv on this
machine before running model code.

---

## Interpretability Experiment Conventions

All experiments read activations via `output_hidden_states=True` on an
`AutoModel` (not `AutoModelForMaskedLM` — use the base model for hidden
states, the MLM head is irrelevant to interpretability work). Layer indexing:
`hidden_states[0]` = embedding output, `hidden_states[1..12]` = transformer
layers 1-12. **"Layer 9" in the thesis means `hidden_states[9]`, i.e. the
output of transformer block 9** — confirm this indexing convention is
followed consistently across CKA/SAE/patching code, since off-by-one here
silently corrupts every downstream number.

**[CLS] token (position 0) is the sentence representation** used for CKA
and retrieval. Don't switch to mean-pooling without flagging it — it changes
every number already reported.

**CKA implementation**: linear CKA via double-centered Gram matrices
(Kornblith et al. 2019 formulation). Validate any reimplementation against:
CKA(X, X) ≈ 1.0, CKA(X, random_noise) ≈ 0.0.

**SAE convention**: 4× expansion factor (`n_features = 4 * d_model = 3072`
for XLM-R-base's `d_model=768`), L1 penalty ~0.001, decoder columns
re-normalized to unit norm after every optimizer step (standard SAE
training stabilization — don't skip this, the SAE collapses without it).

**Activation patching**: target layers 6, 9, 11 by default (early/mid/late
sampling across the 12-layer stack). PEE (Percentage of Effect Explained) is
the reporting metric: `PEE = (patched - corrupted) / (clean - corrupted)`.
A norm-based proxy was used for early pipeline smoke-testing and saturates
to 1.0 for full-residual-stream patches — this is a known limitation of that
proxy, not a real causal signal. **Use TransformerLens hook-based patching
with logit or rank-based measurement for any result that goes in the
thesis**; the manual-hook fallback is for pipeline smoke-testing only.

---

## Validation-Before-Spend Workflow

Before running anything on paid compute (Colab Pro+, Azure), validate the
full experiment pipeline locally against the **general `xlm-roberta-base`**
baseline first, using `output/concept_candidates.tsv` for real concept
pairs (not synthetic fallback data — synthetic repeated sentences inflate
SAE shared-feature percentages and distort probe results at small N).
Expected results on the untrained baseline (use these to judge whether a
"failure" is a real bug or just expected baseline behavior):

**Actual baseline results** are in `mint_pat_outputs/validation_results.json`
(run of `mint_pat_validate_experiments.py` against general `xlm-roberta-base`,
60 concept pairs, 12/15 checks passed). Use these as ground truth, not estimates:

| Check | Actual baseline result | Notes |
|---|---|---|
| CKA upper layers (avg) | `upper_matched=0.4443`, `upper_mismatched=0.0557` | Average across layers 9–12. Individual layers vary widely — layer 8 hit 0.8517, layer 3 hit 0.7277 on baseline already. H1 threshold (>0.7) may already be met at specific layers before retraining — frame carefully in thesis. |
| Language ID probe | 1.0 at all layers (except layer 0 = 0.5) | Saturates immediately — confirms N=60 is too small. Need ≥300 examples before trusting layer trend. |
| SAE final loss | 0.0002 | Converged correctly. |
| SAE shared-feature % | 88.33% (2385/2700 active features shared) | High on baseline — validate this is not inflated by repeated sentences in concept_candidates.tsv before citing. |
| nDCG@10 / MAP | 0.0838 / 0.0873 (n=60) | Low as expected at small N. Re-check formula before assuming model failure. |
| Patching PEE | 1.0 at layers 6, 9, 11 | Saturated — norm-proxy confirmed. Must switch to TransformerLens for thesis results. |

To validate the **smoke-test checkpoint**: change `MODEL_PATH` in
`mint_pat_validate_experiments.py` to the saved checkpoint path
(e.g. `/content/drive/MyDrive/xlm_roberta_patent_smoke/final` if running
in Colab, or local path if downloaded) and re-run. Diff the output JSON
against `mint_pat_outputs/validation_results.json` to see what changed.

A 12/15 or higher pass rate on real concept-pair data with explainable
failures (not silent crashes) means the **pipeline is validated** — switch
`MODEL_PATH` to the full retrained checkpoint and re-run unchanged.

---

## Thesis Document Conventions

**Which file is canonical — check, don't assume.** Multiple thesis file variants
routinely coexist in `~/FYP/` at once (seen historically: `Thesis.docx`,
`Thesis_revised.docx`, `Thesis_revised_v2.docx`, `Thesis_final.docx`, plus
timestamped backups). Do not hardcode any one filename as "the" canonical draft.
Before editing anything: list the `Thesis*.docx` files present, check modification
times, and confirm with the user which one is currently live — the most recently
modified one is a reasonable default guess, but treat it as a guess to confirm,
not a fact, since a stale guess here silently invalidates an entire session's work.
As of the last confirmed check (2026-07-16), the canonical file was
`Thesis_revised_v2.docx`; `Thesis.docx` was an older, superseded copy. This will
change again once a final version is produced — re-verify every session rather
than trusting this note.

Before editing the canonical file programmatically (e.g. via python-docx or a
docx-generation pipeline):
- Read the existing structure first — don't assume chapter/section numbering
  without checking the current document, since both the numbering and the
  chapter contents have been confirmed to drift from what's written below.
- Preserve the design-rationale-vs-implementation-vs-findings split described
  below — don't blend "why we designed it this way" language with "here's what
  we measured" language in the same section.
- Any numeric claim inserted into the thesis must trace back to
  `output/dataset_stats.json`, `mint_pat_outputs/validation_results*.json`, or
  another experiment's saved results file — flag if asked to insert a number
  that doesn't have a traceable source. Cross-check figure/table numbers
  particularly carefully: this thesis has confirmed instances of in-text
  chapter-based references (e.g. "Table 6.10") pointing at Word captions using
  independent sequential numbering (e.g. "Table 19") — the two schemes coexist
  and do not agree, so a reference "looking right" in one system doesn't confirm
  it's right in the other.

**Confirmed chapter map (verified against the actual document, 2026-07-16)** —
this superseded an earlier, incorrect version of this table that was off by
roughly one chapter throughout; if you find contents that don't match this
version either, re-verify against the live document rather than trusting either
table blindly:

| Chapter section | Content | Source |
|---|---|---|
| Ch.3 (§3.1–3.5) | Research methodology: named agile approach, preprocessing/retraining/interpretability **design rationale** (why, not what happened), ethical considerations | No run numbers — pure methodology |
| Ch.4 | Literature review (5 strands + theoretical frameworks + gap synthesis) | — |
| Ch.5 (§5.1–5.8) | Data preprocessing **design rationale** | No run numbers — pure methodology. Note: heading numbering here has a confirmed bug (duplicate "5.1") — don't copy its numbering pattern without checking it's been fixed |
| Ch.6 §6.1 (.1–.8) | Pipeline **implementation** + results | `output/dataset_stats.json`, `output/DATA_CARD.md` |
| Ch.6 §6.2 (.1–.6), incl. §6.2.1 Training Environment | Model retraining implementation | Training run logs, before/after perplexity — largely on Google Drive/Colab, not always locally present; flag rather than invent if unverifiable |
| Ch.6 §6.3 (.1–.7), incl. §6.3.1 Experiment runner | Interpretability tooling implementation | CKA/SAE/patching pipeline code |
| Ch.7 (§7.1–7.8) | Findings and discussion: CKA (H1), probe, SAE (H2), retrieval, RAF, causal patching, synthesis, ethical analysis (RQ2) | `mint_pat_outputs/validation_results_retrained.json` for the retrained-model numbers; `mint_pat_outputs/validation_results.json` historically held the baseline run before being overwritten by a later retrained run — check which one it currently contains before citing it, don't assume |
| Ch.8 (§8.1–8.3) | Conclusion: contributions, limitations, future work | — |
| Appendix A | Script excerpts with line counts | All pipeline scripts, e.g. `mint_pat_preprocess.py`, `level2_*.py`, `level3_*.py`, `mint_pat_validate.py`, `mint_pat_validate_experiments.py`, `final_train.py` — verify line/size counts against the actual files before citing them, confirmed drift found previously |

---

## Things Repeatedly Re-Derived — Just Check Here First

- "Should I use mBERT instead?" → No, see Model & Training Config section.
- "Should I filter by EN:JA character length ratio?" → No, explicitly rejected.
- "Why are there zero `prior art` pairs?" → Confirmed corpus property, needs manual collection, not a pipeline bug.
- "Where's the raw corpus?" → Not on this machine. Only processed `output/` is here. Flag explicitly if a task needs raw data.
- "Train.txt line count looks wrong, which output folder is correct?" → `~/FYP/output/` is the correct, already-fixed location — the nested-path bug from the original WSL run does not apply here.
- "Which alignment types should go into training data?" → Only `1to1`. `cross_section` and `many_to_many` are always discarded; `one_to_many_ja`/`many_to_one_en` are conditional, not default-included.
- "What's layer 9?" → `hidden_states[9]`, confirm indexing before trusting any number downstream.
- "What is untitled1.py?" → Legacy file — superseded by `final_train.py`. Same content, just renamed. Safe to delete.
- "Which script runs interpretability experiments?" → `mint_pat_validate_experiments.py`. Change `MODEL_PATH` at the top to point to any checkpoint. Baseline results already in `mint_pat_outputs/validation_results.json`.
- "How do I validate the smoke-test model?" → Change `MODEL_PATH` in `mint_pat_validate_experiments.py` to `xlm_roberta_patent_smoke/final` and re-run. Diff output JSON against baseline in `mint_pat_outputs/`.
- "Where are the preprocessing scripts?" → All now in `~/FYP/`: `mint_pat_preprocess.py`, `level2_format_analysis.py`, `level2_summary_table.py`, `level3_alignment.py`, `level3_extra_stats.py`, `mint_pat_benchmark.py`, `mint_pat_validate.py`. Raw corpus is still not on this machine — scripts are for reference and Appendix B.