# Table for Section 1.4 — The Gap No Previous Work Has Closed

Place after the Muller et al. (2023) paragraph, before Section 1.5.

| Study | Focus | Established | Gap relative to MINT-Pat |
|---|---|---|---|
| Pires et al. (2019); Cao et al. (2020) | Multilingual NLP (mBERT probing) | Language-agnostic representations emerge in higher layers; language identity persists at all layers | No domain adaptation; no patent-domain concepts; no causal validation |
| Bricken et al. (2023) | Mechanistic interpretability (SAEs) | SAEs decompose polysemantic activations into interpretable, approximately monosemantic features at scale | Applied only to general-domain, monolingual models; no cross-lingual analysis |
| Gururangan et al. (2020) | Domain adaptation (DAPT) | Continued pre-training on in-domain corpora consistently improves specialised downstream performance | No multilingual dimension; no interpretability analysis of what retraining changes internally |
| Zhou et al. (2021) | Patent NLP | Domain-specific Transformers outperform general baselines on patent classification/retrieval | English-only; no cross-lingual alignment; no interpretability |
| Muller et al. (2023) — closest prior work | Multilingual interpretability (SAEs on code-switching/translation pairs) | Most activation patterns remain entangled with linguistic features in general multilingual models | (1) No domain-adapted models (2) No parallel technical corpus (3) No causal interventions |

*Table X: Prior work and the gaps MINT-Pat addresses.*

**Numbering note:** inserting this as a new table in Chapter 1 makes it "Table 1," which pushes the current Table 1 (Research Objectives, Ch.2.3) to Table 2, and the current Table 2 (Literature Synthesis, Ch.4.7) to Table 3 — update the "Table of Tables" front matter accordingly.
