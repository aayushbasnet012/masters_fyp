# Change report — thesis_final_submission_v8.docx

## What this pass covered
Your AI_content.txt export flagged 607 sentence spans (avg. reported AI probability 0.89) across the thesis. I matched those spans against the document's paragraphs: 471 matched cleanly to 237 distinct prose paragraphs; the rest matched headings, table/figure captions, TOC entries, and appendix source-code listings (e.g. `import os`, `VALID_SECTIONS = {...}`) — none of which are edited by the humanize-thesis process, since rewriting a heading, caption, or code line would either be meaningless or would break the document. Detectors flagging code and formulaic captions as "AI-like" is a known false-positive pattern, not something prose editing can fix.

I read all 237 flagged prose paragraphs against the AI-tell checklist (filler connectives, hedge-stacking, templated triads, inflated verbs, empty summarizing sentences, uniform rhythm).

## What I found
This document has already been through a humanization pass (HUMANIZED.docx). The flagged prose is dense, citation- and statistic-heavy academic and technical writing — nearly every sentence carries a number, a citation, or a specific technical claim. It does not contain the classic AI markers the skill targets (no "moreover/furthermore" openers, no "delve/leverage/underscore," no empty summarizing sentences, no hedge-stacking, no templated triads). A keyword sweep for the full tell list turned up only technically legitimate uses (e.g. "causally robust," "feature landscape," "not only X but also Y" with a genuine contrast) — nothing to cut.

## What was changed
4 edits applied, all zero-risk to meaning:
- Removed one redundant "together" (p.87).
- Split 3 long run-on sentences into two sentences each, breaking up uniform rhythm, without changing a single word of substance (p.24 abstract, p.238, p.322).

## Flagged — left untouched
The remaining 233 flagged paragraphs were left as-is. Nearly all of them are results/methodology prose where every clause is load-bearing (a statistic, a citation, a hyperparameter, a named method) — exactly the case the skill's tie-break rule says to leave alone rather than risk altering a finding for a cosmetic rewrite.

## Integrity statement
No numbers, statistics, units, dates, citations, quotations, technical terms, cross-references, or the direction/strength of any claim were altered. Mechanical verification (`verify.json`): 0 violations, 0 warnings across all 876 paragraphs.

## Honest note on detector scores
AI detectors are unreliable and change constantly, and this particular export shows very high confidence (avg. 0.89) on dense, well-cited technical writing — a pattern where such detectors are known to misfire, since domain-expert academic prose is inherently low-perplexity regardless of who wrote it. This pass improves rhythm and trims one redundancy; it does not guarantee any change in detector score, and I did not use gimmicks (invisible characters, homoglyphs, synonym-spam) to chase one. If you want, I can look at whether the flagged spans cluster in a way that suggests the detector is keying on formatting (e.g. dense parenthetical citations) rather than wording — that's a different problem than word choice.
