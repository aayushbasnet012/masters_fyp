---
name: humanize-thesis
description: >-
  Lightly polish an academic thesis, dissertation, or report (.docx) so it reads
  as natural human academic writing — while preserving meaning, numbers,
  citations, and claims exactly. Use this skill whenever the user wants academic
  writing humanized, de-AI'd, naturalized, or polished: "humanize my thesis",
  "make it less AI / not sound like AI", "de-AI my report", "make my chapter
  sound human", "it sounds robotic / like ChatGPT wrote it", "academic docx
  polish". Also use it when the user points at a thesis, FYP report,
  dissertation, or journal manuscript .docx and asks for it to read more
  naturally or sound like they wrote it. Do NOT use for marketing copy, casual
  text, emails, social media posts, or code.
---

# humanize-thesis

Light editorial polish for academic .docx documents: natural rhythm, plain direct
verbs, varied sentence shape — with meaning drift as close to zero as possible.
Removing AI "tells" is a byproduct of good editing, never a goal pursued through
gimmicks. Act like a careful human copy-editor with a strict brief: improve how
sentences read; change nothing about what they say.

## Contract

- **Input**: path to a `.docx` — a full thesis or a single chapter.
- **Output**: a new copy named `<original>_humanized.docx` (never overwrite the
  source) plus `<original>_change-report.md`, saved next to the original unless
  the user says otherwise.
- Long documents are processed section-by-section (per heading), keeping the
  neighbouring paragraphs in view so tone stays consistent across the document.

## Hard constraints — never alter meaning (highest priority)

Never change, reorder, or remove:

- numbers, statistics, units, dates
- in-text citations and their exact positions; reference-list entries
- direct quotations and quoted material
- technical terms, variable names, equations, symbols
- figure/table/section cross-references ("see Section 2.3", "as shown in Figure 4.1")
- the direction or strength of any claim — no reversing, no softening, no
  sharpening of findings

Skip entirely (do not edit): reference/bibliography lists, equations, code
blocks, figure/table captions, direct quotes, author names, headings, tables,
and table-of-contents entries.

Tie-break rule: if a passage cannot be improved without risking meaning, leave
it untouched and flag it in the change report. An awkward sentence left alone is
fine; a subtly altered finding is a serious failure.

## Edit depth: light polish only

Preserve wording closely. Do not restructure arguments, merge or split sections,
or reorder content. Within those limits:

- **Vary rhythm.** Break uniform sentence lengths; let an occasional sentence be
  short. Break repeated paragraph templates (claim → three examples →
  mini-summary, over and over).
- **Cut AI connective tissue and filler**: *moreover, furthermore, additionally*
  as reflexive paragraph openers; *in conclusion, it is important to note, it is
  worth noting, delve, leverage, underscore, tapestry, navigate the landscape,
  plays a pivotal/crucial role, serves as a testament, in the realm of,
  multifaceted, myriad, seamlessly, holistic,* and *robust, foster, facilitate,
  endeavors* when used as inflation rather than meaning.
- **Collapse hedge-stacking**: "it could be argued that it may potentially" →
  one deliberate hedge ("this may reflect …"). Keep exactly the hedging strength
  the claim already has — remove redundancy, not caution.
- **Delete empty summarizing sentences** ("Overall, these benefits demonstrate
  the value of the system.") — but only when the sentence carries no number,
  citation, or technical term; otherwise condense around those elements.
- **Prefer plain, direct verbs and concrete phrasing**: "conducted an analysis
  of" → "analysed"; "is able to provide" → "provides"; "in order to" → "to".
- Never touch *significant/significance* anywhere near statistics — there it is
  a technical term, not diction.
- Follow the document's existing spelling convention (BrE/AmE), notation, and
  formality. No contractions in formal academic prose.

Read `references/ai-tells.md` before the first rewriting pass — it has the
extended tell list and before/after examples.

## Voice

Generic, discipline-neutral academic register: formal but not stiff, measured
hedging, consistent tense (typically past for methods and results, present for
established knowledge and for discussing the document itself). Use passive voice
or first-person-plural the way the document already does — follow its dominant
convention rather than imposing one.

**Optional calibration**: if the user offers samples of their own writing, read
`references/style-calibration.md` and apply it. Never require samples.

## Workflow

The bundled script `scripts/humanize_docx.py` (requires `python-docx`:
`pip install python-docx --break-system-packages`) does the mechanical work. It
edits only text nodes in a copy of the file, so styles, numbering, images,
fields, footnotes, headers/footers, and tables survive untouched. For unusual
inputs (tracked changes, .dotx templates, damaged files), consult the `docx`
skill first.

1. **Extract** — `python scripts/humanize_docx.py extract thesis.docx -o segments.json`.
   Every paragraph gets a category (prose / heading / caption / quote /
   reference / equation / code / toc / empty) and, for prose, a list of
   segments. Only segments marked `"editable": true` (plain body text) may be
   rewritten. Locked segments — citation fields, hyperlinks, footnote marks,
   formatted spans, math — are fixed anchors, which is what physically prevents
   citations and formatted terms from moving or changing.
2. **Review the classification.** Scan the JSON for misfiled paragraphs (an
   unstyled caption, a quote without quote style, the start of an appendix) and
   leave anything doubtful out of your edits.
3. **Rewrite, section by section.** Work through one heading's paragraphs at a
   time with neighbouring text in view. For each editable segment, either leave
   it or write a light-polish replacement. Rules: keep every number, citation,
   quotation, cross-reference, and technical term verbatim and in the same
   order; never move content across a locked anchor; treat each segment as
   self-contained; keep each claim's direction and strength. Record edits as
   `[{"p": …, "s": …, "old": "<exact current text>", "new": "…"}]` in `edits.json`.
4. **Apply** — `python scripts/humanize_docx.py apply thesis.docx edits.json -o thesis_humanized.docx`.
   Always applies to a fresh copy of the original, so it is safe to fix rejected
   edits and rerun.
5. **Verify** (separate pass — non-negotiable) —
   `python scripts/humanize_docx.py verify thesis.docx thesis_humanized.docx -o verify.json`,
   then a semantic re-read. The script diffs every paragraph and hard-fails on
   changed numbers, changed/moved/removed citations, altered quoted material,
   edits to non-prose paragraphs, or structural drift; it warns on changed
   technical tokens and on shifts in negation/direction words. **Treat every
   violation as a bug**: remove or fix the offending edit, re-apply, re-verify.
   Then re-read each edited paragraph against the original and confirm no claim
   changed direction or strength and no technical term was dropped — the script
   cannot judge meaning; you must.
6. **Change report** — write `<original>_change-report.md` using the template
   below.

## Change report template

```markdown
# Change report — <filename>
Edited N of M prose paragraphs. Skipped: headings, captions, quotes,
references, equations, tables.

## What was changed (brief)
One short paragraph: the kinds of edits made (filler removed, rhythm varied,
hedges collapsed, empty summaries deleted).

## Flagged — left untouched
- p.N "<first few words…>" — why it was left alone.

## Integrity statement
No numbers, statistics, units, dates, citations, quotations, technical terms,
cross-references, or the direction/strength of any claim were altered.
Mechanical verification: <pass/fail summary from verify.json>.

Note: AI detectors are unreliable and constantly changing. This pass improves
writing quality and reduces obvious AI patterns, but no detector score is
guaranteed, and it does not misrepresent who wrote the work.
```

## Detection reality

AI detectors are unreliable and constantly changing. This skill improves writing
quality and reduces obvious AI patterns, but it guarantees no detector score and
never fabricates authorship or misrepresents who wrote the work. Do not use
invisible unicode, homoglyphs, deliberate typos, or synonym-spam — those are
gimmicks that damage the document and its author.

## Built-in test

`test/sample-paragraphs.md` holds five AI-flavored thesis paragraphs with
expected behaviour; `python test/make_sample_docx.py sample_thesis.docx` builds
them into a real document. Run the full workflow on it: the result should read
more naturally, `verify` must pass with zero violations, and formatting (the
heading, caption, italic term, block quote, and references) must be untouched.
Also confirm the negative case: an edit that drops "(Chen et al., 2021)" must
make `verify` exit non-zero.

## Files

- `scripts/humanize_docx.py` — extract / apply / verify pipeline
- `references/ai-tells.md` — extended tell list and before/after examples
- `references/style-calibration.md` — optional user-style calibration hook
- `test/sample-paragraphs.md`, `test/make_sample_docx.py` — built-in test
