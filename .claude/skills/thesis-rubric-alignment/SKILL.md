---
name: thesis-rubric-alignment
description: Use this skill when checking whether the MSc thesis meets the university's official assessment brief and marking criteria, as opposed to general writing quality. Trigger on requests like "will this pass", "does this meet the rubric", "check against the marking criteria", "am I covering everything the brief asks for", "how much of the 40% category have I covered", "grade this against the assessment brief", or any mention of the assignment brief, marking criteria, learning outcomes, weighting, or the 40% pass threshold. This skill encodes the university's own grading document (four weighted categories, five module Learning Outcomes, pass threshold, ethics/academic-integrity/submission requirements) as the standard to audit against. It is distinct from the thesis-writing-excellence skill, which critiques prose craft and structure against an annotated distinction-grade example thesis — that skill asks "is this well written", this skill asks "does this satisfy what the university is actually going to grade." Use both together for a full pre-submission check: rubric-alignment for coverage and compliance, writing-excellence for quality of execution.
---

# Thesis Rubric Alignment

## What this is

The official assessment brief and marking scheme for this module, transcribed directly from the university's assignment document, turned into an audit checklist. Where `thesis-writing-excellence` asks whether the writing is good, this skill asks whether the thesis actually contains — in the right proportion — what the university is contractually going to award marks for. A beautifully written thesis that under-serves the 40%-weighted category still fails the assignment; this skill exists to catch that before submission, not after.

**Known gap**: the brief references a detailed "Assessment Marking Criteria" section with grade-band descriptors (e.g. what distinguishes 40-49% from 70%+ work per category) that has not been supplied yet. Everything below uses the weighted categories and their named sub-components as given — critique in qualitative bands (Missing / Weak / Adequate / Strong), not simulated percentages. If the user later provides the full grade-band table, add it as a new section here and switch to band-referenced critique.

## The official standard

**Pass mark: 40% overall, one attempt.** The work must be the student's own independent work (academic integrity requirement stated explicitly in the brief).

### Four weighted categories

**1. Research Methods and Literature Review — 30%**
- *Research Methods*: identify and describe the methods used in the research.
- *Literature Review*: review existing literature relevant to the project.
- *Discussion and Analysis*: analyze the findings from the literature and discuss their relevance to the project — not just summarize what was found, but connect it back to this project specifically.

**2. Tools, Techniques, and Solution Construction — 40% (the largest single category — weight the audit accordingly)**
- *Tools and Techniques*: detail the tools and techniques utilized within the chosen specialism.
- *Solution Construction*: explain how those tools and techniques were applied in constructing the actual solution — the "how it was built," not just "what was used."

**3. Project Management and Performance Evaluation — 20%**
- *Project Management*: discuss the management of the project, including timelines, resources, and coordination.
- *Critical Evaluation of Performance*: evaluate performance and conduct throughout the project, identifying strengths and areas for improvement — a self-critical retrospective, not a summary of what was done.

**4. Evaluation of Legal, Social, and Ethical Issues — 10%**
- *Legal Issues*: address any legal considerations related to the research and project work.
- *Social Issues*: discuss the social impact of the project.
- *Ethical Issues*: evaluate the ethical implications of the work.

These three are graded as **separate named sub-items**, even though many student ethics chapters blend them into one undifferentiated "ethics" discussion. Legal and social must each get their own identifiable treatment, not be folded silently into the ethical discussion.

### Five module Learning Outcomes (map every category back to these)

1. Present and execute **novel resolutions** to the research questions using suitable methodologies, tools, and techniques, with **critical evaluation of the problem-solving process itself** — not just the outcome.
2. **Thorough evaluation of existing literature/information resources** on the problem domain, with a **well-reasoned justification** for the proposed solution.
3. Use suitable **research strategies**, analyze findings **thoroughly**, and communicate conclusions effectively in **both written and verbal** form.
4. Use **project management techniques** appropriate to a substantial individual project.
5. Thoroughly evaluate the **professional, legal, social, and ethical** issues pertaining to the project.

Note LO5 says "professional, legal, social, and ethical" — four terms — while the weighted rubric only lists legal/social/ethical as separate scored lines. "Professional" isn't a separate weighted sub-item, but UK computing/data-science ethics chapters are conventionally expected to reference a professional code of conduct (BCS, IEEE, or ACM) as part of demonstrating this LO. Its absence won't cost marks under the stated weighting, but its presence is a cheap, expected signal — flag it as a recommended addition, not a mandatory one.

**LO-to-category mapping**, for when a critique needs to trace a specific LO to where it should be evidenced:

| LO | Primarily evidenced in |
|---|---|
| LO1 (novel resolution, critical evaluation of method) | Tools/Techniques & Solution Construction (40%), reinforced in Findings |
| LO2 (literature evaluation, justification) | Research Methods & Literature Review (30%), reinforced in Justification of the Study |
| LO3 (research strategy, analysis, communication) | Research Methods & Literature Review (30%), plus overall writing quality throughout |
| LO4 (project management) | Project Management & Performance Evaluation (20%) |
| LO5 (professional/legal/social/ethical) | Evaluation of Legal, Social and Ethical Issues (10%), reinforced wherever ethics recurs elsewhere in the thesis |

An LO with no section anywhere in the thesis that clearly evidences it is a real problem, independent of word count — flag it by name.

### Ethics policy (conditional — check applicability first)

The brief states: if the project involves **primary research with human participants** (questionnaires, interviews, surveys, user studies), the implications must be considered and relevant information communicated to every participant during data collection — i.e. informed consent is required and should be documented.

If the project is **desk-based / uses only secondary or public data** (no human participants), this requirement doesn't apply substantively — but the ethics chapter should **say so explicitly** ("this research did not involve primary data collection from human participants; all data used was drawn from publicly available sources") rather than leaving the examiner to infer it. An ethics chapter that discusses only data-privacy-of-public-datasets without ever stating whether primary human-subject research occurred reads as incomplete, even when nothing was actually required.

### Administrative / compliance checklist (not separately weighted, but non-negotiable)

- File naming: `NAME_studentID`
- Format: `.docx` or `.pdf`
- Submitted via Campus 4.0, using the link released ~2 weeks before deadline
- A statement or clear internal evidence that the work is the student's own independent work
- Language standards: "effective, accurate, and appropriate" — this is a soft LO3/communication signal, not just proofreading

## Three modes

1. **Full audit mode** — the user wants a complete pre-submission check. Go category by category (in weight order: 40% → 30% → 20% → 10%), rate each named sub-item Missing / Weak / Adequate / Strong, cite where in the thesis it is (or should be) addressed, and end with a prioritized fix list ordered by weight × current weakness — a Weak item in the 40% category outranks a Missing item in the 10% category.
2. **Section-focused mode** — the user shares one chapter/section and asks if it satisfies the brief. Identify which rubric sub-item(s) that section is meant to cover, check it against the sub-item's exact wording (not a vague sense of "ethics content"), and say plainly whether it satisfies that item or what's missing.
3. **Gap-prioritization mode** — the user already has a rough sense of what's covered and wants to know what to work on next. Answer in weight order, not writing-effort order — a quick fix to a 40%-category gap matters more than polishing an already-adequate 10%-category section.

## Common blind spots (check for these explicitly, every time)

- **Project Management as a real chapter, not a methodology aside.** Many technical theses describe *what* methodology was used (e.g. "agile sprints") without ever separately discussing actual timelines, resource allocation, or coordination as its own topic, and without a **self-critical performance evaluation** ("here's what went well, here's what I'd do differently"). This is worth 20% — a full fifth of the grade — and is one of the most commonly under-invested categories in technical MSc theses, because it's the least technically interesting part to write. Confirm both *Project Management* (timelines/resources/coordination) and *Critical Evaluation of Performance* exist as identifiable content, not just implied by the methodology chapter.
- **Legal issues treated as identical to ethics.** "Legal" should be its own identifiable thread — e.g. data licensing/terms of service of any third-party dataset or API used, intellectual property, GDPR *as a legal requirement* (not just an ethical principle), or any regulatory context relevant to the domain. If the ethics chapter never uses the word "legal" or references a specific law/regulation/license, this sub-item is likely Missing even if the ethics discussion itself is strong.
- **Discussion and Analysis vs. Literature Review treated as the same thing.** The brief lists them as two separate sub-items under the 30% category. A literature review that only summarizes sources, without a distinct pass analyzing what those findings mean *for this specific project*, is missing the second half of the category.
- **Tools/Techniques described without Solution Construction.** Listing what was used (Python, scikit-learn, etc.) satisfies the first 40%-category sub-item; explaining *how* those tools were actually assembled into the working solution is a separate, larger sub-item and is where most of the 40% weight actually lives. A tools section that reads as an inventory rather than a build narrative is under-serving this category.
- **No explicit "own independent work" signal**, or no clear statement of primary-vs-secondary research status in the ethics section.

## How to give feedback

State the category and its weight first, then the specific sub-item, then the rating, then the fix — in that order, every time, so the weight is never buried: e.g. "Tools, Techniques & Solution Construction (40%) → Solution Construction: Weak. The tools section lists scikit-learn, pandas, and Jupyter but doesn't narrate how they were assembled into the actual pipeline — add a build-narrative paragraph walking through data → features → model → evaluation as it was actually implemented." Don't soften a real gap in the 40% or 30% categories to be polite — that's exactly where marks are lost. Do acknowledge clearly-satisfied items briefly rather than padding critique for the sake of "balance."

## Relationship to `thesis-writing-excellence`

Run this skill first (or in parallel) to establish *coverage* against the brief; use `thesis-writing-excellence` to sharpen *how* the covered content is written once the rubric gaps are closed. A section can pass this skill's audit (the right content is present) while still needing work under the writing-excellence rubric (vague gap statements, no critical evaluation of cited work, etc.) — they check different things and both matter for the final grade.
