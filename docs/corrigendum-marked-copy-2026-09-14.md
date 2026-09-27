# Marked copy returned to IGI Global, 2026-09-14

On 2026-09-14 the IGI Global Book Proofing desk replied to the corrigendum letter of
2026-09-13 (`corrigendum-sent-2026-09-13.md`) asking for the corrections as comments inside
the digital copy rather than as prose, and supplied the published chapter PDF. The reply also
set expectations on scope:

> I must also point out that since the manuscript is already published, there are limitations
> to revisions. We can only guarantee the revisions for the digital copy of the publication if
> the revisions are accepted. If the revisions exceed our limitations, they may be forwarded to
> the editorial managers for further review in order to grant a correction erratum to your
> manuscript.

The marked copy prepared on 2026-09-14 is `Aligned-Minds-Efficient-Machines-CORRECTIONS-MARKED.pdf`:
the publisher's own PDF with 26 annotations added using PyMuPDF, each anchored on the text it
concerns. It is not committed here; the source is the publisher's copyrighted typeset file.

## What the marked copy contains

One `Text` (sticky note) annotation on printed p. 113 summarising the request and naming the
four groups, plus 25 `Highlight` annotations carrying numbered comments. Every comment gives
the published wording, the replacement wording, and the reason.

| # | Printed page | Group | Location |
|---|---|---|---|
| 1 | 113 | A, V1 mechanism | Abstract, "STAC V1 demonstrates a hybrid fine-tuning approach" |
| 2 | 118 | A | "established a complete, end-to-end differentiable pipeline" |
| 3 | 118 | A | "actively encouraged during fine-tuning through an L1 spike regularization term" |
| 4 | 118 | A | "The maturity of this version was confirmed by a comprehensive validation suite"; "was a successful" |
| 5 | 119 | D, "validated" | "a validated psychometric-to-agent mapping methodology" (carries the group's reasoning) |
| 6 | 120 | A | "a hybrid model that fine-tunes a pre-trained transformer and SNN model" |
| 7 | 123 | D | "creates a direct, empirically validated, and clinically relevant link" |
| 8 | 129 | D | "the validated SCANAQ provides a clear and actionable pathway" |
| 9 | 131 | A | "a landmark project designed to prove the fundamental viability" |
| 10 | 133 | A | "The crucial innovation … was making these parameters learnable" |
| 11 | 133 | A | "This proxy provides a useful learning signal…" (the central correction) |
| 12 | 133 | A | HEMM, "allows the model's own recent past activity to influence its current processing" |
| 13 | 134 | A | Integrated L1 Spike Regularization, audit note added after the paragraph |
| 14 | 134 | A | "While STAC V1 demonstrated feasibility, its scaling limitations…" |
| 15 | 136 | A | Deletion of the contrast with V1's L1 regularization |
| 16 | 137 | D | "captured through the validated SCANAQ assessment" |
| 17 | 143 | B, references | Tate (2024c), truncated DOI `10.5281/zenodo.140532` |
| 18 | 143 | B | Tate (2025a), wrong version and DOI shared with Tate (2025b); names all four in-text citations |
| 19 | 148 | C, Appendix B | Table 9, Section A inverted |
| 20 | 148 | C | Table 10, Section B sums two constructs |
| 21 | 149 | C | Table 11, Section C inverted |
| 22 | 149 | C | Section E rule undefined for ties (43.4% of 3,125 patterns) |
| 23 | 151 | C | Table 16, Section H empathy total |
| 24 | 151 | C | Global Notes, reverse-keyed items summed raw |
| 25 | 144 | C, Appendix A | Table 1, the eight Section A items replaced with the instrument 2.0.0 wording |

The groups are ordered in the summary note by the order I would ask the publisher to work them:
A (mechanism), B (references), C (Appendix B), D (the "validated" wording). D is marked
explicitly as the one to drop if the request as a whole exceeds what can be applied, so that the
lowest-value group cannot stall the rest.

## What reading the typeset chapter changed

The letter of 2026-09-13 quoted the submitted manuscript (Edit12, 2025-10-12). The typeset
chapter differs from it in three ways that matter.

**The Ostrau item is withdrawn.** The published reference list carries Ostrau et al. (2022) on
printed p. 142, in alphabetical position. Production added it after submission. The item was
already conditional in the sent letter; the marked copy withdraws it outright and apologises for
the confusion. This closes the one item that step 4 of the runbook left open.

**Three of the five Part I quotations do not match the printed text.** Copy-editing changed the
wording, so the letter's "Published" column is accurate to the manuscript but not to the chapter
a reader holds. The strongest of the five, the Abstract's claim that the hybrid was fine-tuned
"for sparse, event-driven learning", was removed in production: the printed Abstract reads
"STAC V1 demonstrates a hybrid fine-tuning approach", which needs only the verb changed. The
marked copy quotes the printed text throughout.

**Two locations the letter never covered.** Printed p. 118 states that energy efficiency "was
actively encouraged during fine-tuning through an L1 spike regularization term", and printed
p. 134 opens the V2 section with "While STAC V1 demonstrated feasibility". Neither appears in
the letter, because neither appears in that form in the manuscript. Both are comments 2 and 6 in
the marked copy.

## Notes on the file itself

The typeset PDF carries invisible characters inside the text runs: U+200B (zero-width space),
U+00AD (soft hyphen) and U+2011 (non-breaking hyphen). A search for a full sentence therefore
fails even when the sentence is plainly on the page. Anchors were chosen as short fragments that
avoid them, and every anchor was asserted to resolve to at least one quad before the annotation
was written. The full script and the rendered verification pages are in the session scratchpad.

Annotation author string: `Ben Kennedy (author) - correction request`. Highlight colour
(1, 0.85, 0.2) at 45% opacity, chosen to stay legible over the black body text.

## Second pass, same day

The first build of the marked copy carried 15 comments, drawn from the letter of 2026-09-13. A
systematic re-read of the whole chapter, rather than of the passages the letter already knew
about, found nine more and one error in the corrigendum itself. The file was rebuilt with 24.

**Six more V1 mechanism locations.** Printed p. 118 says the pipeline was "complete, end-to-end
differentiable" and that "the maturity of this version was confirmed by a comprehensive
validation suite of seven distinct test functions"; p. 120 says V1 "fine-tunes a pre-trained
transformer"; p. 133 says the AdEx parameters were made "learnable" and that the HEMM "allows the
model's own recent past activity to influence its current processing". None of these survive the
audit, and none were in the letter.

**Four "validated SCANAQ" sentences**, on pp. 119, 123, 129 and 137. A different class of
overstatement, recorded in `corrigendum-2026-07.md`. They go in as group D with an explicit note
that they are the first thing to drop.

**The corrigendum was itself wrong about the blast radius.** Its *What is and isn't invalidated*
section said the backbone, the projection layers, the HEMM and the head "still trained" and that
V1's loss figures were "a real measurement of that pipeline". `scripts/verify_v1_downstream.py`
shows otherwise: the backbone's gradient was exactly zero, the HEMM returned a constant, and two
unlike inputs produce bit-identical logits. As released, V1 was a constant predictor. That
section is now corrected, and the marked copy states the wider finding rather than the narrow one.

**What was checked and is fine.** Appendix B Sections D (Risk Propensity) and F (Self-Efficacy)
were read item by item against their scales and are correct as printed. Every DOI in the
reference list was checked; four that looked truncated are line-break artefacts of text
extraction and resolve correctly, leaving the two known defects. The STAC V2 material, including
the p. 136 statement that the energy figure is theoretical, needs no change.

One cosmetic defect was found and not raised: the Vaswani et al. (2017) entry on p. 143 has a
mismatched parenthesis and a garbled editor string. It is a typesetting slip with no effect on
retrieval, and adding it would dilute a request that already asks for a lot.

## Third addition, 2026-09-16

Comment 25 replaces the eight Section A items of Appendix A with the original wording of
instrument 2.0.0. On 2026-09-16 the reuse terms of all eight SCANAQ source scales were checked
against each owner's own statement (`PROVENANCE.md` in SCAN-Resources, section *Reuse terms,
verified 2026-09-16*). Six are free with citation, one (GDMS) has no author statement, and one is
not free: the BRIEF-A, published by PAR Inc., whose position is that it "will not grant permission
to include an entire test or scale in any publication". The printed Section A items track BRIEF-A
wording closely, which is why instrument 2.0.0 reworded them. Replacing the items in the digital
copy removes the exposure without a permissions request; the comment gives the reason plainly and
without the word "licensing". Sections B to H of Appendix A are unchanged. Only this 25-comment
build was sent to the publisher.

## Fourth build, 2026-09-20: 32 comments

Built after SCAN-Resources 2.1.0 was released (version DOI 10.5281/zenodo.22865240). Three changes
from the 25-comment build. Group B grows from two comments to nine: the seven defective entries
from `reference-audit-2026-09-17.md` are comments 19 to 25, each with a replacement entry whose
authors, volume, issue and pages were taken from Crossref, or a deletion with the one body
citation re-pointed (Li et al. 2021 → Bechara et al. 2005, already in the list; Ling et al. 2022
→ sentence trimmed). Comment 17 now points readers at the 2.1.0 DOI. The Appendix A comment,
now 26, carries the 2.1.0 wording and the 2.1.0 DOI. Groups C's comments renumber 26 to 32.
The Appendix B attachment was re-rendered from the 2.1.0 tree (7 pages, Tinos, no dashes, metadata
guard clean) because its Section A content pointers restate the new wording. This is the build
sent.
