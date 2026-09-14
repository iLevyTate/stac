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

The marked copy returned the same day is `Aligned-Minds-Efficient-Machines-CORRECTIONS-MARKED.pdf`:
the publisher's own PDF with 16 annotations added using PyMuPDF, each anchored on the text it
concerns. It is not committed here; the source is the publisher's copyrighted typeset file.

## What the marked copy contains

One `Text` (sticky note) annotation on printed p. 113 summarising the request and naming the
three groups, plus 15 `Highlight` annotations carrying numbered comments. Every comment gives
the published wording, the replacement wording, and the reason.

| # | Printed page | Group | Location |
|---|---|---|---|
| 1 | 113 | V1 mechanism | Abstract, "STAC V1 demonstrates a hybrid fine-tuning approach" |
| 2 | 118 | V1 mechanism | "actively encouraged during fine-tuning through an L1 spike regularization term" |
| 3 | 131 | V1 mechanism | "a landmark project designed to prove the fundamental viability" |
| 4 | 133 | V1 mechanism | "This proxy provides a useful learning signal…" (the central correction) |
| 5 | 134 | V1 mechanism | Integrated L1 Spike Regularization, audit note added after the paragraph |
| 6 | 134 | V1 mechanism | "While STAC V1 demonstrated feasibility, its scaling limitations…" |
| 7 | 136 | V1 mechanism | Deletion of the contrast with V1's L1 regularization |
| 8 | 148 | Appendix B | Table 9, Section A inverted |
| 9 | 148 | Appendix B | Table 10, Section B sums two constructs |
| 10 | 149 | Appendix B | Table 11, Section C inverted |
| 11 | 149 | Appendix B | Section E rule undefined for ties (43.4% of 3,125 patterns) |
| 12 | 151 | Appendix B | Table 16, Section H empathy total |
| 13 | 151 | Appendix B | Global Notes, reverse-keyed items summed raw |
| 14 | 143 | References | Tate (2024c), truncated DOI `10.5281/zenodo.140532` |
| 15 | 143 | References | Tate (2025a), wrong version and DOI shared with Tate (2025b) |

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
