# Corrigendum runbook — what is left, in order

Companion to [`corrigendum-combined.md`](corrigendum-combined.md) (the letter) and
[`corrigendum-2026-07.md`](corrigendum-2026-07.md) (the technical account). This file is the
operational checklist: what is done, what remains, who does it, and the exact click or command.
Updated 2026-09-07.

## Done

| Item | Evidence |
|---|---|
| STAC 4.0.0 released | https://github.com/iLevyTate/stac/releases/tag/4.0.0 (2026-09-06) |
| Zenodo 4.0.0 deposit | https://doi.org/10.5281/zenodo.22554655; concept DOI 10.5281/zenodo.14545340 resolves to it |
| DOI metadata corrected | `CITATION.cff` and README badge carry the concept DOI; 18023657 listed as the 3.0.0-beta version DOI; `.zenodo.json` declares `isVersionOf` the concept DOI |
| Two drafts merged into one letter | `docs/corrigendum-combined.md`; the two source drafts carry a superseded notice |
| Reference-list section reconciled | Tate (2025a) → 4.0.0 DOI; Tate (2025b) keeps 15867066 (confirmed to be the 2.0.0.3 record); Tate (2024c) → 14053202; Ostrau et al. (2022) added |
| Signature | Ben Kennedy, Capitol Technology University, matching the chapter byline |
| Reproduction | `python scripts/verify_v1_corrigendum.py` on `main` reproduces every figure the letter quotes (log in `corrigendum-2026-07.md`) |
| CI | green on `main` at fabab59 (run 54); the same steps re-run locally on 397c058: 63 passed, 3 skipped |
| Colab notebook | §3–§9 run on CPU 2026-09-07; two defects it surfaced are fixed (see step 5) |
| Quoted passages | all five Part I passages and the Appendix B tables match the submitted manuscript verbatim |
| Appendix B PDF | rendered from SCAN-Resources `Forms/Appendix-B-Scoring-v2.0.md`, metadata guard clean, delivered as a file (not committed: the repository's PDFs are the published 1.0.0 record) |
| Gmail draft | "Correction request — chapter DOI 10.4018/979-8-3373-5702-7.ch005", in the gmail.com account, body identical to the letter, no placeholders left |
| Branch housekeeping | the two audit branches are already deleted on origin; only `main` remains |

## Remaining, in dependency order

### 1. Cut SCAN-Resources 2.0.0 — done 2026-09-07

Released from commit 2155c39 (after PR #5 merged) as
https://github.com/iLevyTate/SCAN-Resources/releases/tag/2.0.0; Zenodo minted version DOI
10.5281/zenodo.22598618 the same day. Because PR #5 was in the tree, the deposit carries
instrument 2.0.0 (Sections A and D reworded) alongside scoring model 2.0.0, and the GitHub
release notes, extracted from the changelog's scoring-model section, omit the instrument
rework. The SCAN-Resources changelog now records what actually shipped; the release notes on
GitHub can be edited by hand to match (Releases → 2.0.0 → edit) but nothing depends on it.

### 2. Fill the SCAN-Resources DOI everywhere — done 2026-09-07

Version DOI 10.5281/zenodo.22598618 is in `docs/corrigendum-combined.md` (Part II and Part
III), the Gmail draft, SCAN-Resources `CITATION.cff` and README. The letter gained one sentence
noting the instrument 2.0.0 rework in the deposit. Every DOI in both trees resolves.

### 3. Confirm the recipients (you)

The draft goes to `bookproofing@igi-global.com`, cc `cust@igi-global.com` and the volume editor
`ZianShah.Kabir@uts.edu.au`. No prior mail from either IGI address exists in the mailbox. If you
have the production or proofing contact from the chapter's proof stage, use that address instead.
Failing that, Cassandra Martin (`cmartin@igi-global.com`, acquisitions, wrote on 2026-01-12) can
redirect it.

### 4. Typeset check (you, optional)

The quotations were diffed against the submitted manuscript, not the typeset chapter. If you
have the publisher's final PDF, open it at the five Part I locations and Appendix B and confirm
nothing was changed in copy-editing. Send it to me and I will diff it.

The one load-bearing item here is closed. The Appendix B tables had been read from interleaved
text extraction, which left the 6–12 Impulsivity row's label inferred from row order. On
2026-09-09 the manuscript's Table B3 was extracted cell by cell and the page rendered and read
as an image: row 5B is "High Motor Impulsivity, Low Non-Planning Impulsivity", 6–12,
"Acts impulsively; limited advance planning." Correction ② stands on a direct read. What
remains for the typeset PDF is house-style copyedits, which would change wording in the
"Published" quotations but not the substance of any correction.

### 5. Colab — done 2026-09-07 (CPU)

Every section of `notebooks/stac_v2_colab.ipynb` was executed cell by cell on CPU against
`main`; the numbers are in the verification log in `corrigendum-2026-07.md`. The run found two
things an editor would have seen and both are fixed: §3 reported one failed test (an absolute
coherence bar that the unconverted base model also fails; now a parity test), and §7 logged a
warning that RoPE is dropped directly above a result proving it is applied. A GPU run of §8 at
the notebook's default settings is optional; the CPU probe already shows the perplexity
recovery (18,468 → 1,137 in 50 steps).

### 6. Send (you)

Send from `bkennedy1@captechu.edu`, the address IGI has on file. The draft is in the gmail.com
account, so either forward it to yourself and send from the university account, or copy the body.
Attach `Appendix-B-Scoring-v2.0.pdf` (the rendered file from step 4 of the pre-send table; if it
needs re-rendering, print `Forms/Appendix-B-Scoring-v2.0.md` from SCAN-Resources `main` to PDF and
run the metadata guard from that repository's `validate.yml` on the result before attaching).

### 7. Record it (me)

Add the send date, recipients, and any ticket number to the *Submission* section of
`corrigendum-2026-07.md`, and a line to `CHANGELOG.md`.

## Not blocking

- [PR #5](https://github.com/iLevyTate/SCAN-Resources/pull/5) merged before the release rather
  than after, so there is no separate instrument-2.0.0 release to cut. Nothing further needed.
- The other IGI chapter, "Beyond Intelligence: The Synthetic Cognitive Augmentation Network
  Using Experts" (*Ensuring Secure and Ethical STM Research in the AI Era*). If it reproduces
  the Appendix B scoring tables, it needs the same correction. Not checked; the draft is in
  Drive ("Chapter 7 Beyond Intelligence …pdf").
