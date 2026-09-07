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
| CI | green on `main` at fabab59 (run 54) |
| Quoted passages | all five Part I passages and the Appendix B tables match the submitted manuscript verbatim |
| Appendix B PDF | rendered from SCAN-Resources `Forms/Appendix-B-Scoring-v2.0.md`, metadata guard clean, delivered as a file (not committed: the repository's PDFs are the published 1.0.0 record) |
| Gmail draft | "Correction request — chapter DOI 10.4018/979-8-3373-5702-7.ch005", in the gmail.com account, body identical to the letter, one placeholder left for the SCAN-Resources DOI |
| Branch housekeeping | the two audit branches are already deleted on origin; only `main` remains |

## Remaining, in dependency order

### 1. Cut SCAN-Resources 2.0.0 (you; one click; irreversible because it mints a DOI)

The letter says the corrected scoring model is "archived on Zenodo as scoring model 2.0.0". Today
the SCAN-Resources concept DOI resolves to 1.1.0, which carries the scoring the letter retracts.
Nothing else in this list can finish before this.

Decide first whether [PR #5](https://github.com/iLevyTate/SCAN-Resources/pull/5) (reworded
Sections A and D, instrument 2.0.0) merges before or after the release. If before, the release
tree no longer matches the letter's "instrument version 1.0.0, unchanged" statement and the
changelog needs an entry for it. The simpler path: release 2.0.0 from `main` as it stands
(67532d5), then merge PR #5 as a later release.

Then: https://github.com/iLevyTate/SCAN-Resources/actions/workflows/release.yml → *Run workflow*
→ branch `main` → tag `2.0.0` → leave pre-release unchecked → *Run workflow*. The workflow
re-runs the scoring self-test, both dataset validators, and the PDF metadata guard before it
creates the release; all four passed locally on 2026-09-07. Zenodo picks the release up within
a few minutes.

Then say "SCAN-Resources released" and the rest of step 2 is scripted.

### 2. Fill the SCAN-Resources DOI everywhere (me, after step 1)

- Read the new version DOI from https://zenodo.org/api/records/14053202 (it redirects to the
  latest version).
- Replace `[SCAN-RESOURCES-2.0.0-DOI]` in `docs/corrigendum-combined.md` and in the Gmail draft.
- Add the version DOI to SCAN-Resources `CITATION.cff` under `identifiers:` and drop the
  "pending" comment there.
- Re-check every DOI in both trees resolves.

### 3. Confirm the recipients (you)

The draft goes to `bookproofing@igi-global.com`, cc `cust@igi-global.com` and the volume editor
`ZianShah.Kabir@uts.edu.au`. No prior mail from either IGI address exists in the mailbox. If you
have the production or proofing contact from the chapter's proof stage, use that address instead.
Failing that, Cassandra Martin (`cmartin@igi-global.com`, acquisitions, wrote on 2026-01-12) can
redirect it.

### 4. Typeset check (you, optional but recommended)

The quotations were diffed against the submitted manuscript, not the typeset chapter. If you
have the publisher's final PDF, open it at the five Part I locations and Appendix B and confirm
nothing was changed in copy-editing. Send it to me and I will diff it.

### 5. Colab (you)

Open `notebooks/stac_v2_colab.ipynb` from `main` in Google Colab and run it top to bottom. Part
IV offers it to the editor; it should have been run once from a clean session by the author.

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

- The other IGI chapter, "Beyond Intelligence: The Synthetic Cognitive Augmentation Network
  Using Experts" (*Ensuring Secure and Ethical STM Research in the AI Era*). If it reproduces
  the Appendix B scoring tables, it needs the same correction. Not checked; the draft is in
  Drive ("Chapter 7 Beyond Intelligence …pdf").
