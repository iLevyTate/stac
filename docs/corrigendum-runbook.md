# Corrigendum runbook — what is left, in order

Companion to [`corrigendum-combined.md`](corrigendum-combined.md) (the letter) and
[`corrigendum-2026-07.md`](corrigendum-2026-07.md) (the technical account). This file is the
operational checklist: what is done, what remains, who does it, and the exact click or command.
Updated 2026-09-09.

## Done

| Item | Evidence |
|---|---|
| STAC 4.0.0 released | https://github.com/iLevyTate/stac/releases/tag/4.0.0 (2026-09-06) |
| Zenodo 4.0.0 deposit | https://doi.org/10.5281/zenodo.22554655; concept DOI 10.5281/zenodo.14545340 resolves to it |
| DOI metadata corrected | `CITATION.cff` and README badge carry the concept DOI; 18023657 listed as the 3.0.0-beta version DOI; `.zenodo.json` declares `isVersionOf` the concept DOI |
| Two drafts merged into one letter | `docs/corrigendum-combined.md`; the two source drafts carry a superseded notice |
| Reference-list section reconciled | Tate (2025a) → 4.0.0 DOI; Tate (2025b) keeps 15867066 (confirmed to be the 2.0.0.3 record); Tate (2024c) → 14053202; Ostrau et al. (2022) added conditionally (Crossref shows it in the published list; see step 4). The 4.0.0 deposit's author of record is "Kennedy, Ben" (DataCite), not "Tate, L."; the cover note now says so and offers the editor the Kennedy, B. (2026) form. |
| Signature | Ben Kennedy, Capitol Technology University, matching the chapter byline |
| Reproduction | `python scripts/verify_v1_corrigendum.py` on `main` reproduces every figure the letter quotes (log in `corrigendum-2026-07.md`) |
| CI | green on `main` at fabab59 (run 54); the same steps re-run locally on 397c058: 63 passed, 3 skipped |
| Colab notebook | §3–§9 run on CPU 2026-09-07; two defects it surfaced are fixed (see step 5) |
| Quoted passages | all five Part I passages and the Appendix B tables match the submitted manuscript verbatim |
| Appendix B PDF | rendered from SCAN-Resources `Forms/Appendix-B-Scoring-v2.0.md` (`main`, ae7c7d0) with headless Chromium on 2026-09-09, 8 pages, metadata guard 0 hits, delivered as a file (not committed: the repository's PDFs are the published 1.0.0 record). It is not attached to the draft; attach it at send. |
| Gmail draft | "Correction request — chapter DOI 10.4018/979-8-3373-5702-7.ch005", in the gmail.com account, body identical to the letter, no placeholders left. Re-saved 2026-09-09 as plain text: the earlier save had wrapped every URL in a `google.com/url?q=` redirect, which would have gone out to the publisher. |
| Branch housekeeping | Not done: origin still carries six branches besides `main` and the current working branch. `claude/stac-doi-metadata-wbiwdd` held one stranded commit (17ab08d, the full Tate (2025a) form in the two superseded drafts), cherry-picked on 2026-09-09; `claude/stac-doi-metadata-yw5zvv` is an older superset that `main` has overtaken; `docs/readme-rewrite` is PR #25, closed unmerged on 2026-09-09; the other three are fully merged. All six can be deleted. |

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

### 3. Confirm the recipients — done 2026-09-09

The draft goes to `bookproofing@igi-global.com`, cc `cust@igi-global.com` and the volume editor
`ZianShah.Kabir@uts.edu.au`. All three come from this chapter's own production trail in the
university mailbox: the proofing desk sent the typeset-proof notice for this book on
2025-11-14, the submission system sent the chapter approval on 2025-10-24 with Dr. Kabir in
copy and `cust@` as the support address. Nothing to change.

### 4. Typeset check (you) — one item now depends on it

Crossref's record of the chapter (deposited by IGI on 2026-08-27, 40 references) lists
Ostrau et al. (2022) as reference 21, alphabetically placed, while the truncated `140532`
DOI and the duplicated `15867066` are still there. Production may have added the missing
entry. The letter and the Gmail draft were reworded on 2026-09-09 so the Ostrau item is
conditional ("if it was added in production please disregard that one item") and the
letter is correct either way. Reading the typeset chapter's reference list would let you
drop the item outright before sending.

The quotations were diffed against the submitted manuscript, not the typeset chapter. The
typeset proof is on IGI's platform at
`https://www.igi-global.com/submission/proofing/document/?did=152536` (login required;
Chrome, Firefox, or Edge). Open it at the five Part I locations and Appendix B and confirm
nothing was changed in copy-editing, or save it as PDF and send it to me to diff.

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

## Copies in circulation (tell them once the letter goes)

- James Tankard (jtankard@captechu.edu) received the full exegesis and all three chapters
  as one attachment on 2026-05-14 for a design-science dissertation built on the SCAN
  architecture. He holds the uncorrected STAC V1 account and the uncorrected Appendix B.
- The researcher outreach script on Drive (`SCAN_Researcher_Outreach_Campaign.md`,
  2026-06-07) describes STAC V1 as "a hybrid fine-tuned SNN-transformer with learnable AdEx
  neurons, surrogate gradients, and L1 spike regularization" and SCANAQ as "drawn from eight
  validated scales". If the campaign ran, those messages went out; the script needs the same
  rewording before any further use.
- No other outbound mail with the chapters attached was found in the gmail.com account.
- An unpublished draft on Drive, "SCANAQ and SCANUE: Bridging Psychometrics and AI for
  Personalized Cognitive Augmentation" (Kennedy, Mohammad, Wyandt; last edited 2025-04-19),
  cites "SCANAQ's scoring rubric (Tate et al., 2024a)" and walks a worked example through
  it. No venue, acceptance mail, or DOI for it was found; the exegesis lists three
  publications, not four. If it is ever submitted, the rubric reference and the worked
  example must move to scoring model 2.0.0 first.

## Not blocking

- [PR #5](https://github.com/iLevyTate/SCAN-Resources/pull/5) merged before the release rather
  than after, so there is no separate instrument-2.0.0 release to cut. Nothing further needed.
- The other IGI chapter, "Beyond Intelligence: The Synthetic Cognitive Augmentation Network
  Using Experts" (*Ensuring Secure and Ethical STM Research in the AI Era*), checked
  2026-09-09 from the Drive manuscript (41 pages). It has no appendices and no scoring tables;
  the SCANAQ appears only as an upcoming instrument, and STAC only as future work with no
  mechanism claim. Its four Zenodo references (Tate 2024a–d: 14052759, 14510407, 14053203,
  14545341) all resolve. No correction needed.
- The Springer chapter, "Synthetic Cognitive Augmentation Network" (Kennedy, Mohammad,
  Wyandt; *SEET 2025*, CCIS 2725, pp. 179–188, DOI 10.1007/978-3-032-08977-9_13, online
  2026-01-02), checked 2026-09-09 against the SEET submission manuscript on Drive and the
  Crossref reference deposit. STAC appears only as a future component and a pipeline diagram
  (Fig. 1), with no mechanism or result claim; there is no scoring appendix. Its three Zenodo
  references (SCANUE 14052759, SCAN-Resources 14053203, SCAN 14052885, all 1.0.0-alpha) resolve
  through DataCite to the right records. One sentence in §2.2 of the manuscript, "Although
  SCANAQ has not yet been implemented, it has been validated through current research,"
  overstates the instrument's status (see SCAN-Resources `PROVENANCE.md`); it is a soft claim
  with no dependent result, and the galley was not re-read for it. No correction letter to
  Springer is warranted. The galley proof (`643787_1_En_13_Chapter_Author.pdf`, 2025-11-11)
  sits in the university mailbox if a verbatim check is ever wanted.

## Needs its own amendment

- The PhD exegesis (June 2026) restates the STAC V1 feasibility claim in four places,
  reproduces the chapter's Appendix B in full, and carries two reference-list defects of its
  own, one of which (the SCAN 1.0.0-alpha entry pointing at the stac 2.0.0.3 DOI) does not
  occur in any chapter. The passages and the proposed wording are in
  [`exegesis-corrections.md`](exegesis-corrections.md). Route: the university's dissertation
  office. This is independent of the IGI send and does not block it.
- The ecosystem site (scanerad.com) carried a "3-4× less energy" claim, an estimated
  performance chart with no source, a "validated" label on SCANAQ, and paper cards that
  omitted the Springer chapter and pointed two cards at the same IGI chapter. Corrected on
  2026-09-09 on branch `claude/eloquent-faraday-vdkqhl` of iLevyTate/ScanEcosystem; merge to
  `main` to deploy.
