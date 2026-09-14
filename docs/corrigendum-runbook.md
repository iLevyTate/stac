# Corrigendum runbook — what is left, in order

Companion to [`corrigendum-combined.md`](corrigendum-combined.md) (the letter) and
[`corrigendum-2026-07.md`](corrigendum-2026-07.md) (the technical account). This file is the
operational checklist: what is done, what remains, who does it, and the exact click or command.
Updated 2026-09-14 (second revision, after the proofing desk's reply).

## Done

| Item | Evidence |
|---|---|
| STAC 4.0.0 released | https://github.com/iLevyTate/stac/releases/tag/4.0.0 (2026-09-06) |
| Zenodo 4.0.0 deposit | https://doi.org/10.5281/zenodo.22554655; concept DOI 10.5281/zenodo.14545340 resolves to it |
| DOI metadata corrected | `CITATION.cff` and README badge carry the concept DOI; 18023657 listed as the 3.0.0-beta version DOI; `.zenodo.json` declares `isVersionOf` the concept DOI |
| Two drafts merged into one letter | `docs/corrigendum-combined.md`; the two source drafts carry a superseded notice |
| Reference-list section reconciled | Tate (2025a) → 4.0.0 DOI; Tate (2025b) keeps 15867066 (confirmed to be the 2.0.0.3 record); Tate (2024c) → 14053202; Ostrau et al. (2022) withdrawn on 2026-09-14: the typeset chapter carries the entry on printed p. 142. The 4.0.0 deposit's author of record is "Kennedy, Ben" (DataCite), not "Tate, L."; the cover note now says so and offers the editor the Kennedy, B. (2026) form. |
| Signature | Ben Kennedy, Capitol Technology University, matching the chapter byline |
| Reproduction | `python scripts/verify_v1_corrigendum.py` on `main` reproduces every figure the letter quotes (log in `corrigendum-2026-07.md`) |
| CI | green on `main` at fabab59 (run 54); the same steps re-run locally on 397c058: 63 passed, 3 skipped |
| Colab notebook | §3–§9 run on CPU 2026-09-07; two defects it surfaced are fixed (see step 5) |
| Quoted passages | all five Part I passages and the Appendix B tables match the submitted manuscript verbatim |
| Appendix B PDF | rendered from SCAN-Resources `Forms/Appendix-B-Scoring-v2.0.md` (`main`, ae7c7d0) with headless Chromium on 2026-09-09, 8 pages, metadata guard 0 hits, delivered as a file (not committed: the repository's PDFs are the published 1.0.0 record). It is not attached to the draft; attach it at send. |
| Send copy | The Gmail draft could not be used: Gmail rewrites every URL in a stored draft into a `google.com/url?q=` redirect, and did so again after a clean re-save. The draft was retitled "DO NOT SEND FROM GMAIL" and the letter was sent instead from a plain-text copy pasted into a new message from the university account. That copy is recorded as `corrigendum-sent-2026-09-13.md`. |
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

### 4. Typeset check — done 2026-09-14

Closed by the publisher itself. The proofing desk supplied the published chapter PDF on
2026-09-14 when it asked for the corrections as in-document comments, so the typeset text was
read directly rather than inferred from Crossref. Three findings, all in
[`corrigendum-marked-copy-2026-09-14.md`](corrigendum-marked-copy-2026-09-14.md):

1. Ostrau et al. (2022) is in the published reference list on printed p. 142. Item withdrawn.
2. Three of the five Part I quotations were reworded in copy-editing. The Abstract's strongest
   claim, that the hybrid was fine-tuned "for sparse, event-driven learning", was already removed
   in production; the printed sentence needs one verb changed.
3. Two locations need correction that no manuscript-based draft covered, on printed pp. 118 and
   134.

The Appendix B corrections and the two reference-list defects were all confirmed present in the
typeset chapter, unchanged from the manuscript.

### 5. Colab — done 2026-09-07 (CPU)

Every section of `notebooks/stac_v2_colab.ipynb` was executed cell by cell on CPU against
`main`; the numbers are in the verification log in `corrigendum-2026-07.md`. The run found two
things an editor would have seen and both are fixed: §3 reported one failed test (an absolute
coherence bar that the unconverted base model also fails; now a parity test), and §7 logged a
warning that RoPE is dropped directly above a result proving it is applied. A GPU run of §8 at
the notebook's default settings is optional; the CPU probe already shows the perplexity
recovery (18,468 → 1,137 in 50 steps).

### 6. Send - done 2026-09-13

Sent from `bkennedy1@captechu.edu` to `bookproofing@igi-global.com`, copying
`ZianShah.Kabir@uts.edu.au` and `cust@igi-global.com`, subject "Correction request - chapter DOI
10.4018/979-8-3373-5702-7.ch005", with `Appendix-B-Scoring-v2.0.pdf` attached. The attachment was
re-rendered in Times New Roman (Tinos, the metric-compatible substitute available on the build
host) with all dashes removed; 7 pages, identifying-metadata guard clean. The body as sent is
`corrigendum-sent-2026-09-13.md`.

### 7. Record it - done 2026-09-14

The send is recorded in the *Submission* section of `corrigendum-2026-07.md`, in row 6 of the
pre-send table in `corrigendum-combined.md`, and in `CHANGELOG.md`.

### 8. Human reply — received 2026-09-14

The auto-reply's warning about the loosely monitored inbox did not bite. A member of the proofing
team replied the next day asking for the corrections as comments in the digital copy rather than
as prose, and attaching the published chapter PDF. Two things in that reply set expectations:

> We can only guarantee the revisions for the digital copy of the publication if the revisions are
> accepted. If the revisions exceed our limitations, they may be forwarded to the editorial
> managers for further review in order to grant a correction erratum to your manuscript.

So the print edition is out of reach, the digital corrections are not automatic, and an erratum is
the escalation path rather than the default. The escalation routes from the earlier draft of this
step are held in reserve and are not needed unless the marked copy goes unanswered:

1. Reply directly to Dr. Kabir, since a volume editor can route a post-publication correction
   internally.
2. Add a note against this chapter in IGI's proofing system:
   `https://www.igi-global.com/submission/proofing/document/?did=152536` (login required).
3. IGI Global's editorial contact form at `www.igi-global.com/contact/`, citing the chapter DOI.

### 9. Marked copy returned — done 2026-09-14 (you: send it)

`Aligned-Minds-Efficient-Machines-CORRECTIONS-MARKED.pdf` was built from the publisher's own file
and delivered as a session file. Reply to the same thread with it attached. Two points worth
making in the covering message: the Group 1 and Group 3 comments are small edits well inside the
digital-copy limits, while Group 2 replaces Appendix B wholesale and is the part most likely to
need the editorial managers, so an erratum for that group alone is an acceptable outcome. The
corrected Appendix B PDF was already attached to the 2026-09-13 message and does not need
resending unless they ask.

If the answer is that some corrections exceed the limits, the fallback is the erratum route they
named. The one thing not to accept quietly is Group 2 going uncorrected with no erratum, since a
reader following the published Appendix B assigns inverted profiles.

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
