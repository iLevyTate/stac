# Corrigendum as sent, 2026-09-13

The record of what actually went to the publisher. The working letter is
[`corrigendum-combined.md`](corrigendum-combined.md); this file is that letter as it was
formatted for a plain-text email and sent.

| | |
|---|---|
| **Sent** | 2026-09-13, from `bkennedy1@captechu.edu` |
| **To** | `bookproofing@igi-global.com` |
| **Cc** | `ZianShah.Kabir@uts.edu.au`, `cust@igi-global.com` |
| **Subject** | Correction request - chapter DOI 10.4018/979-8-3373-5702-7.ch005 |
| **Attachment** | `Appendix-B-Scoring-v2.0.pdf`, the corrected appendix rendered in Times New Roman, 7 pages, identifying-metadata guard clean |
| **Response** | Automatic reply from the proofing desk at 23:08 UTC the same day; no human reply and no ticket number as of 2026-09-14 |

Differences from `corrigendum-combined.md`, all presentational:

- Every en dash and em dash removed, along with the Unicode minus sign and the ellipsis, so the
  body survives any mail client. Numeric ranges are hyphenated, clause dashes became colons,
  commas, or sentence breaks, and three sentences were restructured where a straight swap would
  have produced a comma splice.
- The loss function in Part I item 4 is written as plain text (`L = L_CE + lambda * ||S||_1`)
  rather than with norm bars and a subscript.
- US spelling of "behavior" throughout, and a closing signature block.

One open question for this record: whether the Ostrau et al. (2022) block in Part III was deleted
before sending. It was conditional on the typeset chapter's reference list, which Crossref suggests
already carries the entry. If it was deleted, this file overstates by one item.

---

Dear Dr. Kabir and the IGI Global editorial team,

I am writing to request a correction to my chapter, "Aligned Minds, Efficient Machines: Integrating Neuromorphic Computing for Personalized AI," in Innovative Decision-Making in Engineering: The Role of Cognition, Heuristics, and Human Factors (IGI Global, 2026, pp. 113-152; DOI 10.4018/979-8-3373-5702-7.ch005).

Two independent issues came to light after publication, and I would like to correct both in a single notice so that the chapter's record is updated once.

1. STAC V1 spiking pathway. A code audit of the software repository cited in the chapter established that the STAC V1 spiking pathway was inactive in the released implementation. Two coupled defects were responsible: the AdEx neurons received input current roughly two orders of magnitude below their firing threshold, and the Gaussian surrogate gradient underflowed float32 to exactly zero at the resulting operating point. Each independently prevented spiking; together they were mutually masking, because the sparsity metric that would have revealed the problem reads zero both for a maximally efficient network and for a completely silent one. No published quantitative result is affected: the chapter reports no STAC V1 perplexity, accuracy, or spike-rate figures, and the STAC V2 material and hardware-validation caveats are unchanged. The corrections are confined to statements of mechanism in the abstract and the STAC V1 methodology section (Part I below). Both defects are now fixed in the repository, the corrected behavior is covered by a regression baseline, and the original defect is reproducible from a script included there.

2. Appendix B (SCANAQ scoring breakdown). The scoring specification in Appendix B is incorrect in four of its eight sections, such that a reader following it would assign profiles that are inverted or undefined, and two further sections sum reverse-keyed items without transforming them. These errors affect how questionnaire responses are turned into profiles. They do not affect the body of the chapter, its claims, its figures, or Appendix A, the questionnaire instrument itself, whose item text, item counts, and response scales are unchanged and correct. A corrected Appendix B is attached, and I would be grateful if it could be substituted for the original in the online version where the format allows (Part II below).

3. Reference list. While preparing this request I found four reference-list issues in the submitted manuscript: a truncated DOI, one DOI shared by two entries, a version that does not contain the cited artefact, and a work cited six times in the text with no entry in the list. Crossref's record of the published chapter does list that last work, so if it was added in production please disregard that one item; the other three stand. The corrected entries are in Part III. The software and data cited in the chapter were deposited on Zenodo under "Tate, L.", an alias I published repository releases under. The STAC 4.0.0 deposit cited in the corrected entry carries my own name, with the alias recorded in its citation metadata. The corrected entries keep the "Tate, L." form so the in-text citations continue to resolve; if house style requires the deposit's author of record, that one entry should read Kennedy, B. (2026) and its in-text citations change to match.

Every claim in Part I is reproducible from the cited repository, and Part IV describes how. Whether this is best handled as proof corrections, an erratum linked to the chapter's online record, or a note in a future printing, I will follow your process. The chapter commits to publishing failure reports as a governance practice; these corrections are offered in that spirit.

With thanks,
Ben Kennedy
Capitol Technology University

Attachment: Appendix-B-Scoring-v2.0.pdf, the corrected Appendix B (scoring model 2.0.0).


PART I: STAC V1 SPIKING PATHWAY

(1) Abstract

Published:
STAC V1 demonstrated feasibility via a hybrid fine-tuning methodology that combines a pretrained transformer with spiking elements for sparse, event-driven learning (Tate, 2025a).

Corrected:
STAC V1 demonstrated feasibility of a hybrid fine-tuning methodology combining a pretrained transformer with spiking elements. A subsequent code audit established that the spiking pathway was inactive in the released implementation (see Corrigendum); the sparse, event-driven learning it was designed to provide was not achieved.

(2) STAC V1 methodology: surrogate gradients

Published:
This provides a valuable learning signal, allowing gradients to flow back through the spiking neurons and enabling the backpropagation algorithm to fine-tune both the synaptic weights and the neurons' intrinsic firing properties.

Corrected:
This was intended to provide a learning signal allowing gradients to flow back through the spiking neurons. A July 2026 audit of the released implementation established that this did not occur: the Gaussian surrogate was parameterized with unit width while its argument (V - V_th) is expressed in millivolts and remained 10 to 20 mV from threshold, so the kernel evaluated below the float32 subnormal limit and truncated to exactly zero. No gradient reached the spiking layer.

(3) STAC V1 methodology: feasibility claim

Published:
STAC V1, as illustrated in the diagram in Figure 3, demonstrated the feasibility of creating a high-performance hybrid SNN transformer.

Corrected:
STAC V1, as illustrated in the diagram in Figure 3, established the architecture and training pipeline for a hybrid SNN transformer. As released, its spiking pathway did not activate, so the demonstration covers the pipeline's construction rather than the contribution of its spiking components.

(4) STAC V1 methodology: spike regularization

Add after the sentence "The total loss is defined as L = L_CE + lambda * ||S||_1.":
Audit note. Because the AdEx neurons emitted no spikes in the released implementation, the L1 term evaluated to exactly zero throughout training and exerted no regularization pressure. The membrane potential settled at V_rest + I/(1+a), approximately 14.7 mV below threshold, against a required drive of (1+a)(V_th - V_rest) = 75. Reported training behavior therefore reflects the cross-entropy objective alone.

(5) Initial Results and Future Standards

Strike the clause "rather than the integrated L1 spike regularization used during STAC V1 fine-tuning". The comparison has no referent, since that term was identically zero.


PART II: APPENDIX B, SCANAQ SCORING MODEL

Appendix B specifies how each SCANAQ section is scored and labelled. In four of the eight sections the specification is incorrect, such that a reader following it would assign profiles that are backwards or undefined. Two further sections produce incorrect totals. The attached corrected Appendix B replaces the appendix in full; the changes are:

1. Section A (Executive Functioning): inverted interpretation. All eight items are worded as difficulties on a 1 = Never / 3 = Often scale, so the lowest total (8) reflects the fewest difficulties. The published table labels the low range (8-13) as "Difficulty initiating and planning tasks" and the high range (14-24) as "Strong working memory", the reverse of the items' direction.

2. Section C (Impulsivity): inverted interpretation. The same error: the minimum score range (6-12) carries the profile "High Motor Impulsivity, Low Non-Planning Impulsivity" and the description "Acts impulsively; limited advance planning," although 6 is the least impulsive score attainable.

3. Section B (Emotion Regulation): two constructs summed into one. Cognitive reappraisal (Q9-Q10) and expressive suppression (Q11-Q12) are distinct strategies, but the published table sums them into a single total and reads a two-dimensional label from it. As a result two of the four profiles are unreachable, and opposite response patterns receive the same label. For example, maximum reappraisal with minimum suppression (14 + 2 = 16) and its exact opposite (2 + 14 = 16) both total 16 and both map to "Low Cognitive Reappraisal, High Expressive Suppression." The corrected version scores the two strategies separately.

4. Section E (Decision-Making Style): undefined for tied responses. The rule selects "the single highest-scoring item" among Q22-Q26 but specifies nothing for ties. Of the 3,125 possible response patterns, 43.4% contain a tie for the maximum and are therefore undefined. The corrected version adds a co-dominant-style rule and an all-equal case.

5. Sections G (Perceived Stress) and H (Empathy): reverse-keyed items summed raw. Q32 (a positively worded item in the stress section) and Q35 (a reverse-worded item in the empathy section) run opposite to their sections' direction and must be transformed (6 - raw) before summing. The published Global Note ("Scores are summed within each section...") shows they were summed raw, shifting those section totals by up to 4 points. The band labels were unaffected.

The corrected Appendix B also separates the Fantasy item (Q36) from the empathy total, since it measures imaginative transportation into fiction rather than empathy, and adds an explicit missing-data rule. A complete list is in the Summary of corrections at the end of the attached file.

Effect on results already produced. Any profile generated with the published Appendix B should be recomputed from the raw item responses using the corrected scoring. The old profile codes cannot be mechanically converted, because in the affected sections the underlying subscale values were never computed. Where only a profile code was retained, without the raw responses, it cannot be migrated.

Authoritative version. The corrected scoring model is maintained openly in the SCAN-Resources repository and archived on Zenodo as scoring model 2.0.0 (version DOI 10.5281/zenodo.22598618; concept DOI 10.5281/zenodo.14053202), with a full changelog of the corrections and a script that checks the model against a worked example and against regressions for each of the errors above. The attached appendix is a publication-ready extract of it. That release also carries a reworded edition of the questionnaire (instrument 2.0.0, in which the item text of Sections A and D was rewritten; item count, response scales, and scoring are unchanged). The chapter's Appendix A remains instrument 1.0.0, and the attached appendix scores it as published.


PART III: REFERENCE LIST

Tate (2025a): STAC V1

Published:
Tate, L. (2025a). STAC V1 implementation in iLevyTate/stac (Version 2.0.0.3) [Computer software, Jupyter Notebook]. Zenodo. https://doi.org/10.5281/zenodo.15867066

Issue: Tag 2.0.0.3 (commit 26e213d, 2025-07-11) contains no V1 implementation, only stac-v1/README.md. The notebook stac-v1/stacv1.ipynb was committed 2025-07-13 (7b09d54), two days after the tag. The entry also shares its DOI, 10.5281/zenodo.15867066, with Tate (2025b), so the two references were indistinguishable.

Corrected: Cite the release that contains the V1 implementation (the stac_v1/ package) and the corrected behavior described in Part I. Its year changes, so in-text citations of Tate (2025a) become Tate (2026):

Tate, L. (2026). STAC: Spiking Transformer Augmenting Cognition (Version 4.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22554655

Optionally add a retention note: "Earlier releases up to 3.0.0-beta contained the inactive V1 spiking pathway described above."

Tate (2025b): STAC V2

Unchanged. The entry keeps 10.5281/zenodo.15867066, the version DOI of release 2.0.0.3 (2025-07-12), which is the V2 conversion framework as cited. Once Tate (2025a) moves to the 4.0.0 DOI above, the two entries no longer collide.

Tate (2024c): SCAN-Resources

Issue: The entry gives a truncated DOI, 10.5281/zenodo.140532, which does not resolve.

Corrected: The DOI is 10.5281/zenodo.14053202 (the SCAN-Resources concept DOI, which resolves to the latest version). Readers looking for the corrected scoring model should use the 2.0.0 version DOI given in Part II, 10.5281/zenodo.22598618.

Ostrau et al. (2022): missing entry

Issue: Cited in the text six times, including in the Abstract, but absent from the reference list of the submitted manuscript. Crossref's record of the published chapter (deposited 27 August 2026) does include this entry, so it may have been added in production; if the typeset chapter already carries it, this item needs no action.

Add, if absent:
Ostrau, C., Klarhorst, C., Thies, M., & Rückert, U. (2022). Benchmarking neuromorphic hardware and its energy expenditure. Frontiers in Neuroscience, 16, 873935. https://doi.org/10.3389/fnins.2022.873935


PART IV: HOW TO VERIFY

Every claim in Part I is reproducible from the cited repository (https://github.com/iLevyTate/stac), two ways:

1. The V1 defect. Run:
    python scripts/verify_v1_corrigendum.py
It runs the original notebook's neuron and layer code verbatim at its shipped parameters and reports 0 spikes of 262,144 neuron-timesteps, the membrane 14.70 mV below threshold, the L1 penalty at exactly 0.0, and the surrogate gradient at exactly 0.0. The post-fix behavior (spike rate 0.147, sparsity 86.8%, L1 term 1.47e-06) is pinned by the regression baseline in docs/baselines/stac_v1_smoke.json, which tests/test_v1_baseline.py asserts against on every commit.

2. The V2 findings, end to end, in the browser. Open notebooks/stac_v2_colab.ipynb from the repository in Google Colab and run it top to bottom. It converts a model (faithful with spiking off), shows the frozen-model collapse and the energy projection with spiking on, and runs the spike-aware retraining that recovers quality, the evidence behind the V2 statements that the chapter's scope note leaves unchanged.

For Part II, scripts/score_scanaq.py in the SCAN-Resources repository (https://github.com/iLevyTate/SCAN-Resources) checks scoring model 2.0.0 against the worked example in Forms/SCANAQ Numerical Scoring Breakdown.md and carries a regression for each of the five errors listed above; it runs in that repository's CI.


SCOPE STATEMENT

The following are unaffected and require no correction:

- All STAC V2 material: the conversion pipeline, the disabled-spiking disclosure, the T=1 disclosure, and the 3-4x software-latency figure, which was explicitly scoped to PyTorch-profiler execution time rather than joules.
- Every hardware-validation caveat, which the chapter already states.
- STAC V1's architecture, training pipeline, and HEMM design, which are as described. The model did train; the non-spiking components (GPT-2 backbone, projections, HEMM, lm_head) are differentiable and untouched by the defect. Only the spiking mechanism's contribution is withdrawn.
- All SCANUE material.
- Appendix A (the SCANAQ instrument: item text, item counts, response scales) and the body text describing it. Only Appendix B, the scoring breakdown, is corrected, per Part II.

Sincerely,

Ben Kennedy.
