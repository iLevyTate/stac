# Exegesis — passages that repeat the corrected claims

The PhD exegesis, *The Synthetic Cognitive Augmentation Network (SCAN) Ecosystem: A
Prefrontal-Cortex-Inspired, Psychometrically Aligned, Neuromorphic Framework for Cognitive
Augmentation* (Capitol Technology University, June 2026), synthesises the three published
chapters and restates the STAC V1 and Appendix B material that the combined corrigendum
corrects. It is a fourth record of the same claims and needs its own amendment.

**Source read:** the Google Doc "The Synthetic Cognitive Augmentation Network (SCAN)
Ecosystem:" in the author's Drive, last modified 2025-12-11 (the pre-defence draft). The
deposited June 2026 text was not available to this check, so every quotation below must be
re-diffed against the version of record before anything is sent. Whether the exegesis has been
deposited with ProQuest is unconfirmed: a reader wrote on 2026-05-14 that they could not find
it there.

**Route:** the university's dissertation office (Director of Dissertations, who chaired the
work), asking how an errata sheet or addendum is attached to the deposited copy; ProQuest
accepts author-submitted corrections through the institution if the record exists there.

## Passages

Section names are the exegesis's own headings; the wording is from the 2025-12-11 draft.

### 1. Methodological Contributions

**Draft text**

> Additionally, the Spiking Transformer Augmenting Cognition (STAC) pipeline contributes a
> neuromorphic conversion-and-deployment methodology: STAC V1 demonstrates a hybrid
> SNN–transformer training regime, and STAC V2 offers a conversion […]

**Amend to**

> STAC V1 established the architecture and training pipeline for a hybrid SNN–transformer;
> as released, its spiking pathway did not activate (see the corrigendum to Kennedy, 2026),
> and STAC V2 offers a conversion […]

### 2. Practical and Technical Contributions

**Draft text**

> In practice, STAC V1 validates a hybrid training path for spiking–transformer systems,
> whereas STAC V2 operationalizes an end-to-end conversion toolchain […]

**Amend to**

> In practice, STAC V1 built the hybrid training path for spiking–transformer systems (its
> spiking components were later found inactive in the released code), whereas STAC V2
> operationalizes an end-to-end conversion toolchain […]

### 3. Publication 3: Aligned Minds, Efficient Machines

**Draft text**

> The initial version, STAC V1 (Tate, 2025a), was a landmark research project designed to
> prove the fundamental viability of creating a high-performance hybrid SNN-transformer. Its
> architecture, illustrated in Figure 5, was a meticulous exercise in neuro-inspired
> engineering, defined by three key technical innovations that are directly observable in the
> project's implementation artifacts.

**Amend to**

> The initial version, STAC V1 (Tate, 2026), was designed to prove the viability of a hybrid
> SNN-transformer. Its architecture, illustrated in Figure 5, defines three technical
> components that are present in the implementation artifacts. A July 2026 audit established
> that the spiking pathway was inactive in the released implementation: the AdEx neurons never
> fired and the surrogate gradient underflowed to zero, so the demonstration covers the
> pipeline's construction rather than the contribution of its spiking components.

### 4. Findings Relative to the Guiding Questions, item 3

**Draft text**

> STAC V1 (Hybrid fine-tuning) established feasibility, and STAC V2 (Conversion-oriented
> prototype) demonstrated neuromorphic-compatible execution with software-level latency and
> efficiency proxies […]

**Amend to**

> STAC V1 (hybrid fine-tuning) established the training pipeline but not the spiking
> mechanism, which was inactive as released; STAC V2 (conversion-oriented prototype)
> demonstrated neuromorphic-compatible execution with software-level latency and efficiency
> proxies […]

### 5. Findings item 2 and the SCANAQ descriptions

**Draft text**

> Validated SCANAQ traits were operationalized into specific agent behaviors […]

and, in several places, "derived from validated scales" / "grounded in validated constructs".

**Issue.** The *source* scales are validated; the SCANAQ composite is not. `PROVENANCE.md` in
SCAN-Resources records that the instrument has undergone no reliability or validity testing,
that seven outputs rest on single items, and that Section G is re-anchored relative to the
PSS. The exegesis's own Limitations section says this correctly ("The instrument's
reliability and construct validity … require empirical validation"), so only the summary
wording overstates.

**Amend to** "SCANAQ traits, drawn from validated source scales, were operationalized …" in
item 2, and leave the "derived from validated scales" phrasing, which is accurate.

### 6. Appendix B

The exegesis reproduces the chapter's Appendix B in full, including the Section E rule
("Determine the single highest scoring item among Q22–Q26 …") and the Global Note ("Scores
are summed within each section …"). It therefore carries all five scoring errors listed in
Part II of [`corrigendum-combined.md`](corrigendum-combined.md). Replace the appendix with
`Forms/Appendix-B-Scoring-v2.0.md` from SCAN-Resources (scoring model 2.0.0, version DOI
10.5281/zenodo.22598618), or attach it as an erratum.

### 7. Reference list

| Entry | Draft | Issue | Correct |
|---|---|---|---|
| Tate, L., & Sanders, P. (2024). iLevyTate/SCAN (Version 1.0.0-alpha) | 10.5281/zenodo.**15867066** | That DOI is the *stac* 2.0.0.3 record (DataCite: "iLevyTate/stac: 2.0.0.3"), not SCAN. The Springer chapter cites SCAN 1.0.0-alpha correctly. | 10.5281/zenodo.14052885 |
| Tate, L. (2025a). STAC V1 implementation (Version 2.0.0.3) | 10.5281/zenodo.15867066 | Same defect as the chapter: the tag has no V1 implementation, and the DOI is shared with (2025b). | Tate, L. (2026). STAC: Spiking Transformer Augmenting Cognition (Version 4.0.0). 10.5281/zenodo.22554655; in-text (2025a) → (2026) |
| Tate, L. (2025b). STAC V2 implementation (Version 2.0.0.3) | 10.5281/zenodo.15867066 | Unchanged once (2025a) moves. | as is |
| Kennedy, B. (2025a). Aligned minds and efficient machines … [Manuscript in preparation] | — | Superseded by Kennedy, B. (2026), the published chapter, which is also listed. Delete or merge. | — |

Checked 2026-09-09: 14052759 (SCANUE 1.0.0-alpha), 14053203 (SCAN-Resources 1.0.0-alpha),
14052885 (SCAN 1.0.0-alpha), 14510407 (scanue-v22), 14545341 (stac 1.0.2.1-alpha) and
15867066 (stac 2.0.0.3) all resolve through DataCite to the records named here.

## Unaffected

The exegesis already hedges the energy claim ("projected energy savings that await rigorous
hardware validation"; "energy efficiency claims are based on spike-count proxies … rather
than empirical hardware measurements"), and its Limitations section states the SCANAQ
validation gap. Those passages need no change. The three figure captions (STAC V1
architecture, STAC V2 workflow, SCANAQ mapping) describe designs, not results.
