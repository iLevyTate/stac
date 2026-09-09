# Corrigendum — send-ready draft to the publisher

A clean, copy-paste draft of the correction request for the book chapter, plus a
verification section so the editor (or you) can independently confirm every claim. The
technical account lives in [`corrigendum-2026-07.md`](corrigendum-2026-07.md); the annotated
change list in [`paper-corrigendum-submission.md`](paper-corrigendum-submission.md). This
file was the thing to send for the STAC V1 correction alone.

> **Superseded.** The chapter also needs its Appendix B scoring model corrected (tracked in the
> SCAN-Resources repository). Both corrections target the same chapter DOI, so they are
> submitted together as one letter: [`corrigendum-combined.md`](corrigendum-combined.md).
> Send that file. This one is kept as the STAC-only source it was merged from.

---

## Publication details

- **Chapter:** "Aligned Minds, Efficient Machines: Integrating Neuromorphic Computing for
  Personalized AI" — Ben Kennedy, Capitol Technology University
- **Book:** *Innovative Decision-Making in Engineering: The Role of Cognition, Heuristics,
  and Human Factors*, ed. Zian Shah Kabir
- **Publisher:** IGI Global Scientific Publishing, Hershey, PA · © 2026 · pp. 113–152
- **Chapter DOI:** 10.4018/979-8-3373-5702-7.ch005 · e-ISBN: 979-8-3373-5702-7
- **Route:** the chapter is published online with a DOI, so this is a request for a
  **correction/erratum notice linked to the chapter's DOI landing page**. Submit via IGI
  Global's editorial/rights team (`www.igi-global.com/contact/`), copying the volume editor.

---

## Cover email (copy-paste)

> **Subject:** Correction request — chapter DOI 10.4018/979-8-3373-5702-7.ch005
>
> Dear Dr. Kabir and the IGI Global editorial team,
>
> I am writing to request a correction to my chapter, "Aligned Minds, Efficient Machines:
> Integrating Neuromorphic Computing for Personalized AI," in *Innovative Decision-Making in
> Engineering: The Role of Cognition, Heuristics, and Human Factors* (IGI Global, 2026,
> pp. 113–152; DOI 10.4018/979-8-3373-5702-7.ch005).
>
> A code audit of the repository cited in the chapter established that the STAC V1 spiking
> pathway was inactive in the released implementation. Two coupled defects were responsible:
> the AdEx neurons received input current roughly two orders of magnitude below their firing
> threshold, and the Gaussian surrogate gradient underflowed float32 to exactly zero at the
> resulting operating point. Each independently prevented spiking; together they were
> mutually masking, because the sparsity metric that would have revealed the problem reads
> zero both for a maximally efficient network and for a completely silent one.
>
> No published quantitative result is affected. The chapter reports no STAC V1 perplexity,
> accuracy, or spike-rate figures, and the STAC V2 material and hardware-validation caveats
> are unchanged. The corrections are confined to statements of mechanism in the abstract and
> the STAC V1 methodology section, plus one reference-version correction, all listed below.
>
> Both defects are now fixed in the repository, the corrected behaviour is covered by a
> regression baseline, and the original defect is reproducible from a script included there
> (`scripts/verify_v1_corrigendum.py`). A runnable notebook also reproduces the V2 findings
> end to end. The repository carries the full technical account (`docs/corrigendum-2026-07.md`).
>
> Whether this is best handled as proof corrections, an erratum linked to the chapter's
> online record, or a note in a future printing, I will follow your process. The chapter
> commits to publishing failure reports as a governance practice; this correction is offered
> in that spirit.
>
> With thanks,
> Ben Kennedy
> Capitol Technology University

---

## Corrections

### ① Abstract

**Published**

> STAC V1 demonstrated feasibility via a hybrid fine-tuning methodology that combines a
> pretrained transformer with spiking elements for sparse, event-driven learning (Tate, 2025a).

**Corrected**

> STAC V1 demonstrated feasibility of a hybrid fine-tuning methodology combining a pretrained
> transformer with spiking elements. A subsequent code audit established that the spiking
> pathway was inactive in the released implementation (see Corrigendum); the sparse,
> event-driven learning it was designed to provide was not achieved.

### ② STAC V1 methodology — surrogate gradients

**Published**

> This provides a valuable learning signal, allowing gradients to flow back through the
> spiking neurons and enabling the backpropagation algorithm to fine-tune both the synaptic
> weights and the neurons' intrinsic firing properties.

**Corrected**

> This was intended to provide a learning signal allowing gradients to flow back through the
> spiking neurons. A July 2026 audit of the released implementation established that this did
> not occur: the Gaussian surrogate was parameterized with unit width while its argument
> (V − V_th) is expressed in millivolts and remained 10–20 mV from threshold, so the kernel
> evaluated below the float32 subnormal limit and truncated to exactly zero. No gradient
> reached the spiking layer.

### ③ STAC V1 methodology — feasibility claim

**Published**

> STAC V1, as illustrated in the diagram in Figure 3, demonstrated the feasibility of
> creating a high-performance hybrid SNN transformer.

**Corrected**

> STAC V1, as illustrated in the diagram in Figure 3, established the architecture and
> training pipeline for a hybrid SNN transformer. As released, its spiking pathway did not
> activate, so the demonstration covers the pipeline's construction rather than the
> contribution of its spiking components.

### ④ STAC V1 methodology — spike regularization

**Add after** the sentence defining `L = L_CE + λ‖S‖₁`:

> *Audit note.* Because the AdEx neurons emitted no spikes in the released implementation, the
> L1 term λ‖S‖₁ evaluated to exactly zero throughout training and exerted no regularization
> pressure. The membrane potential settled at V_rest + I/(1+a), approximately 14.7 mV below
> threshold, against a required drive of (1+a)(V_th − V_rest) = 75. Reported training
> behaviour therefore reflects the cross-entropy objective alone.

### ⑤ Initial Results and Future Standards

**Strike** the clause "rather than the integrated L1 spike regularization used during STAC V1
fine-tuning" — the comparison has no referent, since that term was identically zero.

### ⑥ Reference — Tate (2025a)

**Published**

> Tate, L. (2025a). STAC V1 implementation in iLevyTate/stac (Version 2.0.0.3) [Computer
> software, Jupyter Notebook]. Zenodo. https://doi.org/10.5281/zenodo.15867066

**Issue** — tag `2.0.0.3` (commit `26e213d`, 2025-07-11) contains no V1 implementation, only
`stac-v1/README.md`. The notebook `stac-v1/stacv1.ipynb` was committed 2025-07-13 (`7b09d54`),
two days after the tag.

**Corrected** — cite the release that actually contains the V1 implementation (in the
`stac_v1/` package) and the corrected behaviour described in this corrigendum:

> Tate, L. (2026). STAC: Spiking Transformer Augmenting Cognition (Version 4.0.0) [Computer
> software]. Zenodo. https://doi.org/10.5281/zenodo.22554655

> `10.5281/zenodo.22554655` is the version DOI of the 4.0.0 Zenodo deposit (published
> 2026-09-06). STAC's other DOIs, for reference: the *concept* DOI is `10.5281/zenodo.14545340`
> (always resolves to the latest version); `10.5281/zenodo.18023657` is the *version* DOI of
> 3.0.0-beta, the last release with the inactive V1 pathway, and must not be cited as the
> corrected version. (`10.5281/zenodo.21856231` is a different project, SCANUE-V22 — not this
> one.)

Optionally add a retention note: *"Earlier releases up to 3.0.0-beta contained the inactive
V1 spiking pathway described above."*

---

## How to verify (offer this to the editor; use it yourself first)

Every claim above is reproducible from the cited repository. Two independent checks:

**1 · The V1 defect (the subject of this correction).**

```bash
python scripts/verify_v1_corrigendum.py
```

Runs the original notebook's neuron and layer code verbatim at its shipped parameters and
reports **0 spikes** of 262,144 neuron-timesteps, the membrane 14.70 mV below threshold, and
the L1 penalty at exactly 0.0 — then the post-fix behaviour (spike rate 0.147, non-zero L1).

**2 · The V2 findings, end to end, in the browser (no setup).**

Open the Colab notebook in the repository (`notebooks/stac_v2_colab.ipynb`) and run it top to
bottom. It converts a model (faithful with spiking off), shows the frozen-model collapse and
the energy projection with spiking on, and runs the spike-aware retraining that recovers
quality — the evidence behind the V2 statements in the chapter's scope note.

---

## Scope statement (include if the editor asks)

The following are **unaffected** and require no correction:

- All STAC V2 material: the conversion pipeline, the disabled-spiking disclosure, the T=1
  disclosure, and the 3–4× software-latency figure, which was explicitly scoped to
  PyTorch-profiler execution time rather than joules.
- Every hardware-validation caveat, which the chapter already states.
- All SCANAQ and SCANUE material.
- STAC V1's architecture, training pipeline, and HEMM design, which are as described. The
  model did train; the non-spiking components (GPT-2 backbone, projections, HEMM, lm_head)
  are differentiable and untouched by the defect. Only the spiking mechanism's contribution
  is withdrawn.
