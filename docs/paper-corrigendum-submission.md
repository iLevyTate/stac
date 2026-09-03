# Corrigendum submission — ready to send

Companion to [`corrigendum-2026-07.md`](corrigendum-2026-07.md), which holds the technical
account. This file holds the text to send.

The work is a **book chapter** in an IGI Global edited volume, so the correction is handled
by the **volume editor and IGI Global's editorial/production team**, not a journal desk.

**Publication details:**
- Chapter: "Aligned Minds, Efficient Machines: Integrating Neuromorphic Computing for
  Personalized AI" (Ben Kennedy, Capitol Technology University)
- Book: *Innovative Decision-Making in Engineering: The Role of Cognition, Heuristics, and
  Human Factors*, ed. Zian Shah Kabir
- Publisher: IGI Global Scientific Publishing, Hershey, PA · © 2026 · pp. 113–152
- Chapter DOI: 10.4018/979-8-3373-5702-7.ch005 · e-ISBN: 979-8-3373-5702-7

## How this chapter gets corrected

The chapter is published online with a DOI, so **route 2 below is the relevant one** — an
erratum/correction notice attached to the chapter's DOI record. IGI Global handles this
through their editorial/production team, usually with the volume editor copied.

1. **If still within a correction window / proofs:** send the corrections to the editor and
   IGI Global production as proof corrections — cleanest, avoids a formal erratum.
2. **Published online with a DOI (this case):** request a **correction/erratum notice linked
   to the chapter's DOI landing page** (10.4018/979-8-3373-5702-7.ch005). Contact IGI Global
   via their editorial/rights team (`www.igi-global.com/contact/`) and copy the volume editor.
3. **Print copies:** ask that an **erratum be carried in any subsequent printing or edition**.
   Regardless of what the print artifact can accommodate, the repository now carries the
   authoritative technical record (`docs/corrigendum-2026-07.md`), so anyone reaching the code
   from the chapter finds the correction.

---

## Cover note to the volume editor / publisher

> Dear Dr. Kabir and the IGI Global editorial team,
>
> I am writing to request a correction to my chapter "Aligned Minds, Efficient Machines:
> Integrating Neuromorphic Computing for Personalized AI" in *Innovative Decision-Making in
> Engineering: The Role of Cognition, Heuristics, and Human Factors* (IGI Global, 2026,
> pp. 113–152; DOI 10.4018/979-8-3373-5702-7.ch005).
>
> A code audit of the repository cited in the chapter established that the STAC V1 spiking
> pathway was inactive in the released implementation. Two coupled defects were responsible:
> the AdEx neurons received input current roughly two orders of magnitude below their firing
> threshold, and the Gaussian surrogate gradient underflowed float32 to exactly zero at the
> resulting operating point. Each independently prevented spiking; together they were
> mutually masking, since the sparsity metric that would have revealed the problem reads
> zero under both a maximally efficient network and a silent one.
>
> No published quantitative result is affected. The chapter reports no STAC V1 perplexity,
> accuracy, or spike-rate figures, and the STAC V2 material and hardware-validation caveats
> are unchanged. The corrections are confined to statements of mechanism in the abstract and
> the STAC V1 methodology section, plus one reference-version correction, all listed below.
>
> Both defects are now fixed in the repository, the corrected behaviour is covered by a
> regression baseline, and the defect is reproducible via a script included there
> (`scripts/verify_v1_corrigendum.py`). The repository also carries the full technical
> account (`docs/corrigendum-2026-07.md`).
>
> Whether this is best handled as proof corrections, an erratum linked to the chapter's
> online record, or a note in a future printing, I will follow your process. The chapter
> commits to publishing failure reports as a governance practice; this correction is offered
> in that spirit.

---

## Corrections

### ① Abstract

**Published**

> STAC V1 demonstrated feasibility via a hybrid fine-tuning methodology that combines a
> pretrained transformer with spiking elements for sparse, event-driven learning (Tate,
> 2025a).

**Corrected**

> STAC V1 demonstrated feasibility of a hybrid fine-tuning methodology combining a
> pretrained transformer with spiking elements. A subsequent code audit established that the
> spiking pathway was inactive in the released implementation (see Corrigendum); the sparse,
> event-driven learning it was designed to provide was not achieved.

### ② STAC V1 methodology — surrogate gradients

**Published**

> This provides a valuable learning signal, allowing gradients to flow back through the
> spiking neurons and enabling the backpropagation algorithm to fine-tune both the synaptic
> weights and the neurons' intrinsic firing properties.

**Corrected**

> This was intended to provide a learning signal allowing gradients to flow back through the
> spiking neurons. A July 2026 audit of the released implementation established that this
> did not occur: the Gaussian surrogate was parameterized with unit width while its argument
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

> *Audit note.* Because the AdEx neurons emitted no spikes in the released implementation,
> the L1 term λ‖S‖₁ evaluated to exactly zero throughout training and exerted no
> regularization pressure. The membrane potential settled at V_rest + I/(1+a), approximately
> 14.7 mV below threshold, against a required drive of (1+a)(V_th − V_rest) = 75. Reported
> training behaviour therefore reflects the cross-entropy objective alone.

### ⑤ Initial Results and Future Standards

**Strike** the clause "rather than the integrated L1 spike regularization used during STAC
V1 fine-tuning" — the comparison has no referent, since that term was identically zero.

### ⑥ Reference — Tate (2025a)

**Published**

> Tate, L. (2025a). STAC V1 implementation in iLevyTate/stac (Version 2.0.0.3) [Computer
> software].

**Issue** — tag `2.0.0.3` (commit `26e213d`, 2025-07-11) contains no V1 implementation, only
`stac-v1/README.md`. The notebook `stac-v1/stacv1.ipynb` was committed 2025-07-13
(`7b09d54`), two days after the tag.

**Corrected** — cite the `4.0.0` release, which contains the V1 implementation (in the
`stac_v1/` package) and the corrected behaviour described in this corrigendum:

> Tate, L. (2026). STAC: Spiking Transformer Augmenting Cognition (Version 4.0.0) [Computer
> software]. Zenodo. https://doi.org/10.5281/zenodo.[4.0.0-VERSION-DOI]

`[4.0.0-VERSION-DOI]` is the version-specific DOI minted when the v4.0.0 Zenodo deposit is
published (the Release workflow cuts the GitHub release that Zenodo archives). For reference,
STAC's concept DOI is `10.5281/zenodo.14545340`, which always resolves to the latest version;
`10.5281/zenodo.18023657` is the version DOI of 3.0.0-beta, the last release carrying the
inactive V1 pathway, and is not a substitute for the 4.0.0 DOI here. Optionally add a
retention note: *"Earlier releases up to 3.0.0-beta contained the inactive V1 spiking pathway
described above."*

---

## Scope statement (include if the editor asks)

The following are **unaffected** and require no correction:

- All STAC V2 material: the conversion pipeline, disabled-spiking disclosure, T=1
  disclosure, and the 3–4× software-latency figure, which was explicitly scoped to
  PyTorch-profiler execution time rather than joules.
- Every hardware-validation caveat, which the chapter already states.
- All SCANAQ and SCANUE material.
- STAC V1's architecture, training pipeline, and HEMM design, which are as described. The
  model did train; the non-spiking components (GPT-2 backbone, projections, HEMM, lm_head)
  are differentiable and untouched by the defect. Only the spiking mechanism's contribution
  is withdrawn.
