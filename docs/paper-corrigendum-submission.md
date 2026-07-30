# Corrigendum submission — ready to send

Companion to [`corrigendum-2026-07.md`](corrigendum-2026-07.md), which holds the technical
account. This file holds the text to send to the publisher.

Fill in `[TITLE]`, `[VENUE]`, `[DATE]`, `[DOI]` before sending.

---

## Cover note to the editor

> I am writing to request a corrigendum to "[TITLE]", published in [VENUE] on [DATE], DOI
> [DOI].
>
> A code audit of the referenced repository established that the STAC V1 spiking pathway was
> inactive in the released implementation. Two coupled defects were responsible: the AdEx
> neurons received input current roughly two orders of magnitude below their firing
> threshold, and the Gaussian surrogate gradient underflowed float32 to exactly zero at the
> resulting operating point. Each independently prevented spiking; together they were
> mutually masking, since the sparsity metric that would have revealed the problem reads
> zero under both a maximally efficient network and a silent one.
>
> No published quantitative result is affected. The paper reports no STAC V1 perplexity,
> accuracy, or spike-rate figures, and the STAC V2 results and hardware-validation caveats
> are unchanged. The corrections are confined to statements of mechanism in the abstract and
> the STAC V1 methodology section, plus one reference version correction.
>
> Both defects are fixed in the current repository, the corrected behaviour is covered by a
> regression baseline, and the defect is reproducible via a script included in the
> repository (`scripts/verify_v1_corrigendum.py`).
>
> The paper commits to publishing failure reports as a governance practice; this correction
> is submitted in that spirit.

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

**Corrected** — repoint to a version that contains the artifact. Either cite the commit
directly, or mint a release tag from it and cite that.

---

## Scope statement (include if the venue asks)

The following are **unaffected** and require no correction:

- All STAC V2 material: the conversion pipeline, disabled-spiking disclosure, T=1
  disclosure, and the 3–4× software-latency figure, which was explicitly scoped to
  PyTorch-profiler execution time rather than joules.
- Every hardware-validation caveat, which the paper already states.
- All SCANAQ and SCANUE material.
- STAC V1's architecture, training pipeline, and HEMM design, which are as described. The
  model did train; the non-spiking components (GPT-2 backbone, projections, HEMM, lm_head)
  are differentiable and untouched by the defect. Only the spiking mechanism's contribution
  is withdrawn.
