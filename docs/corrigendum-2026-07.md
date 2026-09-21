# Corrigendum — STAC V1 spiking pathway was inactive as released

**Date:** 2026-07
**Affects:** STAC V1 (`stac_v1/`, and the `stac-v1/stacv1.ipynb` notebook it was
consolidated from) in every release up to and including `3.0.0-beta`.
**Does not affect:** STAC V2, its conversion pipeline, or its reported latency figures.

---

## Summary

In the released STAC V1 implementation, the AdEx spiking layer emitted **zero spikes for
every input**. The component was structurally present, was constructed and invoked on every
forward pass, and reported no error — but it never fired, and no gradient ever reached it.

Two defects were jointly responsible. Either alone is sufficient to hold spike output at
zero, and each conceals the other:

**1 · The operating point lay below threshold.** The injected current came from an untrained
`nn.Linear` over transformer hidden states, giving `I = O(1)`. The AdEx sub-threshold fixed
point is

```
V* = V_rest + I / (1 + a)
```

so firing requires `I ≥ (1 + a)(V_th − V_rest)`. At the shipped parameters
(`V_th = −50`, `V_rest = −65`, `a = 4.0`) that threshold is **75**, against an observed
maximum injected current of **2.4**. The membrane settled ~14.7 mV below threshold and
stayed there.

**2 · The surrogate gradient underflowed.** The backward pass used a unit-width Gaussian

```
g(x) = exp(−x² / 2) / √(2π),    x = V − V_th  [mV]
```

The argument is in millivolts while the kernel width is 1. At `|x| ≈ 14.7` the kernel
evaluates to ≈`3e−48`, below the float32 smallest subnormal (`1.4e−45`), so it truncates to
**exactly zero**. The Jacobian of the spiking layer was structurally zero, not merely small.

## Why it went unnoticed

The metric that would have exposed this reads zero under both a maximally efficient network
and a completely silent one. The L1 sparsity term `λ‖S‖₁` evaluated to exactly `0.0`
throughout training, which is indistinguishable from a perfectly sparse solution unless the
raw spike count is also inspected. No spike-count telemetry existed at the time.

## Reproduction

`scripts/verify_v1_corrigendum.py` runs the original notebook's neuron and layer code
verbatim, at its shipped parameters, and reports the spike count:

```bash
python scripts/verify_v1_corrigendum.py
```

A second script traces the consequence downstream:

```bash
python scripts/verify_v1_downstream.py
```

Observed on the pre-fix code:

| Quantity | Value |
| --- | --- |
| Spikes emitted | **0** of 262,144 neuron-timesteps |
| Membrane potential vs. threshold | 14.70 mV below |
| Current required to fire | 75.0 (observed max: 2.4) |
| L1 spike penalty `λ·mean\|S\|` | exactly `0.00000000` |
| Surrogate gradient at operating point | exactly `0.0` |
| Gradient reaching the GPT-2 backbone | exactly `0.0` |
| Logit difference between two unlike inputs | exactly `0.0` |
| Distinct tokens predicted across a whole sequence | **1** |

## What is and isn't invalidated

> **Revised 2026-09-14.** The first version of this section said the rest of the model "still
> trained" and that V1's loss figures were "a real measurement of that pipeline". That was
> wrong, and wrong in the direction that flattered the work. `scripts/verify_v1_downstream.py`
> traces what the dead spiking layer did to everything behind it. The corrected account follows;
> the marked copy returned to the publisher on 2026-09-14 carries it.

**Invalidated** — any claim that STAC V1 demonstrated *functioning* surrogate-gradient
training of a spiking transformer, that gradients flowed through its spiking neurons, that
its intrinsic firing properties were fine-tuned, or that its L1 term produced sparse spiking
activity.

**Also invalidated, on the second look.** `DLPFCLayer` returns only spike tensors; no residual
path carries the GPT-2 hidden state past it. With the spike train uniformly zero, three further
things follow, and none of them were in the first draft of this document:

| Claim | What the released code did |
| --- | --- |
| The GPT-2 backbone was fine-tuned | It received a gradient of **exactly zero**. It sat in `AdamW(model.parameters())` and never moved, because the only path from it to the loss runs through a layer whose Jacobian is structurally zero. |
| The HEMM let recent activity influence current processing | `torch.mean` over a zero spike tensor is zero; the projection of zero is zero; the MLP returned its bias path. The memory bias was **one constant vector**, identical for every position and every input. |
| V1's loss and perplexity figures measure the hybrid pipeline | `combined = spk_trains + memory_bias` was that same constant, so the head saw identical input for every token of every sequence. Two deliberately unlike inputs give **bit-identical logits**. As released, V1 was a constant predictor: it could learn the unigram distribution of its training data and nothing else. |

Only the HEMM MLP's **bias** terms and the `lm_head` received gradient, and both were fed a
fixed vector. Any recorded V1 loss curve is a real number, but it measures a unigram model,
not the architecture the paper describes.

**Not invalidated** — the architecture and the training pipeline stand as *designs*. Nothing
here says the HEMM or the AdEx layer is a bad idea; the current release runs both with the
defects fixed. What fails is every statement in the past tense about what the released
implementation did.

## Fixes

| Defect | Fix |
| --- | --- |
| Sub-threshold operating point | `CurrentDrive` module — `gain·LayerNorm(x) + offset` with `offset = (1+a)(V_th − V_rest)` — centres the population steady state on threshold, so roughly half the neurons are active at initialisation |
| Surrogate gradient underflow | Gaussian width `σ` tied to `delta_T`, the AdEx model's own exponential-slope factor, so the surrogate's support matches the voltage scale over which the escape rate varies |
| No spike telemetry | `spike_metrics.py` counts real spikes and synaptic operations per forward pass |
| No regression guard | `tests/test_v1_baseline.py` asserts against `docs/baselines/stac_v1_smoke.json` |

Post-fix measured behaviour: spike rate **0.147**, sparsity **86.8%**, L1 term
**1.47e−06** — non-zero, and therefore actually exerting pressure.

## Effect on the accompanying paper

The paper reports **no quantitative STAC V1 results** — no perplexity, no accuracy, no spike
rates. It claims only "feasibility." No published number requires correction. The
corrections are confined to statements of mechanism:

| Location | Published claim | Status |
| --- | --- | --- |
| Abstract | V1 combined a pretrained transformer with spiking elements "for sparse, event-driven learning" | Intended, not achieved |
| V1 methodology | "allowing gradients to flow back through the spiking neurons" | Did not occur |
| V1 methodology | "enabling backpropagation to fine-tune… the neurons' intrinsic firing properties" | Did not occur |
| V1 methodology | "demonstrated the feasibility of creating a high-performance hybrid SNN transformer" | Overstated: covers pipeline construction, not spiking contribution |
| Spike regularization | Total loss `L = L_CE + λ‖S‖₁` | Second term evaluated to exactly zero |
| Initial Results | "the integrated L1 spike regularization used during STAC V1 fine-tuning" | No referent |
| V1 overview (typeset p. 118) | "established a complete, end-to-end differentiable pipeline" | Not differentiable end to end: the spiking layer's Jacobian was structurally zero |
| V1 overview (typeset p. 118) | "The maturity of this version was confirmed by a comprehensive validation suite of seven distinct test functions" | The suite passed on a silent network; sparsity metrics read their best value there |
| Background (typeset p. 120) | "a hybrid model that fine-tunes a pre-trained transformer and SNN model" | The backbone received exactly zero gradient |
| V1 methodology (typeset p. 133) | "The crucial innovation … was making these parameters learnable" | Declared learnable; received no gradient, so never updated |
| HEMM (typeset p. 133) | "allows the model's own recent past activity to influence its current processing" | Pooled a zero spike train; the memory bias was a constant |
| V2 pivot (typeset p. 134) | "While STAC V1 demonstrated feasibility" | Feasibility of the spiking mechanism was not demonstrated |

### A second class of overstatement, found 2026-09-14

Sweeping the typeset chapter for claim language rather than for the passages the corrigendum
already knew about turned up a defect of a different kind. Four sentences describe the SCANAQ,
or the mapping built on it, as *validated*:

| Typeset page | Text |
| --- | --- |
| 119 | "a **validated** psychometric-to-agent mapping methodology" |
| 123 | "creates a direct, **empirically validated**, and clinically relevant link" |
| 129 | "the **validated** SCANAQ provides a clear and actionable pathway" |
| 137 | "captured through the **validated** SCANAQ assessment" |

The chapter is right that the eight source scales are validated instruments, and the sentences
on pp. 118 and 121 that say so need no change. These four transfer that standing to the
composite. `PROVENANCE.md` in SCAN-Resources is explicit that it does not hold: the SCANAQ is an
ad-hoc composite; item subsets were taken, so the source instruments' reliability, norms and
cutoffs do not apply; Section G is anchored 1–5 against the PSS's native 0–4, shifting every
total by three points; and seven scored outputs rest on a single item, where reliability is
undefined. No validation study of the SCANAQ or of the mapping exists.

The same claim appears once in the Springer SEET chapter ("SCANAQ … has been validated through
current research") and is already listed for the exegesis in `exegesis-corrections.md` item 5.
It was corrected on scanerad.com on 2026-09-09. The IGI letter of 2026-09-13 did not raise it;
the marked copy of 2026-09-14 does, as its lowest-priority group, so that it cannot hold up the
mechanical corrections.

### Verification log

| Date | Check | Result |
| --- | --- | --- |
| 2026-09-07 | `scripts/verify_v1_corrigendum.py` on `main` (fabab59), torch CPU build | 0 of 262,144 spikes; gap 14.70 mV; max current 2.3988 vs 75.0 required; L1 = 0.00000000; surrogate gradient 0.000e+00. Exit 0. |
| 2026-09-07 | Published passages ①–⑤ diffed against the submitted manuscript with appendices (Edit12, 2025-10-12) | All five match verbatim. Typeset chapter PDF not yet checked. |
| 2026-09-09 | Appendix B Tables B2 and B3 of the manuscript extracted cell by cell (PyMuPDF `find_tables`) and the page rendered to an image and read | Row 5B: "High Motor Impulsivity, Low Non-Planning Impulsivity", 6–12, items 13–18, "Acts impulsively; limited advance planning." Row 2B: "Low Cognitive Reappraisal, High Expressive Suppression", 4–19; row 2A: 20–28. Corrections ② and ③ confirmed on direct reads, closing the column-alignment caveat from PR #23. |
| 2026-09-09 | The other IGI chapter ("Beyond Intelligence…", *Ensuring Secure and Ethical STM Research in the AI Era*) searched for Appendix B material and reference defects | No appendices, no scoring tables, SCANAQ mentioned only as upcoming, STAC only as future work; all four Zenodo DOIs it cites resolve. No correction needed. |
| 2026-09-09 | The Springer chapter ("Synthetic Cognitive Augmentation Network", *SEET 2025*, CCIS 2725, DOI 10.1007/978-3-032-08977-9_13) read from the SEET submission manuscript and its Crossref reference deposit | STAC named only as a future component with a pipeline figure; no V1 mechanism claim, no Appendix B. Three Zenodo references (14052759, 14053203, 14052885) resolve via DataCite. One soft overstatement ("SCANAQ … has been validated through current research"). No correction needed. |
| 2026-09-09 | The PhD exegesis (Drive draft of 2025-12-11) searched for the same claims | Restates the V1 feasibility claim in four passages, reproduces Appendix B (scoring model 1.0.0) in full, and cites SCAN 1.0.0-alpha under the stac 2.0.0.3 DOI (15867066 instead of 14052885). Needs its own amendment: `exegesis-corrections.md`. |
| 2026-09-09 | Crossref metadata for the chapter DOI fetched (`api.crossref.org/works/10.4018/979-8-3373-5702-7.ch005`) | Published 2025-11-20; references deposited 2026-08-27, 40 entries. Entries 34–36 carry the truncated `10.5281/zenodo.140532` and the duplicated `10.5281/zenodo.15867066`, confirming those two defects in the published record. Entry 21 is Ostrau et al. (2022), which the submitted manuscript lacked: the typeset chapter may carry it. Letter reworded to make the Ostrau item conditional. No `update-to` (erratum) relation exists yet. |
| 2026-09-14 | The published chapter PDF (40 pages, printed pp. 113–152, Adobe InDesign 20.0) supplied by the proofing desk, read directly and diffed against the manuscript quotations | The Ostrau et al. (2022) entry is present on printed p. 142: that item is withdrawn. Three of the five Part I quotations do not match the printed text, the Abstract's "for sparse, event-driven learning" having been removed in copy-editing. Two further locations need correction (printed pp. 118 and 134) that no manuscript-based draft covered. |
| 2026-09-14 | `scripts/verify_v1_downstream.py`, the shipped notebook's layer and neuron code traced past the spiking layer | 0 spikes; gradient to the upstream projection exactly 0.0; AdEx parameter gradient exactly 0.0; HEMM MLP weight gradient exactly 0.0 while its bias trains; bit-identical logits for two unlike inputs; one distinct predicted token across a whole sequence. The released V1 was a constant predictor. Exit 0. |
| 2026-09-14 | The shipped notebook (`stac-v1/stacv1.ipynb` at 7b09d54) read directly to confirm the structure the probe assumes | `DLPFCLayer.forward` returns only `torch.cat(spk_list)`, with no residual path; `DLPFCTransformer.forward` computes `combined_repr = spk_trains + memory_bias.unsqueeze(1)`; HEMM pools with `torch.mean(spike_train, dim=1)`; the optimizer is `AdamW(model.parameters())`, so the backbone was in scope and still received nothing. |
| 2026-09-14 | The whole typeset chapter swept for assertive claim language (validated, proven, confirmed, successful, ensures, guarantee) rather than for known passages | Four "validated SCANAQ" sentences found (pp. 119, 123, 129, 137), none previously raised with the publisher. Six further V1 mechanism claims found (pp. 118 ×3, 120, 133 ×2). Appendix B Sections D and F checked item by item and are correct as printed. Every other DOI in the reference list checked and resolves; the two defects already known are the only ones. |
| 2026-09-14 | 16 annotations written into the publisher's PDF with PyMuPDF and each anchor asserted to resolve; pages 113, 133 and 148 rendered at 110 dpi and read back | Every highlight lands on its intended passage; the summary note is visible on p. 113. Returned as `Aligned-Minds-Efficient-Machines-CORRECTIONS-MARKED.pdf`; see `corrigendum-marked-copy-2026-09-14.md`. |
| 2026-09-07 | CI workflow steps run locally on `main` (397c058), Python 3.11: flake8, compileall, import checks, `tests/test_v1.py`, pytest on the offline fixture | 8/8 V1 tests; 63 passed, 3 skipped. |
| 2026-09-07 | `notebooks/stac_v2_colab.ipynb` §3–§9 executed cell by cell on CPU (no GPU) | §4 spiking off: max logit diff 2.29e-05, top-1 100%. §5 spiking on: energy ratio 0.13× (SNN worse), logits differ 10.1 between T=1 and T=8. §6 both energy scripts exit 0. §7 SmolLM2-135M: diff 1.43e-05, top-1 100%. §8 CPU probe (T=2, seq 32, 50 steps, MLP only): eval perplexity 18,468.56 → 1,137.43. §9 generates. §3 initially showed 1 failed: the coherence test's absolute bar, which unconverted DistilGPT-2 also fails at 30%; made a parity test (see CHANGELOG), after which the suite passes in notebook mode too. |

### Submission

**Sent 2026-09-13** from `bkennedy1@captechu.edu` to `bookproofing@igi-global.com`, copying the
volume editor `ZianShah.Kabir@uts.edu.au` and `cust@igi-global.com`, subject "Correction request -
chapter DOI 10.4018/979-8-3373-5702-7.ch005", with `Appendix-B-Scoring-v2.0.pdf` attached. The text
as sent is [`corrigendum-sent-2026-09-13.md`](corrigendum-sent-2026-09-13.md); the working letter it
was drawn from is [`corrigendum-combined.md`](corrigendum-combined.md).

No ticket number was issued. The proofing desk returned an automatic reply at 23:08 UTC the same
day which carries a warning worth recording:

> Emails sent to the proofing inboxes (bookproofing@igi-global.com and
> journalproofing@igi-global.com) without a corresponding note in the proofing system will not be
> seen by the typesetter in time for the changes to be implemented.

That inbox is described in the same reply as "loosely monitored". The warning turned out not to
apply: a member of the proofing team replied on 2026-09-14, asking for the corrections as comments
inside the digital copy rather than described in prose, and supplying the published chapter PDF.
They also set the scope: revisions are guaranteed only for the digital copy and only if accepted,
and anything exceeding those limits goes to the editorial managers to consider a correction
erratum.

**Marked copy returned 2026-09-14**, the same day, with 15 numbered comments anchored on the
passages they concern and a summary note on printed p. 113. Reading the typeset text withdrew one
item, corrected three quotations, and added two locations the letter never covered:
[`corrigendum-marked-copy-2026-09-14.md`](corrigendum-marked-copy-2026-09-14.md).

**Marked copies sent 2026-09-21.** The ch005 reply (32 comments, with the re-rendered Appendix B 2.0)
went on the original thread; the ch007 request (8 comments) went as a new message with the
co-authors in copy. Both to the same proofing desk on the same day, by design. Record of contents:
[`corrigendum-marked-copy-2026-09-14.md`](corrigendum-marked-copy-2026-09-14.md) and
[`reference-audit-2026-09-17.md`](reference-audit-2026-09-17.md).

### Citation accuracy

The paper cites the V1 implementation as *iLevyTate/stac* **Version 2.0.0.3**. That tag
(`26e213d`, 2025-07-11) contains no V1 implementation — only `stac-v1/README.md`. The
notebook `stac-v1/stacv1.ipynb` was committed 2025-07-13 (`7b09d54`), two days after the
tag. The reference should point at a version that actually contains the artifact.
