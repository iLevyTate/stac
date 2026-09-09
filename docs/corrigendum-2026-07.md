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

Observed on the pre-fix code:

| Quantity | Value |
| --- | --- |
| Spikes emitted | **0** of 262,144 neuron-timesteps |
| Membrane potential vs. threshold | 14.70 mV below |
| Current required to fire | 75.0 (observed max: 2.4) |
| L1 spike penalty `λ·mean\|S\|` | exactly `0.00000000` |
| Surrogate gradient at operating point | exactly `0.0` |

## What is and isn't invalidated

**Invalidated** — any claim that STAC V1 demonstrated *functioning* surrogate-gradient
training of a spiking transformer, that gradients flowed through its spiking neurons, that
its intrinsic firing properties were fine-tuned, or that its L1 term produced sparse spiking
activity.

**Not invalidated** — the model still trained. The GPT-2 backbone, the projection layers,
the HEMM memory bias and the `lm_head` are all differentiable and untouched by these
defects. Any loss or perplexity figure recorded from V1 is a real measurement of that
pipeline; it simply owes nothing to the spiking mechanism. The architecture, the training
pipeline and the HEMM design stand as described.

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
| 2026-09-07 | CI workflow steps run locally on `main` (397c058), Python 3.11: flake8, compileall, import checks, `tests/test_v1.py`, pytest on the offline fixture | 8/8 V1 tests; 63 passed, 3 skipped. |
| 2026-09-07 | `notebooks/stac_v2_colab.ipynb` §3–§9 executed cell by cell on CPU (no GPU) | §4 spiking off: max logit diff 2.29e-05, top-1 100%. §5 spiking on: energy ratio 0.13× (SNN worse), logits differ 10.1 between T=1 and T=8. §6 both energy scripts exit 0. §7 SmolLM2-135M: diff 1.43e-05, top-1 100%. §8 CPU probe (T=2, seq 32, 50 steps, MLP only): eval perplexity 18,468.56 → 1,137.43. §9 generates. §3 initially showed 1 failed: the coherence test's absolute bar, which unconverted DistilGPT-2 also fails at 30%; made a parity test (see CHANGELOG), after which the suite passes in notebook mode too. |

### Submission

Not yet sent. The combined letter is [`corrigendum-combined.md`](corrigendum-combined.md);
the remaining steps are in [`corrigendum-runbook.md`](corrigendum-runbook.md). Record the send
date, recipients, and any ticket number here once it goes.

### Citation accuracy

The paper cites the V1 implementation as *iLevyTate/stac* **Version 2.0.0.3**. That tag
(`26e213d`, 2025-07-11) contains no V1 implementation — only `stac-v1/README.md`. The
notebook `stac-v1/stacv1.ipynb` was committed 2025-07-13 (`7b09d54`), two days after the
tag. The reference should point at a version that actually contains the artifact.
