# Does the model survive the spike coverage an energy advantage needs?

[`energy-crossover.md`](energy-crossover.md) established that an energy advantage requires
~90%+ of MACs to be spike-driven, and left the quality question open. This answers it.

**No. Post-hoc conversion does not degrade the model — it collapses it.** The converted
network emits near-constant output: 6 distinct predictions across 256 positions on
distilgpt2, with a comma accounting for 86.7% of them. Three separate hypotheses for
rescuing it were tested and all three failed.

Reproduce with `scripts/coverage_quality_sweep.py` and `scripts/logit_calibration_probe.py`.

---

## 1 · The headline

distilgpt2, WikiText-2, T=8, sliding-window perplexity. ANN baseline **53.64**.

| Level | Coverage | T=8 ppl | T=32 ppl | Energy @ T=32 |
| --- | ---: | ---: | ---: | --- |
| ANN baseline | — | 53.64 | 53.64 | — |
| spiking-attn only *(existing code)* | 1.4% | 14,493 | 14,546 | 31.5× worse |
| mlp | 33.9% | 4,184 | 3,678 | 21.1× worse |
| mlp + projections | 50.9% | 2,859 | 4,190 | 15.7× worse |
| **mlp + projections + lm_head** | **97.2%** | **2,193** | 3,272 | 1.10× better |
| all + spiking-attn | 98.6% | 2,857 | 4,622 | 2.21× better |

The energy target is reachable. The model that reaches it does not work.

---

## 2 · It is collapse, not degradation

Perplexity alone understates this. Prediction diversity over 256 positions, full coverage,
T=8:

| Model | unique predictions | most-frequent token |
| --- | ---: | --- |
| distilgpt2 — ANN | 100 | ` in` (9.4%) |
| distilgpt2 — SNN | **6** | `,` (**86.7%**) |
| SmolLM2-135M — ANN | 92 | ` in` (7.0%) |
| SmolLM2-135M — SNN | **15** | `\n` (23.8%) |

Next-token accuracy lands at **0.0127 on all three models tested** — the base rate of
frequent punctuation. The converted network is not a degraded language model; it is a
near-constant function that happens to emit common tokens.

---

## 3 · Three rescue hypotheses, all falsified

### 3.1 Calibration — no

Perplexity is exquisitely sensitive to logit scale while argmax is invariant to it, so a
network with the right shape and wrong magnitude reads as destroyed while its predictions
are intact. `TemporalSpikeProcessor` even carries a `logit_scale` parameter for this,
initialised to 1.0 and never fitted.

Measured on paired logits (distilgpt2, T=8, full coverage):

| | ANN | SNN |
| --- | ---: | ---: |
| logit std | 19.11 | 1.64 |
| top-1 agreement | — | 0.0500 |
| next-token accuracy | 0.2994 | 0.0100 |
| perplexity | 53.64 | 2,409.55 |
| perplexity at best fitted scale | — | 2,409.55 **at scale 1** |

**0.0% of the gap closes.** The optimal temperature is 1. No rescaling helps because the
ranking is wrong, not the scale.

### 3.2 Leak — no

A no-leak IF integrates input bias without bound. Adding a leak bounds that drift but
under-counts, since charge is lost between spikes. In the full network the trade goes
entirely the wrong way:

| τ | top-1 agreement | next-token acc | perplexity |
| --- | ---: | ---: | ---: |
| none (IF) | 0.0500 | 0.0100 | 2,410 (44.9×) |
| 32 | 0.0350 | 0.0055 | 2,753 (47.7×) |
| 8 | 0.0240 | 0.0005 | 3,738 (64.8×) |
| 2 | 0.0060 | 0.0000 | 26,652 (462×) |

Monotonically worse. Fidelity is the binding constraint and leak costs fidelity.

### 3.3 Scale — no

Larger models have more redundancy and might compound per-layer error more gently. Isolating
spiking from conversion (`--reference converted`, so RoPE loss and activation swaps are held
fixed on both sides), T=8, full coverage:

| Model | reference ppl | SNN ppl | cost of spiking alone |
| --- | ---: | ---: | ---: |
| distilgpt2 | 57.43 | 2,694 | 46.9× |
| SmolLM2-135M | 395.56 | 25,353 | 64.1× |
| SmolLM2-360M | 435.85 | 7,617 | 17.5× |

The cost does fall at 360M, but the model collapses at every scale (§2) and next-token
accuracy is 0.0127 throughout. Nothing here suggests a size at which this becomes viable.

---

## 4 · A separate, serious defect: conversion discards RoPE

`SpikeAttention` replaces the attention block wholesale and never applies rotary position
embeddings. Llama-family models — **every SmolLM2 variant the paper targets** — carry all
their positional information in RoPE. Converting them throws it away. The converter warns
about this; the cost had not been measured.

Conversion alone, no spiking at all:

| Model | positional scheme | ANN ppl | converted ppl | cost |
| --- | --- | ---: | ---: | ---: |
| distilgpt2 | learned embeddings | 53.64 | 57.43 | 1.07× |
| SmolLM2-135M | **RoPE** | 20.79 | 395.56 | **19.0×** |
| SmolLM2-360M | **RoPE** | 15.78 | 435.85 | **27.6×** |

GPT-2 keeps positional information in the embedding layer, which conversion does not touch,
so it loses essentially nothing. SmolLM2 loses 19–28× before any spiking is switched on.

This is independent of everything else in this document, and it is arguably more urgent:
the V2 pipeline is presented as a converter for SmolLM2, and on SmolLM2 it is lossy for
reasons unrelated to spiking, energy, or timesteps.

`SpikeLayerNorm` is likewise a no-op on these models — they use `LlamaRMSNorm`, which the
replacement pass does not match.

---

## 5 · Two metrics that report success on a dead network

Both were encountered directly while doing this work, and they are the same failure mode.

**Cosine similarity to the teacher's logits.** As quality collapsed across the leak sweep,
cosine similarity *rose* — 0.978 → 0.986 — while top-1 agreement fell 8× and accuracy went
to zero. Logit vectors over a 50k vocabulary are dominated by shared frequency structure
that every configuration preserves; the discriminative signal is a small deviation from it.
Cosine similarity is **anti-correlated with quality** here.

**L1 spike sparsity.** STAC V1 reported a spike penalty of exactly 0.0 for its entire life.
That is the best possible value of the metric and it meant the neurons never fired at all
(see [`corrigendum-2026-07.md`](corrigendum-2026-07.md)).

In both cases the convenient metric reads perfect precisely when the network has stopped
working. Any evaluation of this pipeline needs spike counts and prediction diversity
alongside them.

---

## 6 · What this means

**Post-hoc conversion without training does not work.** Not "works with degradation" —
produces a near-constant function. Three plausible rescues were tested and none moved it.

**Coverage is not the obstacle.** Quality *improves* as coverage rises (78× → 53× → 41× at
T=8), because each `SpikeLinear` calibrates its own threshold and adds per-layer scale
normalisation. Encoding fidelity compounding across ~37 layers is the obstacle: 0.99
per-layer correlation is not enough when the network is deep and nonlinear.

**For the roadmap.** Of the paper's three future tracks, quantization-aware conversion is
not a refinement — it is the critical path. Nothing in hardware validation or graduated
spiking addresses a model that emits commas.

**For the paper's existing claims, nothing here is contradicted.** The paper already states
that conversion reduces coherence and that spiking is disabled during inference to preserve
generation quality. This quantifies what enabling it would cost and *supports* that
decision. The RoPE defect in §4 is the one finding that may warrant a note, since it affects
the converter's behaviour on the paper's headline model with spiking switched off.

---

## 7 · What this does not establish

- **No fine-tuning anywhere.** Every number is post-hoc conversion of frozen weights. §6's
  conclusion is that training is the missing ingredient — it is untested here, not shown to
  fail. This is the single largest gap.
- **One encoding family.** Calibrated signed soft-reset IF, with and without leak. Other
  schemes exist: per-layer bias correction, spike-timing codes, threshold calibration on
  spiking rather than ANN statistics.
- **Models up to 360M.** SmolLM2-1.7B was not run. §3.3 shows the cost falling from 135M to
  360M, so a scale argument is not fully closed — though collapse at every size tested makes
  it an unpromising direction.
- **Perplexity and prediction diversity only.** No coherence rating, no downstream tasks.
- **An earlier version of this document attributed part of the damage to DC drift on the
  strength of a burn-in experiment.** That experiment was confounded: during burn-in the
  deep layers emit almost nothing, so discarding those timesteps changes the effective logit
  scale rather than isolating drift. Single-neuron drift is +0.03 against the +0.135
  measured at layer 5, so DC accumulation was never the dominant effect. The attribution has
  been withdrawn.
