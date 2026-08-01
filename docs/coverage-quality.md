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

**Re-measured after the fixes** (RoPE, attention encoding, RMSNorm — see §4's fix note):
the coverage rows are unchanged, as expected, since none of the fixes touch the
`SpikeLinear` path on GPT-2. The attention rows moved substantially:

| Level | pre-fix ppl | post-fix ppl |
| --- | ---: | ---: |
| spiking-attn only | 14,493 (270×) | **2,544 (47×)** |
| all + spiking-attn | 2,857 (53×) | **2,291 (43×)** |

The calibrated encoding makes spiking attention **5.7× less damaging**, and adding it on
top of full coverage now costs ~4% instead of worsening it by 30% — the best
fully-spiking configuration is now 98.6% coverage at 8.83× better projected energy.
The collapse itself persists (6–7 unique predictions over 256 positions), so the
conclusion below stands: encoding fidelity through ~37 compounding layers, not any single
component, is the obstacle, and calibration-only conversion does not clear it.

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

## 4 · A separate, serious defect: conversion discards RoPE — **fixed**

> **Fix status (2026-07).** Both defects in this section are fixed on this branch.
> `SpikeAttention` now applies RoPE (host-passed `position_embeddings`, or a carried-over
> `rotary_emb` for older layouts), and the normalization pass replaces RMSNorm via
> `SpikeRMSNorm`. Measured after the fix, conversion-only perplexity: SmolLM2-135M
> 20.79 → 20.79, SmolLM2-360M 15.78 → 15.78 — **1.00× on both**, from 19.0× / 27.6×.
> Pinned by `tests/test_rope_fidelity.py`. The §3 neuron defect is also fixed:
> `SpikeAttention` now uses the calibrated signed soft-reset IF encoding, with per-layer
> Q/K/V thresholds set by `calibrate_spike_attention()`. The measurements below are kept
> as the record of the pre-fix behaviour.

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

## 5b · The collapse survives every fix (confirmation)

The measurements in §§1–4 predate the RoPE, encoding, and RMSNorm fixes. Re-running on the
fixed pipeline confirms the conclusion is not an artifact of those bugs:

- **distilgpt2**, full coverage, T=8: 6–7 unique predictions over 256 positions, top-1
  agreement with the ANN 0.03. Unchanged from pre-fix.
- **SmolLM2-135M**, full coverage vs. the untouched ANN, T=8: perplexity 897× as-is, and a
  single fitted scalar (0.5) closes **77% of the gap** to 206×. But top-1 agreement is
  0.027 and next-token accuracy 0.0147 — the calibration helps perplexity, not prediction.
  The model still collapses.

The 77% figure is worth noting against GPT-2's 0% (§3.1): with RoPE restored, more of the
SmolLM2 damage is scale rather than shape, so `logit_scale` is now worth fitting — but it
is cosmetic, not curative. The predictions are gone either way.

**Where the error enters.** Per-block correlation between the spiking and dense residual
streams (distilgpt2, coverage-only, T=8):

| block | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| corr | 0.88 | 0.43 | 0.42 | 0.45 | 0.58 | 0.78 |

This is **not** smooth compounding. Block 0 converts cleanly; the discriminative signal
falls off a cliff at block 1 and never recovers (the rise at block 5 is the shared
high-norm residual, not restored prediction).

**Where the damage originates.** Feeding each block the *clean* dense input in isolation
(rate-decoded over T=8) separates a block's own transform fidelity from amplified upstream
error:

| block | cumulative corr | isolated corr | reading |
| --- | ---: | ---: | --- |
| 0 | 0.88 | 0.88 | own transform (mildly lossy) |
| 1 | 0.43 | **0.52** | **own transform (the cliff)** |
| 2 | 0.43 | 0.98 | fine in isolation — fed corrupted input |
| 3 | 0.45 | 0.97 | amplification |
| 4 | 0.58 | 0.97 | amplification |
| 5 | 0.78 | 0.97 | amplification |

The result is sharp: **blocks 2–5 convert almost perfectly on clean input (0.97+).** The
damage originates entirely in blocks 0 and 1 — block 1 worst at 0.52 — and everything
downstream is a faithful converter fed a corrupted signal. The fix therefore does not need
to touch the whole network; it needs to make the first two blocks convert cleanly. Whether
that is achievable by spending more timesteps there (a precision fix, and exactly the
paper's "graduated spiking" track) or requires fine-tuning is the next measurement.

This remains one model and a coarse proxy, but it is now a located defect, not a diffuse
one.

**More timesteps do not rescue it.** If the early-block damage were rate-code precision,
spending more timesteps there would recover it (the paper's "graduated spiking" idea).
Isolated block-1 fidelity across T:

| T | blk0 | blk1 | blk2 | blk3 | blk4 | blk5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.885 | 0.521 | 0.976 | 0.968 | 0.968 | 0.973 |
| 64 | 0.871 | **0.476** | 0.938 | 0.956 | 0.968 | 0.975 |

Block 1 gets slightly *worse* with more timesteps (the no-leak IF's drift accumulating),
never better. **Graduated timesteps is falsified as a fix.** The information block 1's
transform loses to binarisation is not recoverable by averaging more spikes.

**Sublayer split is suggestive but confounded.** Isolating attention vs MLP *contributions*
(the additive deltas, T=8):

| block | attn Δ corr | mlp Δ corr |
| --- | ---: | ---: |
| 0 | 0.98 | 0.57 |
| 1 | 0.47 | 0.11 |
| 2 | 0.58 | 0.25 |

Both of block 1's contributions convert poorly, and block-1 attention (0.47) is far worse
than block-0 attention (0.98) — consistent with early layers forming sharp, content-
specific attention that binarisation blurs. But these correlations are on the *deltas*,
which are small against the residual stream, so they cannot be ranked directly against the
block-level numbers above; read them as "which contribution is hardest to spike," not as a
quality attribution. The clean, load-bearing result is the block-level isolation, not this.

## 5c · Every post-hoc knob has now been tried

| Knob | Result |
| --- | --- |
| coverage (5% → 98%) | quality *improves* slightly; collapse persists |
| logit calibration | 0% of gap on GPT-2, 77% on SmolLM2 but predictions still dead |
| membrane leak (τ) | monotonically worse |
| timesteps (8 → 64) | no help; early blocks slightly worse |
| RoPE / encoding / RMSNorm fixes | conversion faithful (1.00×), collapse persists |

Post-hoc conversion of a frozen transformer is exhausted. The damage is a located,
precision-independent information loss in the first two blocks, and nothing that leaves the
weights frozen recovers it. Spike-aware fine-tuning — the paper's third roadmap track — is
what remains, and it is now demonstrated to be necessary rather than assumed.

## 5d · Training reverses the collapse (proof of concept)

Every knob in §5c leaves the weights frozen. Letting them move is the one untried path, and
it works. `scripts/finetune_spiking.py` trains the fully-converted network end to end
through the T-timestep spiking forward (gradients reach the weights via the neurons'
surrogate gradient), distilling from the original ANN. On distilgpt2, T=4, ~13 minutes on a
4-CPU box:

| step | eval perplexity |
| ---: | ---: |
| 0 (frozen collapse) | 6,161 |
| 75 | 1,057 |
| 150 | 811 |
| 225 | 635 |
| 300 | 580 |

A **10.5× recovery** in 300 steps, monotonic and still falling. This is a proof of concept,
not a solved problem: 580 is still ~11× the ANN's ~54, at a small T, short sequences, and a
few hundred CPU steps. But the contrast with §5c is the whole point — no frozen-weight
remedy moved perplexity at all, and training moves it immediately and substantially. It
confirms the diagnosis: the information the spike quantisation destroys can be *relearned*,
it just cannot be recovered from frozen ANN weights. A conclusive run (T=8, longer
sequences, thousands of steps, GPU) is future work, but the direction is no longer in doubt.

## 6 · What this means

**Post-hoc conversion without training does not work.** Not "works with degradation" —
produces a near-constant function. Three plausible rescues were tested and none moved it,
and the collapse persists after the RoPE / encoding / RMSNorm fixes (§5b), so it is a
property of spike-quantising a frozen network, not a consequence of the bugs those fixes
removed. The conversion is now faithful (1.00× with spiking off) and still collapses once
spiking is switched on — which is the cleanest possible statement of the result.

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
