# Does the model survive the spike coverage an energy advantage needs?

[`energy-crossover.md`](energy-crossover.md) established that an energy advantage requires
~90%+ of MACs to be spike-driven, and deliberately left the quality question open. This
answers it.

**Short answer: no. Post-hoc conversion does not produce a usable model at any coverage or
timestep count tested.** The best result in the entire grid is 41× worse perplexity. The
energy advantage is real and reachable; the model that reaches it does not work.

Reproduce with `scripts/coverage_quality_sweep.py`.

---

## 1 · Setup

distilgpt2, WikiText-2 test split, 3000 tokens, sliding-window perplexity (window 256,
stride 128). Coverage extended with `spike_coverage.SpikeLinear` — a calibrated, signed,
soft-reset integrate-and-fire rate code that reaches correlation 0.9906 against the dense
layer at T=8 (`tests/test_spike_coverage.py`). ANN baseline perplexity: **53.64**.

The existing `SpikeAttention` spiking mode is reported separately from the added coverage,
because mixing them attributes its damage to coverage. That separation turned out to matter
a great deal — see §3.

---

## 2 · Results

| Level | Coverage | T=8 ppl | T=32 ppl | Energy @ T=32 |
| --- | ---: | ---: | ---: | --- |
| ANN baseline | — | 53.64 | 53.64 | — |
| spiking-attn only *(existing code)* | 1.4% | 14,493 | 14,546 | 31.5× worse |
| mlp | 33.9% | 4,184 | 3,678 | 21.1× worse |
| mlp + projections | 50.9% | 2,859 | 4,190 | 15.7× worse |
| **mlp + projections + lm_head** | **97.2%** | **2,193** | 3,272 | 1.10× better |
| all + spiking-attn | 98.6% | 2,857 | 4,622 | 2.21× better |

Best case: **41× worse perplexity** (53.64 → 2193), at T=8 and 97.2% coverage, where the
projection is 4.42× better on energy. That is a working energy result attached to a
non-working model.

---

## 3 · The existing spiking attention is the worst single component

It costs **270× perplexity to buy 1.4% coverage** — the worst trade in the table by a wide
margin. Adding it on top of otherwise-full coverage makes perplexity *worse*
(2193 → 2857 at T=8; 3272 → 4622 at T=32). It is actively harmful, and it is the only
spiking the pipeline currently enables.

The cause is its neuron. `SpikeAttention` constructs
`LIFNode(v_threshold=0.1, v_reset=0.0)` — leaky, hard reset, fixed uncalibrated threshold.
Measured against realistic Q/K/V (std ≈ 1):

| Encoding | corr. with `1[x > 0.1]` | corr. with true magnitude |
| --- | ---: | ---: |
| `LIFNode(0.1, hard reset)` — as shipped | **0.974** | 0.805 |
| calibrated signed soft-reset IF | — | **0.989** |

It is not rate-coding. It is a 1-bit quantizer at an arbitrary threshold, discarding
magnitude entirely. **This is a fixable defect**, independent of everything else here.

---

## 4 · More timesteps makes it worse

Quality *degrades* from T=8 to T=32 at almost every coverage level. This is backwards — a
rate code should sharpen with more timesteps, and in isolation this one does (correlation
0.9906 at T=8, 0.9991 at T=128).

Two effects, both measured:

**Propagation latency.** Per-timestep firing rate through the network:

| Layer | step 1 | step 5 | steady state |
| --- | ---: | ---: | ---: |
| `h.0.mlp` (early) | 0.112 | 0.109 | ~0.14 |
| `h.5.mlp` (deep) | **0.008** | 0.101 | ~0.23 |

Spikes need ~8 timesteps to reach layer 5. `TemporalSpikeProcessor` averages logits across
*all* T timesteps, including the burn-in during which deep layers emit almost nothing.

**DC drift.** The deep layer keeps climbing after the transient settles — 0.223 at step 8,
0.242 at step 32. A soft-reset IF has no leak, so any bias in its input accumulates on the
membrane indefinitely and the firing rate creeps upward.

Discarding the burn-in was the obvious fix. It does not work — it makes things worse:

| Burn-in discarded | Steps used | Perplexity |
| ---: | ---: | ---: |
| 0 | 32 | 3,499 |
| 4 | 28 | 4,524 |
| 8 | 24 | 4,638 |
| 16 | 16 | 4,699 |

Later timesteps are *less* accurate than early ones, so drift dominates latency. That
explains the inverted T behaviour: the network is drifting away from its correct operating
point as timesteps accumulate, and the early steps — before drift builds — are the good
ones.

---

## 5 · What this means

**For the architecture.** Conversion without training does not reach a usable model here.
Coverage is not the obstacle (quality actually *improves* with coverage — 78× → 53× → 41×
at T=8, because each `SpikeLinear` calibrates its own threshold, adding per-layer scale
normalisation). Encoding fidelity is the obstacle.

**For the roadmap.** The paper lists three future tracks: hardware validation, graduated
spiking, and quantization-aware conversion. On this evidence the third is not a refinement —
it is the critical path. Nothing in tracks 1 or 2 addresses a 41× perplexity gap.

**For the paper's existing claims.** Nothing here contradicts them. The paper already states
that conversion "reduce[s] conversational coherence and nuance" and that spiking is disabled
during inference to preserve generation quality. This quantifies how much would be lost if
it were enabled: at minimum 41× perplexity. The decision to ship with spiking off is
supported, not undermined, by this result.

---

## 6 · What this does not establish

- **One encoding scheme.** A calibrated signed soft-reset IF is a standard choice, not the
  only one. Per-layer bias correction, a tuned leak, or thresholds calibrated on spiking
  rather than ANN statistics could all do better. The DC-drift finding in §4 points at a
  specific, untried fix: a small leak would bound the accumulation.
- **No fine-tuning.** Every number is post-hoc conversion of frozen weights. The entire
  point of §5's roadmap conclusion is that this is the missing ingredient — it is untested
  here, not shown to fail.
- **One model, one corpus.** distilgpt2 on 3000 tokens of WikiText-2. Larger models have
  more redundancy and may degrade more gracefully; §4 of `energy-crossover.md` shows they
  also have far better coverage economics.
- **Perplexity only.** Not coherence, not downstream task performance. At a 41× gap this
  hardly matters, but it would if the gap narrows.
