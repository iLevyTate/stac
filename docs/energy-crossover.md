# When does STAC's SNN actually beat the ANN on energy?

`spike_metrics.py` reports that the current conversion projects **7.6× worse** than the
dense ANN it converts. This document answers the follow-up question: is that fixable within
this architecture, and if so, what has to change?

Short answer: **yes, but not by the route the roadmap currently proposes, and not on the
models currently being benchmarked.** The binding constraint is spike *coverage*, not
sparsity — and coverage improves substantially with model scale.

Reproduce anything here with `scripts/energy_analysis.py`.

---

## 1 · The model, and why you can trust it

For a linear layer whose input is a binary spike train at rate `ρ`, every MAC becomes a
spike-driven accumulate, so `SynOps = ρ · MACs` over the covered portion. With `f` the
fraction of total MACs that are spike-driven and `r = E_AC / E_MAC = 0.9/4.6 ≈ 0.196`:

```
E_ANN = M · E_MAC
E_SNN = T · (ρ·f·M·E_AC  +  (1−f)·M·E_MAC)

ratio = E_SNN / E_ANN = T · (ρ·f·r + 1 − f)
```

Requiring `ratio < 1` gives the two forms that matter:

```
required coverage     f*    = (T − 1) / (T · (1 − ρ·r))
affordable timesteps  T_max = 1 / (1 − f · (1 − ρ·r))
```

**Validation.** Against a measured `spike_metrics.py` run (tiny-gpt2, T=8, seq 32,
f=0.0555, ρ=0.0943): measured ratio **7.5567**, closed form **7.5641** — 0.1% relative
error. Pinned by `tests/test_energy_analysis.py::test_closed_form_matches_spike_metrics`.

---

## 2 · Coverage dominates sparsity

`ρ` enters the expression only through the product `ρ·r`, and `r ≈ 0.196`. Driving the
spike rate from 0.094 to **zero** — a perfectly silent network — raises `T_max` at 55%
coverage from 2.17 to 2.27. Raising coverage from 55% to 75% raises it to 3.53.

This cuts against the usual SNN framing, where sparsity is the headline lever. Here it is
close to irrelevant: an accumulate is only ~5× cheaper than a MAC, so avoiding operations
entirely (coverage) beats making them cheaper (sparsity) by a wide margin. Chasing lower
spike rates is not where the energy is.

---

## 3 · The current design has no winning spiking operating point

| Coverage scenario | Coverage | `T_max` | Ratio @ T=8 |
| --- | ---: | ---: | ---: |
| current: `QK^T` only | 2.75% | 1.03 | 7.8× worse |
| + `AV` matmul | 5.50% | 1.06 | 7.6× worse |
| + MLP layers | 38.50% | 1.60 | 5.0× worse |
| + attention projections | 55.01% | 2.15 | 3.7× worse |
| + `lm_head` (everything) | 100% | 34.77 | win |

*(distilgpt2, seq 512, ρ=0.094)*

`T_max = 1.03` means the only timestep count that breaks even is **T = 1** — which is not a
spiking network. As built, there is no operating point that is simultaneously genuinely
spiking and cheaper than the ANN. This is the negative result, and
`test_current_architecture_has_no_winning_spiking_operating_point` will fail if coverage
ever improves enough to invalidate it.

---

## 4 · Model scale changes the answer

`lm_head` costs `d·V` — linear in width. The transformer body costs `L·d²` — quadratic. So
the block that is hardest to spike shrinks as a share of total work as models grow:

| Architecture | `lm_head` share | Body coverage | `T_max` (body) | `T_max` (MLP only) |
| --- | ---: | ---: | ---: | ---: |
| distilgpt2 | 44.99% | 55.01% | 2.17 | 1.61 |
| gpt2 | 29.03% | 70.97% | 3.30 | 1.95 |
| SmolLM2-135M | 18.60% | 81.40% | 4.97 | 2.69 |
| SmolLM2-360M | 12.00% | 88.00% | 7.34 | 3.01 |
| **SmolLM2-1.7B** | **5.71%** | **94.29%** | **13.43** | 3.35 |

*(seq 512, ρ=0.094; "body" = everything except `lm_head`)*

On SmolLM2-1.7B — the model the paper actually targets — spiking the body reaches 94.29%
coverage and affords **T ≤ 13**, projecting **1.78× better** than the ANN at T=8.

> **Caveat on the Llama-family rows.** These are operation counts, and they hold as
> arithmetic. But `SpikeAttention` never applies rotary position embeddings, and every
> SmolLM2 variant carries its positional information in RoPE, so a converted SmolLM2 has
> already lost that information before any coverage or timestep argument applies — measured
> at 19–28× worse perplexity from conversion alone, with spiking switched off. See §4 of
> [`coverage-quality.md`](coverage-quality.md). The energy economics below describe a model
> that must be fixed before the economics matter.

**This means the 7.6× worse figure is substantially an artifact of benchmarking on tiny
models.** A conversion pipeline evaluated on distilgpt2 or tiny-gpt2 structurally
understates what the same design achieves at 1.7B, because `lm_head` dominates the small
model's MAC budget and dominates nothing at scale.

---

## 5 · The stated roadmap is necessary but not sufficient

The paper proposes *"graduated spiking strategies — selectively apply complete temporal
spike encoding to layers where energy benefits exceed quality costs (e.g. MLP layers) while
preserving continuous processing in attention mechanisms."*

Taking that literally — spike the MLP, leave attention continuous — reaches `T_max` of 1.61
(distilgpt2) to 3.35 (1.7B). At T=8 it is still 5.0× worse on distilgpt2 and 2.2× worse on
1.7B. **MLP-only spiking does not reach parity at any interesting timestep count.**

The attention *projections* (Q/K/V and output, 16.5% of distilgpt2's MACs, 22.9% of the
1.7B's) are what carry it over the line on the larger models. Those are ordinary linear
layers; spiking them is not obviously harder than spiking the MLP. The roadmap's split —
spike the MLP, preserve attention — happens to exclude the exact block that closes the gap.

---

## 6 · What to build, in order

1. **Extend spiking coverage past Q/K/V to the MLP and the attention projections.** This is
   the whole result. Everything else is second order.
2. **Benchmark on SmolLM2-360M or 1.7B, not distilgpt2.** Small-model results are
   structurally pessimistic and will keep understating the design.
3. **Keep T as low as the task tolerates.** The energy scales linearly in T against a fixed
   coverage budget; every timestep must be paid for by coverage you may not have.
4. **Do not spend effort driving the spike rate down** until coverage is above ~90%. Below
   that it is a rounding error.
5. **Leave `lm_head` continuous.** At 1.7B it is 5.7% of MACs and the most quality-sensitive
   layer in the model. Spiking it buys `T_max` 13.4 → 54.4, which is far past any plausible
   operating point.

---

## 7 · Caveats

This inherits every limitation of `spike_metrics.py` and adds one of its own.

- **Operation counts, not hardware.** 45nm reference figures (Horowitz, ISSCC 2014).
  Excludes memory movement, which is often dominant in practice.
- **Assumes event-driven hardware** that genuinely skips silent neurons. CPU/GPU simulation
  does not.
- **Ignores quality.** The energy ceiling computed here is an upper bound that says nothing
  about whether the model still works at that coverage.
  [`coverage-quality.md`](coverage-quality.md) measures that, and the answer is that it does
  not: post-hoc conversion costs at least 41x perplexity at every coverage and timestep
  count tested. The energy advantage below is reachable; the model that reaches it is not
  usable without training.
- **`ρ` is assumed uniform across layers.** Real spike rates vary; the closed form uses a
  single measured mean.
