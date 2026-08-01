# STAC audit and conversion study — findings summary

A through-line for the whole investigation. Each section links to the document that carries
the detail and the reproduction. Written to inform the paper: it separates what is
established from what is open, and marks which published claims are affected.

---

## The one-paragraph story

STAC converts a pretrained transformer into a spiking network for energy efficiency. The
audit found that the spiking pathway had never actually functioned — dead in V1, bypassed
in V2. Fixing the bugs that hid this made the pipeline faithful (conversion now reproduces
the source model to 1.00× perplexity with spiking off) and let the real question be asked:
does spike conversion of a *frozen* transformer produce a usable model? It does not — it
collapses the model to near-constant output, and every post-hoc remedy (coverage,
calibration, leak, timesteps) fails. Letting the weights move does work: a short
distillation run recovers perplexity 10.5×, where nothing frozen moved it at all. The
energy premise is separately weaker than assumed: coverage, not sparsity, sets the budget,
and the current design projects worse than the dense baseline. The net position is a
well-supported account of why conversion fails, a demonstrated (if not yet finished) path
through training, and a set of shipped fixes.

---

## Finding 1 — the spiking never functioned

Detail: [`corrigendum-2026-07.md`](corrigendum-2026-07.md),
[`paper-corrigendum-submission.md`](paper-corrigendum-submission.md).

- **V1**: the AdEx neurons emitted zero spikes for every input (membrane pinned ~14.7 mV
  below threshold; required drive 75 vs observed 2.4). The surrogate gradient underflowed
  float32 to exactly zero, so no gradient ever reached the spiking layer, and the L1
  sparsity penalty was identically 0.0 — its best possible value, for the worst possible
  reason. Reproduced verbatim from the published notebook by
  `scripts/verify_v1_corrigendum.py`.
- **V2**: `SpikeAttention` constructed LIF neurons and then wrote `q_spikes = q`, bypassing
  them. "Spiking attention" was ordinary attention with unused modules attached.

**Affects the paper.** The V1 mechanism claims (gradients flow through the spiking neurons,
firing properties are fine-tuned, L1 regularises spiking) describe intended behaviour that
did not occur. No published *number* is wrong — the paper reports no V1 metrics — so this
is a corrigendum, not a retraction. The submission text is drafted and blocked only on the
venue. A citation error is also recorded: the version cited for V1 contains no V1 code.

## Finding 2 — the energy premise: coverage, not sparsity

Detail: [`energy-crossover.md`](energy-crossover.md).

Closed form, validated against `spike_metrics.py` to 0.1%:

```
E_SNN / E_ANN = T · (ρ·f·r + 1 − f)        T_max = 1 / (1 − f·(1 − ρ·r))
```

with `f` the spike-driven fraction of MACs, `ρ` the spike rate, `r = E_AC/E_MAC ≈ 0.196`.
Because `ρ` enters only through `ρ·r`, sparsity is a weak lever: driving the spike rate to
zero moves `T_max` less than a modest coverage gain does. **Coverage sets the timestep
budget.** The shipped design covers ~5% of MACs (Q/K^T only) and projects **7.6× worse**
than the dense ANN. An advantage needs ~90%+ coverage; on SmolLM2-1.7B that is reachable in
principle (the `lm_head`, the block hardest to spike, falls from 45% of MACs on distilgpt2
to 5.7% at 1.7B).

**Affects the paper.** The energy figures are hedged correctly (software latency, not
joules), so nothing is contradicted — but the "3–4×" framing should note that the
operation-count projection comes out worse than the ANN at the current coverage, and that
the advantage is contingent on coverage the pipeline does not yet reach.

## Finding 3 — post-hoc conversion collapses the model

Detail: [`coverage-quality.md`](coverage-quality.md).

Conversion of a frozen transformer does not degrade gracefully — it collapses to
near-constant output (6 distinct predictions over 256 positions on distilgpt2, a comma
86.7% of the time; next-token accuracy 0.0127, near the base rate of frequent
punctuation). Every post-hoc knob was tried:

| Knob | Result |
| --- | --- |
| coverage 5% → 98% | quality improves slightly; collapse persists |
| logit calibration | 0% of gap (GPT-2); cosmetic on SmolLM2 |
| membrane leak (τ) | monotonically worse |
| timesteps 8 → 64 | no help; early blocks slightly worse |

The damage is **located and precision-independent**: on clean input, blocks 2–5 convert at
0.97+ correlation, while the loss originates entirely in blocks 0–1 (block 1 the cliff at
0.52), and more timesteps do not recover it. Nothing that leaves the weights frozen fixes
it.

**Affects the paper.** Consistent with the paper's stated coherence loss and its decision
to disable spiking during inference — this quantifies the cost and *supports* that
decision. It also promotes quantization-aware / spike-aware training from the third
future-work track to the critical path.

## Finding 4 — fixes shipped (merged in #11)

- **RoPE**: `SpikeAttention` now applies rotary embeddings. Conversion-only perplexity on
  SmolLM2-135M/360M went from 19–28× worse to **1.00×**. This was the single largest
  defect on the paper's headline models.
- **Attention encoding**: replaced the fixed-threshold hard-reset LIF (a 1-bit quantiser,
  corr 0.805 with magnitude) with a calibrated signed soft-reset IF (0.989). Spiking
  attention is now 5.7× less damaging.
- **RMSNorm**: `SpikeRMSNorm` handles Llama-family normalisation, which the LayerNorm-only
  pass silently skipped.
- **Liveness tests** (`tests/test_liveness.py`, `test_rope_fidelity.py`,
  `test_energy_analysis.py`, `test_spike_coverage.py`): assert spikes exist, are not
  saturated, and predictions vary by position — the properties whose absence let every
  finding above survive.

## The methodological thread

Every finding shared one signature: **a convenient metric reading its best value on a dead
network.**

- V1's L1 spike penalty sat at exactly 0.0 — read as perfect sparsity, meant zero spikes.
- Cosine similarity to teacher logits *rose* 0.978 → 0.986 as the model collapsed, because
  a 50k-vocab logit vector is dominated by shared frequency structure.
- The conversation test returned `True` unconditionally; the main test file ran zero tests.

This is the most transferable contribution: SNN-conversion evaluation needs spike counts
and prediction diversity alongside loss and similarity, or it certifies dead networks.

---

## Established vs. open

**Established (this session, reproducible):**
- The spiking pathway never functioned as published (V1 and V2).
- The energy law and its coverage-dominates-sparsity consequence.
- Post-hoc conversion of a frozen transformer collapses the model; every post-hoc knob
  fails; the damage is localised to the first two blocks and precision-independent.
- The four fixes, each with a regression test.

**Demonstrated in proof of concept:**
- **Spike-aware fine-tuning recovers the collapse.** Training the converted network end to
  end with ANN distillation (`scripts/finetune_spiking.py`) dropped distilgpt2's eval
  perplexity 10.5× in 300 CPU steps (6,161 → 580), monotonically, where every frozen-weight
  remedy moved it not at all. Still ~11× above the ANN baseline at this tiny scale — a
  direction, not a finished result. See [`coverage-quality.md`](coverage-quality.md) §5d.

**Open (untested):**
- A conclusive fine-tuning run: T=8, longer sequences, thousands of steps, GPU — to see how
  close training gets to the ANN baseline, and at what coverage the energy advantage
  survives it.
- Scale beyond 360M; encodings other than signed soft-reset IF; coherence/task metrics
  beyond perplexity and prediction diversity.

---

## Recommended next steps

1. **File the corrigendum** — a live obligation, drafted, blocked only on the venue.
2. **Fold Findings 2–3 into the paper** as the honest current state of V2: the fixes make
   conversion faithful, and post-hoc conversion without training does not yet yield a
   usable spiking model. This strengthens the paper's existing caveats rather than
   undermining them.
3. **Then decide on fine-tuning** — the substantive research fork, and the only way to turn
   the negative result into a positive one.
