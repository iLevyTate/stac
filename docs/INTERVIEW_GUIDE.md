# STAC Project — Interview Reference Guide

A complete, honest, technically-grounded reference for talking about the STAC
(Spiking Transformer Augmenting Cognition) project in interviews. Everything
here maps to real code in this repository, with file/line pointers and runnable
examples so you can answer follow-ups with confidence.

---

## 1. 30-Second Elevator Pitch

> "STAC is a research framework that explores running transformer language
> models as **Spiking Neural Networks (SNNs)** — the event-driven, brain-inspired
> compute model used by neuromorphic chips like Intel Loihi. It has two halves:
> **V1** trains a spiking model end-to-end with biologically-plausible neurons,
> and **V2** converts an existing pretrained transformer (DistilGPT-2,
> SmolLM2-1.7B) into a spiking equivalent while keeping multi-turn conversational
> ability. The goal is *energy efficiency*: spiking models only do work when a
> neuron fires, so in principle they can run language models far more cheaply on
> the right hardware."

**The single most important honesty point** (say this proactively — it shows
research maturity): *All results are software simulation. No metrics were
collected on physical neuromorphic hardware. Energy numbers are theoretical
projections from spike-count analysis, not measured watt-hours.* The repo states
this in the README and in code comments (`convert.py:71-77`).

---

## 2. Background You Should Be Able to Explain

### What is a Spiking Neural Network?
- Traditional ANNs pass continuous real-valued activations every layer, every
  forward pass. Every neuron does work every time.
- SNNs communicate with discrete **spikes** (0/1 events) over a series of
  **timesteps (T)**. A neuron integrates incoming current into a membrane
  potential and only "fires" a spike when it crosses a threshold.
- Because computation is **event-driven and sparse** (most neurons are silent
  most of the time), neuromorphic hardware (Intel Loihi, BrainChip Akida) can
  skip the silent neurons entirely → potential orders-of-magnitude energy
  savings vs. a GPU doing dense matrix multiplies.

### Why is this hard for transformers?
- Transformers rely on operations that are *not* naturally spike-friendly:
  **GELU activations**, **LayerNorm**, and especially **dense QK^T attention +
  softmax**. Dense attention is the big blocker — it isn't a native neuromorphic
  primitive. The repo encodes this knowledge directly in its Loihi validator
  (`loihi_constraints.py:166-192`), where any dense or `SpikeAttention` module is
  flagged as a `HARD_BLOCK` for hardware export.

### Two ways to get a spiking transformer
1. **Train from scratch** with surrogate gradients (the spike function is
   non-differentiable, so you approximate its gradient). → **STAC V1**.
2. **Convert a pretrained ANN** by swapping operators for spike-compatible
   versions and running it over T timesteps. → **STAC V2**.

---

## 3. Repository Map

```
stac/
├── convert.py                 # V2: generic ANN→SNN pipeline (SpikeZIP-TF style)
├── smollm2_converter.py       # V2: the core engine (1,675 lines) — all the
│                              #     spiking layers + TemporalSpikeProcessor
├── spikingjelly_compat.py     # V2: cross-version shim for the SpikingJelly lib
├── loihi_constraints.py       # Hardware export-readiness validator (both versions)
├── stac_v1/
│   ├── model.py               # V1: AdEx neurons, DLPFC layer, HEMM memory, model
│   └── pipeline.py            # V1: hybrid fine-tuning training loop
├── scripts/
│   ├── run_conversion.py      # V2 CLI entry point
│   ├── run_stac_v1.py         # V1 CLI entry point
│   └── train_snn_adapter.py   # V2: LoRA knowledge-distillation trainer
├── tests/
│   ├── test_conversational_snn.py  # V2 validation suite (10 test functions)
│   └── test_v1.py             # V1 unit + smoke tests
└── docs/                      # workflow, API, hardware, this guide
```

**Tech stack:** Python, PyTorch (≥2.6), HuggingFace Transformers, SpikingJelly
(the SNN library), PEFT/LoRA, BitsAndBytes (8-/4-bit quant), pytest,
torch.profiler.

---

## 4. STAC V2 — Conversion Framework (Deep Dive)

### 4.1 The conversion recipe
The core conversion is a sequence of operator substitutions. From
`smollm2_converter.py:1388` (`simplified_conversion`):

```python
def simplified_conversion(model, timesteps=32, skip_gelu_replacement=False):
    # 1. GELU -> ReLU  (ReLU's non-negative, unbounded output maps to spike rates)
    if not skip_gelu_replacement:
        model = replace_gelu_with_relu(model)
    # 2. record timestep count
    model.T = timesteps
    # 3. LayerNorm -> SpikeLayerNorm
    model = replace_layernorm_with_spikelayernorm(model)
    # 4. Attention -> SpikeAttention (or LoihiCausalContextMixer in loihi mode)
    model = replace_attention_with_spikeattention(model)
    # 5. wrap everything in the temporal driver
    model = TemporalSpikeProcessor(model, T=timesteps)
    return model
```

**Why GELU→ReLU?** ReLU's output is non-negative and proportional to input
magnitude, which lines up with a neuron's firing *rate* (rate coding). GELU is
smooth and goes negative, which doesn't. The honest trade-off, documented in the
code: GELU→ReLU **degrades generation quality** but is **required for true
spike-based execution** — so there's a `skip_gelu_replacement` flag to choose
between "good text" and "hardware-faithful."

### 4.2 The spiking layers (`smollm2_converter.py:83-211`)

**SpikeLayerNorm** — a LayerNorm reimplemented so it can sit in the spiking graph:
```python
class SpikeLayerNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
```

**SpikeAttention** — projects Q/K/V, has LIF (Leaky Integrate-and-Fire) spiking
neurons available on each, handles causal masking + KV-cache. Important detail
you should be ready to discuss: the spiking neurons on Q/K/V are *deliberately
disabled* in the current code (`smollm2_converter.py:163-167`):
```python
# For now, skip spiking neurons in attention to preserve text generation quality
q_spikes = q  # self.q_spk(q)
k_spikes = k  # self.k_spk(k)
v_spikes = v  # self.v_spk(v)
```
This is a candid engineering compromise: fully spiking attention destroyed
generation quality (near-uniform attention → repeated tokens), so the prototype
keeps attention numeric while spiking the rest. SpikeSoftmax similarly disables
its temperature scaling (`:96-114`) for the same reason. **Talking point:** this
is exactly the kind of fidelity-vs-faithfulness tension that defines ANN→SNN
conversion research.

### 4.3 The crown jewel: `TemporalSpikeProcessor` (`smollm2_converter.py:561`)

This is the component to know cold — it's what makes the SNN *conversational*.
It wraps the converted model and does three jobs:

**(a) Temporal accumulation.** Runs the inner model for `T` timesteps and sums
the logits, then normalizes — this is rate coding at the output:
```python
functional.reset_net(self.snn_model)        # clear neuron membrane states
spike_accum = None
for _ in range(effective_T):
    outputs = self.snn_model(model_input_ids, **model_kwargs)
    current_logits = outputs.logits
    spike_accum = current_logits if spike_accum is None else spike_accum + current_logits
final_logits = spike_accum / effective_T     # average over timesteps
final_logits = final_logits * self.logit_scale  # learnable parity scalar
```

**(b) Multi-turn memory via KV-cache management.** It maintains both a global KV
cache *and* per-conversation caches keyed by `batch_id`, so multiple independent
conversations can be batched without leaking context between them. It also has an
`incremental` mode that keeps an internal token cache and only feeds *new* tokens
each turn (`:622-646`), enforcing `max_context_length`.

**(c) Robust output shape handling.** It returns a `_CompatOutput` object
(`:856`) that supports both `output.logits` and tensor-style `output[0, -1, :]`
indexing so it's a drop-in for HuggingFace outputs in downstream code/tests.

**Why this matters in an interview:** the naive "run an SNN" is single-shot. The
hard part of *conversational* SNN is preserving state across turns while
resetting neuron membrane potentials between timesteps but NOT between turns —
this class is the answer, and the test suite specifically validates position-ID
continuity and KV-cache behavior across turns.

### 4.4 Knowledge distillation to recover quality (`scripts/train_snn_adapter.py`)
Conversion alone loses accuracy. To close the gap, there's a LoRA-based
distillation trainer: the original ANN is the **teacher**, the converted SNN is
the **student**, and a small LoRA adapter is trained so the student's logits
match the teacher's. Supports MSE, KL-divergence (with temperature), hard
cross-entropy to teacher argmax, and a combined KL+CE loss:
```python
T = float(args.temperature)
t_probs = F.softmax(t_logits_use.float() / T, dim=-1)
s_logp  = F.log_softmax(s_logits_use.float() / T, dim=-1)
loss = F.kl_div(s_logp, t_probs, reduction="batchmean") * (T * T)
```
Only the LoRA adapter + a learnable `logit_scale` (+ optionally SpikeLayerNorm
affine params) are trained — the backbone stays frozen. This is parameter-
efficient and keeps the transformer's learned semantics intact.

### 4.5 Hardware-readiness path (the Loihi story)
A separate code path (`loihi_mode`) prepares the model to *resemble* something a
neuromorphic chip could accept, without claiming it actually runs there:

- **`LoihiCausalContextMixer`** (`:214`) replaces dense attention with a *causal,
  recurrent leaky-integrator* context mechanism — no QK^T, no softmax:
  ```python
  alpha = torch.sigmoid(self._logit_alpha)   # learnable memory decay in (0,1)
  for t in range(seq_len):
      u_t = torch.tanh(self.ctx_in(x_t))
      c   = alpha * c + (1.0 - alpha) * u_t   # leaky recurrent state
      y_t = self.mix(torch.cat([x_t, self.ctx_out(c)], dim=-1))
  ```
  This is the deliberate answer to "dense attention is a hard Loihi blocker."
- **Fake int8 quantization** (`:269-308, :410`) — symmetric per-tensor int8 with
  a float scale, demonstrating weights are representable in fixed point.
- **Magnitude pruning** (`:492`) — zeroes smallest-magnitude weights to introduce
  sparsity (neuromorphic hardware benefits from sparse connectivity).
- **Hashed/bucketed embeddings** (`:326`) — shrinks the huge vocab embedding
  table via modulo bucketing.

### 4.6 `loihi_constraints.py` — the export-readiness validator
This module is a great thing to highlight because it embodies **intellectual
honesty as code**. It inspects a model and returns structured findings with
severities `HARD_BLOCK | WARNING | INFO`. `export_ready` is true only if there
are zero hard blocks. Crucially, it flags the project's *own* `SpikeAttention`
and any dense attention as hard blocks:
```python
if spike_attention_modules:
    findings.append(Finding(
        id="dense_attention_present_spikeattention",
        severity="HARD_BLOCK",
        message="SpikeAttention modules present. Dense QK^T attention/softmax is "
                "not a Loihi-native primitive; requires a dedicated mapping strategy."))
```
It also checks dtype histograms (float params → warning), quantization wrapper
presence, and weight sparsity, then writes a timestamped JSON report. **Talking
point:** "I built a validator that's honest enough to fail my own model" — this
is the opposite of overselling, and interviewers respect it.

---

## 5. STAC V1 — End-to-End Spiking Transformer (Deep Dive)

V1 is the "from scratch" approach. Files: `stac_v1/model.py`, `stac_v1/pipeline.py`.

### 5.1 Surrogate-gradient spikes (`model.py:12-32`)
The spike function `step(x > 0)` has zero gradient almost everywhere, so you
can't backprop through it. The trick is a **surrogate gradient**: forward is the
hard step, backward uses a smooth approximation (here a Gaussian bump):
```python
class SurrogateSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()                          # hard spike
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        pseudo = torch.exp(-(x**2) / 2.0) / math.sqrt(2*math.pi)  # smooth grad
        return grad_output * pseudo
```
Be ready to explain *why*: this is the central enabling trick of trainable SNNs.

### 5.2 Learnable AdEx neuron (`model.py:47-85`)
An **Adaptive Exponential Integrate-and-Fire** neuron — more biologically
plausible than plain LIF, with an adaptation current `w`. The differentiator
here: the dynamics parameters (`tau_m`, `tau_w`, `a`, `b`, `delta_T`) are
`nn.Parameter`s, so they're **learned by gradient descent**:
```python
exp_term = torch.exp((V - self.V_th) / self.delta_T).clamp(max=exp_clamp)
dV = (dt/self.tau_m) * (-(V - self.V_rest) + self.delta_T*exp_term - w + input_current)
V_new = V + dV
dw = (dt/self.tau_w) * (self.a * (V - self.V_rest) - w)   # adaptation
spike = surrogate_spike(V_new - self.V_th)
V_final = torch.where(spike > 0.5, self.V_reset, V_new)   # reset on fire
w_final = (w + dw) + self.b * spike                       # spike-triggered adapt
```

### 5.3 The full V1 model (`model.py:178-223`)
`DLPFCTransformer` = frozen **GPT-2 backbone** → **DLPFC spiking layer** (processes
the hidden-state sequence step by step through AdEx neurons, with optional
recurrent spiking sublayers) → **Hyperdimensional Memory Module (HEMM)** → LM head.

**HEMM** (`model.py:154-175`) is a neat idea: pool the spike train over time,
project it into a high-dimensional space with a *fixed random* matrix, then a
small MLP produces a "memory bias" added back to the representation:
```python
pooled_spikes = torch.mean(spike_train, dim=1)     # temporal pooling
hdm_vector = pooled_spikes @ self.proj_matrix      # random hyperdim projection
memory_bias = self.mlp(hdm_vector)                 # learned read-out
```

### 5.4 Hybrid fine-tuning + training loop (`pipeline.py`)
- **Hybrid fine-tuning** (`:56`): freeze the pretrained GPT-2, train only the
  spiking head + memory + LM head. Cheap, and keeps the language prior.
- **Loss = cross-entropy + L1 spike penalty** (`:285`):
  ```python
  loss_l1 = cfg.l1_lambda * torch.mean(torch.abs(spk_trains))
  loss = loss_xent + loss_l1
  ```
  The L1 term **penalizes spiking** → fewer spikes → lower (projected) energy.
  This is how "energy efficiency" enters training as an explicit objective.
- Standard hygiene: AdamW, linear warmup schedule, grad clipping, seeding,
  atomic JSON metric writes, spike-rate statistics, and it emits a Loihi
  constraints report each run.

---

## 6. Testing & Validation Methodology

This is a strong section to bring up — the project takes validation seriously.

**V2 suite (`tests/test_conversational_snn.py`, 10 functions):**
- `test_fidelity_parity` / `test_multi_turn_parity` — does the SNN match the ANN?
- `test_tsp_state_retention` — does the TemporalSpikeProcessor actually remember
  context across turns?
- `test_position_id_boundaries` — position IDs continue correctly with KV-cache.
- `test_attention_mask_continuity` — masks stay valid across turns/shapes.
- `test_multi_turn_coherence` — conversational sanity.
- `test_energy_consumption` — uses `torch.profiler` to measure CPU/CUDA time and
  memory for ANN vs SNN across sequence lengths 32/64/128, with a configurable
  efficiency target (`:842`). **Note honestly:** this measures *simulation*
  wall-time/memory on CPU/GPU, which is a proxy, NOT neuromorphic energy.
- `test_mixed_precision`, `test_loihi_compatibility`, `test_loihi_constraints`.

**V1 suite (`tests/test_v1.py`):** unit tests for the surrogate gradient, AdEx
neuron, DLPFC layer, HEMM, the full model, the data pipeline, the freeze logic,
and an end-to-end training smoke test.

---

## 7. Key Engineering Challenges (and How They Were Solved)

| Challenge | Solution in the code |
|-----------|----------------------|
| Spike function isn't differentiable | Surrogate gradient (Gaussian bump) `model.py:26` |
| Conversion tanks generation quality | LoRA distillation from ANN teacher `train_snn_adapter.py` |
| Fully-spiking attention → degenerate text | Keep attention numeric in prototype, spike everything else `:163` |
| Multi-turn state across timesteps | `TemporalSpikeProcessor` with global + per-conversation KV caches `:561` |
| SpikingJelly API churns across versions | `spikingjelly_compat.py` shim with fallbacks |
| Full conversion sometimes fails on complex models | Graceful fallback to `simplified_conversion` `convert.py:227` |
| Dense attention blocks neuromorphic export | `LoihiCausalContextMixer` recurrent replacement `:214` |
| Avoid overclaiming hardware results | `loihi_constraints.py` validator + explicit README/code disclaimers |
| Logit magnitude mismatch after conversion | learnable `logit_scale` parameter `:580` |

---

## 8. Likely Interview Questions — Prepared Answers

**Q: What problem does this solve?**
A: Running transformer LLMs is energy-expensive. SNNs on neuromorphic hardware
promise large energy savings because compute is event-driven and sparse. STAC
explores whether we can get a *conversational* transformer into spiking form,
both by training one (V1) and converting a pretrained one (V2).

**Q: Did you measure real energy savings?**
A: No — and I'm careful about this. Everything is software simulation. The energy
figures are theoretical projections from spike counts, and I built a Loihi
constraints validator that explicitly flags why the current model couldn't run
on hardware yet (dense attention is the blocker). The profiler test measures
simulation time/memory, which is only a proxy.

**Q: What's a surrogate gradient and why do you need it?**
A: A spike is a step function — gradient is zero or undefined, so backprop is
impossible. In the backward pass we substitute a smooth function's derivative
(I used a Gaussian bump) so gradients can flow while the forward pass stays a
hard 0/1 spike. It's the key trick that makes SNNs trainable.

**Q: V1 vs V2 — why both?**
A: V1 trains a spiking model end-to-end with learnable AdEx neurons and a
hyperdimensional memory module — full control, but expensive and from scratch.
V2 converts a pretrained transformer, which is cheaper and inherits a strong
language prior, but loses fidelity that I then recover with LoRA distillation.
Different points on the cost/quality/faithfulness trade-off.

**Q: Why is attention the hard part?**
A: Dense QK^T + softmax isn't a neuromorphic-native primitive and isn't naturally
sparse/event-driven. I tried spiking the Q/K/V projections but it caused
near-uniform attention and degenerate text, so the prototype keeps attention
numeric. For the hardware path I prototyped a recurrent leaky-integrator context
mixer that removes dense attention entirely.

**Q: How do you keep conversational context in an SNN?**
A: The `TemporalSpikeProcessor` resets neuron membrane states between the T
timesteps of a single forward pass, but persists a KV-cache (and an incremental
token cache) across conversational turns. It supports per-conversation caches
keyed by batch ID so batched dialogues don't bleed into each other.

**Q: What was the hardest bug / trade-off?**
A: The fidelity collapse from full spiking. Discovering that spiking the
attention path produced repeated-token gibberish forced a clear design decision:
separate a "faithful-to-hardware" mode from an "inference-quality" mode, and use
distillation to claw back accuracy. It taught me to make trade-offs explicit in
the code (flags, modes) rather than hiding them.

**Q: What would you do next?**
A: Real hardware benchmarking on Loihi-2/Akida, a proper spiking attention or a
fully-recurrent replacement that preserves quality, broader operator support
(rotary embeddings, flash-attention variants), and tightening ANN↔SNN parity
with better distillation.

---

## 9. What to Emphasize About Yourself

- **Systems thinking:** two complementary architectures, a shared hardware
  validator, CLIs, a real test suite, version-compat shims, graceful fallbacks.
- **Research honesty:** you separate measured facts from projections, and you
  wrote tooling that holds your own work to that standard.
- **ML depth:** surrogate gradients, AdEx neuron dynamics, rate coding, knowledge
  distillation (KL/temperature), LoRA/PEFT, quantization, pruning.
- **Engineering craft:** KV-cache correctness, position-ID handling, atomic
  writes, reproducibility (seeding), profiler-based measurement, defensive I/O.

---

## 10. Quick Command Cheat-Sheet

```bash
# Convert DistilGPT-2 to an SNN (fast path)
python scripts/run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# Run the V2 validation suite
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all --timesteps 8

# Distill the ANN teacher into the SNN student (recover quality)
python scripts/train_snn_adapter.py --model_name distilgpt2 --timesteps 8 --loss_type kl_ce

# Run the V1 hybrid fine-tuning pipeline (frozen GPT-2 + spiking/memory head)
python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 3

# V1 unit + smoke tests
python tests/test_v1.py
```

---

*Last note for the interview: lead with the honest framing (simulation-only,
projected energy), then dive into the engineering. The combination of ambitious
research scope and disciplined honesty is the story that lands.*
