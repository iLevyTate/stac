# STAC: Spiking Transformer Augmenting Cognition for Conversational AI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18023657.svg)](https://doi.org/10.5281/zenodo.18023657)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

STAC (Spiking Transformer Augmenting Cognition) is a research framework that explores two complementary approaches to spiking neural network (SNN) language modeling:

- **STAC V1**: A complete end-to-end training pipeline built around learnable Adaptive Exponential (AdEx) neurons. See `stac_v1/`.
- **STAC V2**: An experimental conversion framework that transforms pretrained transformer language models (DistilGPT-2, SmolLM2-1.7B-Instruct) into SNNs. The conversion is numerically faithful with spiking *off*; with spiking *on*, a frozen converted model collapses and needs training to recover — and the projected energy of the current design is worse than the dense ANN. See [`docs/findings-summary.md`](docs/findings-summary.md).

> **Important**: This repository currently runs *software-level* SNN simulations only. No
> metrics have been collected on physical neuromorphic hardware. Energy figures are
> operation-count projections produced by [`spike_metrics.py`](spike_metrics.py) — it
> counts spikes and synaptic operations during a real forward pass and applies published
> 45nm per-operation costs. Run it yourself; it currently projects that the converted V2
> model would be **worse** than the dense ANN (see *Measured results* below).

## Key Features

- Proof-of-concept ANN-to-SNN conversion built on SpikingJelly.
- Temporal Spike Processor for multi-turn KV-cache/state management (the mechanism; conversational quality under genuine spiking requires training).
- Test coverage for position IDs, KV-cache behavior, and spike-rate sanity checks.
- Hardware power profiling: planned, not yet implemented.
- Full operator coverage and optimization: work in progress.

> **Scope note on V2 conversion.** By default the V2 path produces a *structurally*
> spiking model, not a spiking computation: `SpikeSoftmax` calls `torch.softmax` and the
> LIF neurons in `SpikeAttention` are bypassed, so the network is stateless and running it
> for `T` timesteps reproduces the same logits at `T` times the cost. This default exists
> because it reproduces the source model exactly (see *Measured results*).
>
> Pass `--real_spiking` for genuine spiking computation: Q/K/V are routed through the LIF
> neurons and softmax is dropped (Spikformer-style spiking self-attention). This is off by
> default because it changes the model's outputs — measure before relying on it.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 1b. (Optional) Build tiny local models so everything below runs without network access
python scripts/make_test_models.py --out local/test-models

# 2. Convert DistilGPT-2 to an SNN
python scripts/run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# 3. Run a multi-turn conversation smoke test
python tests/snn_multi_turn_conversation_test.py --mode snn --turns 3 --timesteps 8

# 4. Run the comprehensive validation suite
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all --timesteps 8
```

## Core Components

### STAC V2

| Component | Purpose |
|-----------|---------|
| `smollm2_converter.py` | Specialized converter with `TemporalSpikeProcessor`. |
| `convert.py` | Generic ANN-to-SNN conversion pipeline. |
| `scripts/run_conversion.py` | Main CLI entry point for conversions. |
| `spikingjelly_compat.py` | Cross-version compatibility layer for SpikingJelly. |
| `tests/test_conversational_snn.py` | Comprehensive test suite. |
| `tests/snn_multi_turn_conversation_test.py` | Lightweight multi-turn smoke test. |

### STAC V1

| Component | Purpose |
|-----------|---------|
| `stac_v1/` | Runnable, importable V1 implementation (AdEx neurons, DLPFC layer, HEMM). |
| `scripts/run_stac_v1.py` | CLI for the repo-native V1 hybrid fine-tuning pipeline (frozen GPT-2 with a trained spiking and memory head). |
| `stac_v1/README.md` | V1 documentation and research notes. |

## Implementation Status

### STAC V2

**Completed (prototype level)**
- Core conversion flow: GELU-to-ReLU substitution, quantization, and the `ann2snn` call
  path. Note that SpikingJelly's `ann2snn.Converter` requires a `torch.fx`-traceable
  model; HuggingFace causal LMs generally are not, so conversion falls back to the
  simplified path. The fallback is logged and recorded in the saved metadata.
- Temporal dynamics and KV-cache handling in PyTorch.
- Loihi export gating (requires `EXPORT_LOIHI=1` and `lava.lib.dl.slayer`; otherwise the pipeline remains simulation-only and Loihi tests are skipped).

- Spike-count telemetry and an operation-level energy projection (`spike_metrics.py`).
- Genuine spiking attention behind `--real_spiking`, with grouped-query attention support
  (SmolLM2-135M/360M).

**Pending or in progress**
- Making spiking the default. It is opt-in until the quality cost is measured on a trained
  model rather than the random-weight fixtures used offline.
- Spiking activations *throughout* the network. The LIF neurons currently sit only on
  Q/K/V, so just 5.5% of the model's MACs become accumulates — which is why the energy
  projection is unfavourable (see *Measured results*). An advantage needs a spiking MLP
  and residual stream.
- Spiking dynamics in `--loihi_mode`. That path swaps attention for
  `LoihiCausalContextMixer`, which removes the dense-attention blocker but is itself
  non-spiking (tanh + a leaky context accumulator), so the resulting model contains no
  spiking neurons at all and the constraints validator fails it accordingly.
- Hardware benchmarking on Loihi-2 and Akida.
- Expanded operator support (rotary embeddings, flash-attention variants, etc.).
- Integration with the SCANUE multi-agent alignment layer.
- CLI, UX, and documentation polish.

### STAC V1

**Completed (research prototype)**
- End-to-end training pipeline with learnable AdEx neurons.
- Hyperdimensional Memory Module (HEMM) integration, with causal pooling.
- Surrogate-gradient training. `--dataset wikitext2` loads WikiText-2 when the optional
  `datasets` package is installed; otherwise the built-in sample texts are used.
- L1 spike regularization. Note this term was identically zero until the neurons were made
  excitable — see the baseline below.
- Validation suite covering the full pipeline, including a committed metrics baseline.

## Measured results

Every number below is produced by the repository itself and can be regenerated offline.
They come from the generated tiny fixtures (`scripts/make_test_models.py`), which have
**random weights** — they pin behaviour, not language quality.

### Conversion fidelity (vs the unconverted model, T=8)

| Mode | max abs logit difference | top-1 agreement | spiking neurons invoked |
|---|---|---|---|
| default (non-spiking) | 1.19e-07 | 100.0% | 0 / 6 |
| `--real_spiking` | 8.44e-02 | 97.9% | 6 / 6 |

The default reproduces the source model to float precision. With spiking enabled, T=1 and
T=8 differ by 9.0e-02 — against 6.0e-08 (float noise) without it, which is the evidence
that the timestep loop stopped being a no-op.

### Energy projection (`spike_metrics.py`, tiny GPT-2, T=8, seq 32)

| Quantity | Value |
|---|---|
| spike rate / sparsity | 0.094 / 90.6% |
| spike-driven MACs | 65,536 of 1,180,672 (5.5%) |
| projected SNN energy | 41.0 uJ |
| projected ANN energy | 5.4 uJ |
| **ratio** | **7.6x worse than the dense ANN** |

This is the honest result, and it is informative: only QK^T is spike-driven, and the
remaining dense work is paid on every one of the T timesteps. An energy advantage requires
spiking activations throughout the network, not only on Q/K/V.

[`docs/energy-crossover.md`](docs/energy-crossover.md) works out what that would take. In
short: the affordable timestep count is `T_max = 1 / (1 - f(1 - rho*r))`, set almost
entirely by coverage `f` and barely at all by spike rate `rho`. The current 5.5% coverage
affords `T <= 1.06`, so no genuinely spiking operating point wins. Coverage above 90% does
win — and because `lm_head` is `d*V` while the body is `L*d^2`, that is far easier to reach
on a large model than on this one: spiking the body of SmolLM2-1.7B reaches 94.3% coverage
and projects 1.78x *better* at T=8. Much of the 7.6x above is an artifact of benchmarking a
tiny model.

```bash
python scripts/energy_analysis.py --scaling          # coverage vs. model size
python scripts/energy_analysis.py --arch smollm2-1.7b --seq_len 2048
```

```bash
python -c "
from transformers import AutoModelForCausalLM
from smollm2_converter import simplified_conversion
from spike_metrics import measure_spikes
import torch
m = AutoModelForCausalLM.from_pretrained('local/test-models/tiny-gpt2')
t = simplified_conversion(m, 8, skip_gelu_replacement=True, real_spiking=True)
print(measure_spikes(t, torch.randint(0, 200, (1, 32)), use_cache=False).summary())"
```

### STAC V1 baseline

[`docs/baselines/stac_v1_smoke.json`](docs/baselines/stac_v1_smoke.json) records a
reproducible 5-step run (spike rate 0.147, 86.8% sparsity, loss 5.83). Before the audit
fixed the AdEx neurons, spike rate and the L1 penalty were both exactly zero and the model
produced identical logits at every position. `tests/test_v1_baseline.py` guards against
returning to that state.

## Documentation

### STAC V2
- [Conversion Workflow](docs/conversion_workflow.md): step-by-step conversion guide.
- [API Reference](docs/api_reference.md): function and class documentation.
- [Hardware Requirements](docs/hardware_requirements.md): system specifications.

### STAC V1
- [STAC V1 Documentation](stac_v1/README.md): end-to-end training pipeline documentation.
- Run it with `python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 3`.

## Testing and Validation

The repository includes tests that pin multi-turn behavior (cache state, position handling, spike rates) — they guard against regressions, they do not by themselves demonstrate conversational quality under genuine spiking:

```bash
# Test specific components
python tests/test_conversational_snn.py --model_name distilgpt2 --test_position_boundaries
python tests/test_conversational_snn.py --model_name distilgpt2 --test_attention_mask
python tests/test_conversational_snn.py --model_name distilgpt2 --test_multi_turn
python tests/test_conversational_snn.py --model_name distilgpt2 --test_energy

# Run the full suite
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all

# Measure the quality cost of genuine spiking
python tests/test_conversational_snn.py --model_name distilgpt2 --real_spiking --test_fidelity
```

The pytest suite runs offline: if the Hugging Face hub is unreachable and
`STAC_TEST_MODEL` is unset, `tests/conftest.py` generates a tiny local model
automatically.

```bash
python -m pytest tests/ -q
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
