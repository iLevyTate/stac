# STAC: Spiking Transformer Augmenting Cognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14545340.svg)](https://doi.org/10.5281/zenodo.14545340)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

STAC converts pretrained transformer LLMs (e.g., DistilGPT-2, SmolLM2-1.7B-Instruct) into energy-efficient Spiking Neural Networks (SNNs) while preserving coherent multi-turn conversational ability.

The repository contains two approaches:

- **V1** (`stac_v1/`): End-to-end training pipeline built around learnable Adaptive Exponential (AdEx) neurons, with a Hyperdimensional Memory Module (HEMM) and surrogate-gradient training.
- **V2**: ANN-to-SNN conversion framework that takes an existing transformer and replaces dense operations with spiking equivalents via SpikingJelly. The conversion is numerically faithful with spiking off; with spiking on, a frozen converted model collapses and needs training to recover.

> **Simulation only.** All SNN execution is software-level. No metrics have been collected on physical neuromorphic hardware. Energy figures are operation-count projections from [`spike_metrics.py`](spike_metrics.py), which counts spikes and synaptic operations during a real forward pass and applies published 45nm per-operation costs.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# (Optional) Build tiny local models for offline runs
python scripts/make_test_models.py --out local/test-models

# Convert DistilGPT-2 to an SNN
python scripts/run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# Multi-turn conversation smoke test
python tests/snn_multi_turn_conversation_test.py --mode snn --turns 3 --timesteps 8

# Full validation suite
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all --timesteps 8
```

## How V2 Conversion Works

By default, V2 produces a structurally spiking model that reproduces the source model to float precision. `SpikeSoftmax` calls `torch.softmax` and the LIF neurons in `SpikeAttention` are bypassed, so the network is stateless: running it for T timesteps reproduces the same logits at T times the cost.

Pass `--real_spiking` for genuine spiking computation. Q/K/V route through LIF neurons and softmax is dropped (Spikformer-style spiking self-attention). This changes outputs, so measure before relying on it.

### V2 Components

| Component | Purpose |
|-----------|--------|
| `smollm2_converter.py` | Specialized converter with `TemporalSpikeProcessor` |
| `convert.py` | Generic ANN-to-SNN conversion pipeline |
| `scripts/run_conversion.py` | CLI entry point for conversions |
| `spikingjelly_compat.py` | Cross-version SpikingJelly compatibility layer |
| `tests/test_conversational_snn.py` | Full test suite |
| `tests/snn_multi_turn_conversation_test.py` | Lightweight multi-turn smoke test |

### V1 Components

| Component | Purpose |
|-----------|--------|
| `stac_v1/` | V1 implementation (AdEx neurons, DLPFC layer, HEMM) |
| `scripts/run_stac_v1.py` | CLI for hybrid fine-tuning (frozen GPT-2 + trained spiking/memory head) |
| `stac_v1/README.md` | V1 documentation and research notes |

## Measured Results

Every number below is produced by the repository and can be regenerated offline. They come from generated tiny fixtures (`scripts/make_test_models.py`) with **random weights**: they pin behavior, not language quality.

### Conversion Fidelity (vs. unconverted model, T=8)

| Mode | Max abs logit diff | Top-1 agreement | Spiking neurons invoked |
|---|---|---|---|
| Default (non-spiking) | 1.19e-07 | 100.0% | 0 / 6 |
| `--real_spiking` | 8.44e-02 | 97.9% | 6 / 6 |

With spiking enabled, T=1 and T=8 differ by 9.0e-02, compared to 6.0e-08 (float noise) without it. That gap confirms the timestep loop stopped being a no-op.

### Energy Projection (tiny GPT-2, T=8, seq 32)

| Quantity | Value |
|---|---|
| Spike rate / sparsity | 0.094 / 90.6% |
| Spike-driven MACs | 65,536 of 1,180,672 (5.5%) |
| Projected SNN energy | 41.0 μJ |
| Projected ANN energy | 5.4 μJ |
| **Ratio** | **7.6x worse than the dense ANN** |

Only QKᵀ is spike-driven. The remaining dense work is paid on every timestep. An energy advantage requires spiking activations throughout the network, not only on Q/K/V.

[`docs/energy-crossover.md`](docs/energy-crossover.md) works out the break-even: affordable timestep count is `T_max = 1 / (1 - f(1 - ρr))`, driven almost entirely by coverage f. The current 5.5% coverage affords T ≤ 1.06, so no spiking operating point wins. Coverage above 90% does win. Spiking the body of SmolLM2-1.7B reaches 94.3% coverage and projects 1.78x *better* at T=8. Much of the 7.6x above is an artifact of benchmarking a tiny model.

```bash
python scripts/energy_analysis.py --scaling          # coverage vs. model size
python scripts/energy_analysis.py --arch smollm2-1.7b --seq_len 2048
```

### V1 Baseline

[`docs/baselines/stac_v1_smoke.json`](docs/baselines/stac_v1_smoke.json) records a reproducible 5-step run (spike rate 0.147, 86.8% sparsity, loss 5.83). Before the audit fixed the AdEx neurons, spike rate and L1 penalty were both exactly zero and the model produced identical logits at every position. `tests/test_v1_baseline.py` guards against regression.

## Status

### V2: Done (prototype)

- Core conversion: GELU-to-ReLU substitution, quantization, `ann2snn` call path. SpikingJelly's `ann2snn.Converter` requires a `torch.fx`-traceable model; HuggingFace causal LMs are generally not, so conversion falls back to the simplified path (logged in saved metadata).
- Temporal dynamics and KV-cache handling.
- Loihi export gating (requires `EXPORT_LOIHI=1` and `lava.lib.dl.slayer`; otherwise simulation-only).
- Spike-count telemetry and operation-level energy projection.
- Genuine spiking attention behind `--real_spiking`, with grouped-query attention support (SmolLM2-135M/360M).

### V2: Pending

- Making spiking the default (opt-in until quality cost is measured on a trained model rather than random-weight fixtures).
- Spiking activations throughout the network. LIF neurons sit only on Q/K/V (5.5% of MACs), which is why the energy projection is unfavorable. An advantage needs a spiking MLP and residual stream.
- Spiking dynamics in `--loihi_mode` (currently swaps attention for `LoihiCausalContextMixer`, which is non-spiking).
- Hardware benchmarking on Loihi-2 and Akida.
- Expanded operator support (rotary embeddings, flash-attention variants).
- Integration with the SCANUE multi-agent alignment layer.

### V1: Done (research prototype)

- End-to-end pipeline with learnable AdEx neurons.
- HEMM integration with causal pooling.
- Surrogate-gradient training (`--dataset wikitext2` loads WikiText-2 when the `datasets` package is installed; otherwise built-in sample texts).
- L1 spike regularization (was identically zero until neurons were made excitable).
- Validation suite with committed metrics baseline.

## Testing

Tests pin multi-turn behavior (cache state, position handling, spike rates). They guard against regressions; they do not by themselves demonstrate conversational quality under genuine spiking.

```bash
# Individual components
python tests/test_conversational_snn.py --model_name distilgpt2 --test_position_boundaries
python tests/test_conversational_snn.py --model_name distilgpt2 --test_attention_mask
python tests/test_conversational_snn.py --model_name distilgpt2 --test_multi_turn
python tests/test_conversational_snn.py --model_name distilgpt2 --test_energy

# Full suite
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all

# Quality cost of genuine spiking
python tests/test_conversational_snn.py --model_name distilgpt2 --real_spiking --test_fidelity
```

The pytest suite runs offline. If the Hugging Face hub is unreachable and `STAC_TEST_MODEL` is unset, `tests/conftest.py` generates a tiny local model automatically.

```bash
python -m pytest tests/ -q
```

## Docs

- [Conversion Workflow](docs/conversion_workflow.md)
- [API Reference](docs/api_reference.md)
- [Hardware Requirements](docs/hardware_requirements.md)
- [STAC V1 Documentation](stac_v1/README.md) (or run: `python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 3`)

## License

MIT. See [LICENSE](LICENSE).
