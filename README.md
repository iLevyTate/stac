# STAC: Spiking Transformer Augmenting Cognition for Conversational AI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18023657.svg)](https://doi.org/10.5281/zenodo.18023657)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

STAC (Spiking Transformer Augmenting Cognition) is a research framework that explores two complementary approaches to spiking neural network (SNN) language modeling:

- **STAC V1**: A complete end-to-end training pipeline built around learnable Adaptive Exponential (AdEx) neurons. See `stac_v1/`.
- **STAC V2**: An experimental conversion framework that transforms pretrained transformer language models (DistilGPT-2, SmolLM2-1.7B-Instruct) into SNNs, targeting potential energy savings while retaining multi-turn conversational ability in simulation.

> **Important**: This repository currently runs *software-level* SNN simulations only. No metrics have been collected on physical neuromorphic hardware. Energy figures reported here are theoretical projections derived from spike-count analysis, not measured hardware data.

## Key Features

- Proof-of-concept ANN-to-SNN conversion built on SpikingJelly.
- Multi-turn context retention via a Temporal Spike Processor.
- Test coverage for position IDs, KV-cache behavior, and spike-rate sanity checks.
- Hardware power profiling: planned, not yet implemented.
- Full operator coverage and optimization: work in progress.

> **Scope note on V2 conversion.** The V2 path currently produces a *structurally*
> spiking model, not a spiking computation: `SpikeSoftmax` calls `torch.softmax`, and the
> LIF neurons inside `SpikeAttention` are constructed but deliberately bypassed to
> preserve generation quality (see the comments in `smollm2_converter.py`). Because the
> network is stateless, running it for `T` timesteps reproduces the same logits at `T`
> times the cost. Treat V2 as a conversion *scaffold*; STAC V1 (`stac_v1/`) is the part
> that actually spikes.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

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

**Pending or in progress**
- Spike-count telemetry hooks for V2 (STAC V1 reports spike statistics; V2 does not).
- Real spiking dynamics in V2 (`SpikeSoftmax` / `SpikeAttention` currently bypass their
  spiking neurons).
- Hardware benchmarking on Loihi-2 and Akida.
- Expanded operator support (rotary embeddings, flash-attention variants, etc.).
- Integration with the SCANUE multi-agent alignment layer.
- CLI, UX, and documentation polish.

### STAC V1

**Completed (research prototype)**
- End-to-end training pipeline with learnable AdEx neurons.
- Hyperdimensional Memory Module (HEMM) integration.
- Surrogate-gradient training on WikiText-2.
- L1 spike regularization for energy efficiency.
- Validation suite covering the full pipeline.

## Documentation

### STAC V2
- [Conversion Workflow](docs/conversion_workflow.md): step-by-step conversion guide.
- [API Reference](docs/api_reference.md): function and class documentation.
- [Hardware Requirements](docs/hardware_requirements.md): system specifications.

### STAC V1
- [STAC V1 Documentation](stac_v1/README.md): end-to-end training pipeline documentation.
- Run it with `python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 3`.

## Testing and Validation

The repository includes extensive testing for multi-turn conversational correctness:

```bash
# Test specific components
python tests/test_conversational_snn.py --model_name distilgpt2 --test_position_boundaries
python tests/test_conversational_snn.py --model_name distilgpt2 --test_attention_mask
python tests/test_conversational_snn.py --model_name distilgpt2 --test_multi_turn
python tests/test_conversational_snn.py --model_name distilgpt2 --test_energy

# Run the full suite
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
