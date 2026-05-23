# STAC: Spiking Transformer Augmenting Cognition for Conversational AI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15867066.svg)](https://doi.org/10.5281/zenodo.15867066)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img width="2028" height="711" alt="STAC Logo" src="https://github.com/user-attachments/assets/52638288-b9b3-4463-aed4-52f140e55888" />

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

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Convert DistilGPT-2 to an SNN
python run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# 3. Run a multi-turn conversation smoke test
python snn_multi_turn_conversation_test.py --mode snn --turns 3 --timesteps 8

# 4. Run the comprehensive validation suite
python test_conversational_snn.py --model_name distilgpt2 --test_all --timesteps 8
```

## Core Components

### STAC V2

| Component | Purpose |
|-----------|---------|
| `smollm2_converter.py` | Specialized converter with `TemporalSpikeProcessor`. |
| `convert.py` | Generic ANN-to-SNN conversion pipeline. |
| `run_conversion.py` | Main CLI entry point for conversions. |
| `spikingjelly_compat.py` | Cross-version compatibility layer for SpikingJelly. |
| `test_conversational_snn.py` | Comprehensive test suite. |
| `snn_multi_turn_conversation_test.py` | Lightweight multi-turn smoke test. |

### STAC V1

| Component | Purpose |
|-----------|---------|
| `stac-v1/stacv1.ipynb` | End-to-end training pipeline with learnable AdEx neurons. |
| `stac-v1/README.md` | V1 documentation and research notes. |
| `stac_v1/` + `run_stac_v1.py` | Repo-native runnable V1 pipeline demonstrating hybrid fine-tuning (frozen GPT-2 with a trained spiking and memory head). |

## Implementation Status

### STAC V2

**Completed (prototype level)**
- Core conversion flow: GELU-to-ReLU substitution, quantization, and `ann2snn`.
- Temporal dynamics and KV-cache handling in PyTorch.
- Spike-count telemetry hooks and accompanying unit tests.
- Loihi export gating (requires `EXPORT_LOIHI=1` and `lava.lib.dl.slayer`; otherwise the pipeline remains simulation-only and Loihi tests are skipped).

**Pending or in progress**
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
- [STAC V1 Documentation](stac-v1/README.md): end-to-end training pipeline documentation.
- [STAC V1 Implementation](stac-v1/stacv1.ipynb): Jupyter notebook with learnable AdEx neurons.

## Testing and Validation

The repository includes extensive testing for multi-turn conversational correctness:

```bash
# Test specific components
python test_conversational_snn.py --model_name distilgpt2 --test_position_boundaries
python test_conversational_snn.py --model_name distilgpt2 --test_attention_mask
python test_conversational_snn.py --model_name distilgpt2 --test_multi_turn
python test_conversational_snn.py --model_name distilgpt2 --test_energy

# Run the full suite
python test_conversational_snn.py --model_name distilgpt2 --test_all
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
