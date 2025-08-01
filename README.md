# STAC: Spiking Transformer Augmenting Cognition for Conversational AI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15867066.svg)](https://doi.org/10.5281/zenodo.15867066)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img width="2028" height="711" alt="STACLogoResized" src="https://github.com/user-attachments/assets/52638288-b9b3-4463-aed4-52f140e55888" />

## Overview

STAC (Spiking Transformer Augmenting Cognition) is a research framework with two distinct approaches:

- **STAC V1**: Complete end-to-end training pipeline with learnable AdEx neurons (see `stac-v1/`)
- **STAC V2**: Experimental conversion framework that transforms pretrained transformer LLMs (DistilGPT-2, SmolLM2-1.7B-Instruct) into Spiking Neural Networks (SNNs) for *potential* energy savings **while retaining multi-turn conversational ability in simulation**

> ⚠️  **Important**: This repository currently runs *software-level* SNN simulations only. No metrics have been collected on physical neuromorphic hardware yet. Energy savings figures are theoretical projections based on spike-count analysis, not measured hardware data.

## Key Features

✔️ **Proof-of-concept ANN→SNN conversion** using SpikingJelly  
✔️ **Multi-turn context retention** via a Temporal Spike Processor  
✔️ **Extensive software tests** for position IDs, KV-cache, and spike-rate sanity  
➖ **Hardware power profiling** — *planned, not implemented*  
➖ **Full operator coverage & optimisation** — *work in progress*  

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Convert DistilGPT-2 to SNN (fast)
python run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# 3. Test multi-turn conversation
python snn_multi_turn_conversation_test.py --mode snn --turns 3 --timesteps 8

# 4. Run comprehensive validation
python test_conversational_snn.py --model_name distilgpt2 --test_all --timesteps 8
```

## Core Components

### STAC V2 (Current)
| Component | Purpose |
|-----------|---------|
| `smollm2_converter.py` | Specialized converter with `TemporalSpikeProcessor` |
| `convert.py` | Generic ANN→SNN conversion pipeline |
| `run_conversion.py` | Main CLI entry point for conversions |
| `spikingjelly_compat.py` | Cross-version compatibility layer |
| `test_conversational_snn.py` | Comprehensive test suite (1K+ lines) |
| `snn_multi_turn_conversation_test.py` | Simple conversation smoke test |

### STAC V1 (Original Research)
| Component | Purpose |
|-----------|---------|
| `stac-v1/stacv1.ipynb` | Complete end-to-end training pipeline with learnable AdEx neurons |
| `stac-v1/README.md` | V1 documentation and research contributions |

## Implementation Status

### STAC V2 (Current)
**Completed (prototype level)**
- ✅ Core conversion flow (GELU→ReLU, quantization, ann2snn)
- ✅ Temporal dynamics & KV-cache handling in PyTorch
- ✅ Spike-count telemetry hooks and unit tests

**Pending / In Progress**
- ⏳ Hardware benchmarking on Loihi-2 / Akida
- ⏳ Expanded operator support (e.g., rotary embeddings, flash-attention variants)
- ⏳ Integration with SCANUE multi-agent alignment layer
- ⏳ Robust CLI/UX and documentation polish

### STAC V1 (Complete)
**Completed (research prototype)**
- ✅ End-to-end training pipeline with learnable AdEx neurons
- ✅ Hyperdimensional Memory Module (HEMM) integration
- ✅ Surrogate gradient training on WikiText-2
- ✅ L1 spike regularization for energy efficiency
- ✅ Comprehensive validation suite

## Documentation

### STAC V2 (Current)
- 🔄 [Conversion Workflow](docs/conversion_workflow.md) - Step-by-step conversion guide
- 📚 [API Reference](docs/api_reference.md) - Function and class documentation  
- 🖥️ [Hardware Requirements](docs/hardware_requirements.md) - System specifications

### STAC V1 (Original Research)
- 📖 [STAC V1 Documentation](stac-v1/README.md) - End-to-end training pipeline documentation
- 🧠 [STAC V1 Implementation](stac-v1/stacv1.ipynb) - Complete Jupyter notebook with learnable AdEx neurons

## Testing & Validation

The repository includes extensive testing for multi-turn conversational correctness:

```bash
# Test specific components
python test_conversational_snn.py --model_name distilgpt2 --test_position_boundaries
python test_conversational_snn.py --model_name distilgpt2 --test_attention_mask  
python test_conversational_snn.py --model_name distilgpt2 --test_multi_turn
python test_conversational_snn.py --model_name distilgpt2 --test_energy

# Run all tests
python test_conversational_snn.py --model_name distilgpt2 --test_all
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
