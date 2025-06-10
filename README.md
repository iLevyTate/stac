# STAC: Spiking Transformer for Conversational AI

[![DOI](https://zenodo.org/badge/907152074.svg)](https://doi.org/10.5281/zenodo.14545340)

## Overview

STAC (Spiking Transformer Augmenting Cognition) converts pretrained transformer LLMs (e.g., DistilGPT-2, SmolLM2-1.7B-Instruct) into energy-efficient Spiking Neural Networks (SNNs) **while preserving coherent multi-turn conversational ability**.

## Key Features

✅ **End-to-end ANN→SNN conversion** with SpikingJelly integration  
✅ **Multi-turn conversation support** with KV-cache and position ID handling  
✅ **Comprehensive test suite** validating coherence, energy, and compatibility  
✅ **Production-ready pipeline** with TorchScript export capabilities  
✅ **Energy efficiency** targeting 3-4× reduction in power consumption  

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Convert DistilGPT-2 to SNN (fast)
python run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# 3. Test multi-turn conversation
python snn_multi_turn_conversation_test.py --mode snn --turns 3 --timesteps 8

# 4. Run comprehensive validation
python test_conversational_snn.py --test_all --timesteps 8
```

## Core Components

| Component | Purpose |
|-----------|---------|
| `smollm2_converter.py` | Specialized converter with `TemporalSpikeProcessor` |
| `convert.py` | Generic ANN→SNN conversion pipeline |
| `run_conversion.py` | Main CLI entry point for conversions |
| `spikingjelly_compat.py` | Cross-version compatibility layer |
| `test_conversational_snn.py` | Comprehensive test suite (1K+ lines) |
| `snn_multi_turn_conversation_test.py` | Simple conversation smoke test |

## Implementation Status

All **Phase 1-4** objectives are complete:

- ✅ **Core Infrastructure**: SpikingJelly integration, GELU→ReLU, quantization
- ✅ **Temporal Dynamics**: Stateful LIF neurons, timestep calibration  
- ✅ **Conversation Context**: Position IDs, KV-cache, attention masks
- ✅ **Production Readiness**: TorchScript export, energy benchmarking

## Documentation

- 🔄 [Conversion Workflow](docs/conversion_workflow.md) - Step-by-step conversion guide
- 📚 [API Reference](docs/api_reference.md) - Function and class documentation  
- 🖥️ [Hardware Requirements](docs/hardware_requirements.md) - System specifications

## Testing & Validation

The repository includes extensive testing for multi-turn conversational correctness:

```bash
# Test specific components
python test_conversational_snn.py --test_position_boundaries
python test_conversational_snn.py --test_attention_mask  
python test_conversational_snn.py --test_multi_turn
python test_conversational_snn.py --test_energy

# Run all tests
python test_conversational_snn.py --test_all
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
