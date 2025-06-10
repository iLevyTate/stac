# STAC: Conversion Workflow

This document describes the complete workflow for converting pretrained transformer LLMs to Spiking Neural Networks (SNNs) with multi-turn conversational capabilities.

## Overview

STAC converts transformer models (DistilGPT-2, SmolLM2-1.7B-Instruct) to energy-efficient spiking neural networks while preserving coherent dialog across conversation turns. All implementation phases are **complete** and validated.

## Conversion Pipeline Architecture

```
Input Model (HuggingFace) → SpikingJelly Conversion → TemporalSpikeProcessor → Multi-Turn SNN
```

## Implementation Status: ✅ COMPLETE

### ✅ Phase 1: Core Infrastructure

**Status: COMPLETE**

1. **SpikingJelly Integration**
   - ✅ Cross-version compatibility layer (`spikingjelly_compat.py`)
   - ✅ Unified Quantizer/Converter imports
   - ✅ Stable conversion pipeline with fallbacks

2. **Base Conversion**
   - ✅ GELU→ReLU activation replacement
   - ✅ `simplified_conversion()` for fast testing
   - ✅ Full SpikingJelly integration with calibration

### ✅ Phase 2: Temporal Dynamics

**Status: COMPLETE**

1. **Neuron State Management**
   - ✅ Stateful LIF neurons with `functional.reset_net()`
   - ✅ Membrane potential reset between tokens
   - ✅ `TemporalSpikeProcessor` wrapper

2. **Timestep Calibration**
   - ✅ Configurable timesteps (T=8-64)
   - ✅ Threshold scaling with `calibrate_timesteps()`
   - ✅ Logit magnitude restoration

### ✅ Phase 3: Conversation Context

**Status: COMPLETE**

1. **Position ID Management**
   - ✅ HuggingFace-compatible position ID generation
   - ✅ Clamping to `max_position_embeddings`
   - ✅ Continuous tracking across conversation turns

2. **KV Cache Implementation**
   - ✅ Global and per-conversation cache support
   - ✅ Automatic cache growth and truncation
   - ✅ Batch-aware cache management

3. **Attention Mechanism**
   - ✅ Dynamic attention mask growth
   - ✅ Causal masking for autoregressive generation
   - ✅ Context length management

### ✅ Phase 4: Testing and Optimization

**Status: COMPLETE**

1. **Multi-turn Testing**
   - ✅ Comprehensive test suite (`test_conversational_snn.py`)
   - ✅ Factual recall validation with keyword matching
   - ✅ Position ID boundary testing
   - ✅ Attention mask continuity validation

2. **Energy Benchmarking**
   - ✅ Spike counting and energy estimation
   - ✅ Wall-clock timing measurements
   - ✅ Mixed-precision compatibility testing

## Quick Start Commands

### 1. Fast Conversion (Recommended for Testing)

```bash
# Convert DistilGPT-2 with simplified pipeline
python run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified

# Test the converted model
python snn_multi_turn_conversation_test.py --mode snn --turns 3 --timesteps 8
```

### 2. Full SpikingJelly Conversion

```bash
# Convert with full calibration (requires more memory)
python run_conversion.py --model_name distilgpt2 --timesteps 16 --num_samples 10

# Convert SmolLM2 (requires ~20GB VRAM)
python smollm2_converter.py --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct --timesteps 32
```

### 3. Comprehensive Testing

```bash
# Run all validation tests
python test_conversational_snn.py --test_all --timesteps 8

# Test specific components
python test_conversational_snn.py --test_position_boundaries
python test_conversational_snn.py --test_attention_mask
python test_conversational_snn.py --test_multi_turn
python test_conversational_snn.py --test_energy
```

## Key Components

| File | Purpose |
|------|---------|
| `run_conversion.py` | Main CLI entry point for conversions |
| `smollm2_converter.py` | Specialized converter with `TemporalSpikeProcessor` |
| `convert.py` | Generic conversion utilities |
| `spikingjelly_compat.py` | Cross-version compatibility layer |

## Validation Checklist

Before deploying a converted model, ensure all tests pass:

- ✅ Position IDs stay within bounds
- ✅ Attention masks grow correctly across turns  
- ✅ KV cache maintains conversation history
- ✅ Multi-turn coherence with factual recall
- ✅ Energy consumption within expected range
- ✅ TorchScript export compatibility

## Troubleshooting

**Memory Issues**: Use `--simplified` flag or reduce `--timesteps`  
**Conversion Failures**: Check SpikingJelly version compatibility  
**Generation Quality**: Adjust temperature and top-k in generation scripts  

For detailed implementation status, see [Project State Overview](PROJECT_STATE_OVERVIEW.md). 