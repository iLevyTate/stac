# STAC API Reference

This document provides reference documentation for the key components of the STAC multi-turn conversational SNN pipeline.

## Core Components

### TemporalSpikeProcessor

The central component for processing SNN models with multi-turn conversational capabilities.

**Location**: `smollm2_converter.py`

```python
from smollm2_converter import TemporalSpikeProcessor

processor = TemporalSpikeProcessor(
    snn_model,               # The converted SNN model
    T=16,                    # Number of timesteps
    max_context_length=512   # Maximum context length
)
```

#### Key Methods

**`forward(input_ids, attention_mask=None, use_cache=True, batch_ids=None)`**
- Process inputs through the SNN with KV-cache support
- Returns: `_CompatOutput` object with `.logits` and `.past_key_values`

**`reset_cache(batch_id=None)`**
- Reset KV cache for specific batch or all batches
- Args: `batch_id` (optional) - specific conversation to reset

**`get_position_ids()`**
- Return current position IDs tensor for validation
- Returns: `torch.Tensor` of position IDs

**`_create_position_ids(input_shape, past_length=0)`**
- Internal method for HuggingFace-compatible position ID creation
- Handles clamping to `max_position_embeddings`

### Spike-Compatible Layers

#### SpikeLayerNorm

Spiking-compatible layer normalization replacement.

```python
from smollm2_converter import SpikeLayerNorm

layer_norm = SpikeLayerNorm(
    normalized_shape,    # Shape to normalize over
    eps=1e-5            # Epsilon for numerical stability
)
```

#### SpikeAttention

Spiking-compatible self-attention implementation.

```python
from smollm2_converter import SpikeAttention

attention = SpikeAttention(
    embed_dim=768,       # Embedding dimension
    num_heads=12,        # Number of attention heads
    T=16,               # Timesteps for spike processing
    causal=True         # Enable causal masking
)
```

#### SpikeSoftmax

Spiking-compatible softmax using spike rates.

```python
from smollm2_converter import SpikeSoftmax

softmax = SpikeSoftmax(
    T=16,               # Temporal windows
    dim=-1              # Dimension to apply softmax
)
```

## Conversion Functions

### simplified_conversion(model, timesteps=32)

Fast conversion method for testing and development.

**Location**: `smollm2_converter.py`

```python
from smollm2_converter import simplified_conversion

snn_model = simplified_conversion(model, timesteps=16)
```

**Features**:
- GELU→ReLU replacement
- Threshold scaling for SpikeZIP-TF equivalence
- Wrapped forward method for SNN behavior simulation

### Full SpikingJelly Conversion

Complete conversion using SpikingJelly's Converter with calibration.

**Location**: `convert.py`, `smollm2_converter.py`

```python
from convert import convert_model_to_spiking

snn_model = convert_model_to_spiking(
    model, 
    calibration_data, 
    timesteps=64,
    device='cuda'
)
```

## Compatibility Layer

### spikingjelly_compat.py

Cross-version compatibility for SpikingJelly components.

```python
from spikingjelly_compat import get_quantizer, get_converter, get_neuron

Quantizer = get_quantizer()      # Get version-appropriate Quantizer
Converter = get_converter()      # Get version-appropriate Converter  
LIFNode = get_neuron()          # Get LIF neuron implementation
```

## Testing Framework

### test_conversational_snn.py

Comprehensive test suite for multi-turn validation.

**Key Test Functions**:

```python
# Test position ID boundaries
python test_conversational_snn.py --test_position_boundaries

# Test attention mask continuity  
python test_conversational_snn.py --test_attention_mask

# Test multi-turn coherence
python test_conversational_snn.py --test_multi_turn

# Test energy consumption
python test_conversational_snn.py --test_energy

# Run all tests
python test_conversational_snn.py --test_all
```

### snn_multi_turn_conversation_test.py

Simple conversation smoke test.

```python
from snn_multi_turn_conversation_test import run_multi_turn_chat

conversation = run_multi_turn_chat(
    turns=3,
    timesteps=8,
    device_str="cuda",
    temperature=1.0,
    top_k=20,
    mode="snn"  # or "baseline"
)
```

## CLI Entry Points

### run_conversion.py

Main conversion script with comprehensive options.

```bash
python run_conversion.py \
    --model_name distilgpt2 \
    --timesteps 16 \
    --simplified \
    --output_dir ./snn_model
```

### smollm2_converter.py

Specialized converter for SmolLM2 models.

```bash
python smollm2_converter.py \
    --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --timesteps 32 \
    --max_context_length 2048
```

## Output Formats

### _CompatOutput

Custom output object that supports both HuggingFace and tensor-style access.

```python
outputs = model(input_ids)

# HuggingFace style
logits = outputs.logits
past_kv = outputs.past_key_values

# Tensor style  
logits = outputs[0]
next_token_logits = outputs[0, -1, :]
```

## Error Handling

Common issues and solutions:

**Memory Issues**: Use `--simplified` flag or reduce `--timesteps`
**SpikingJelly Compatibility**: Check version with `spikingjelly_compat.py`
**Position ID Overflow**: Automatic clamping in `TemporalSpikeProcessor`
**KV Cache Growth**: Automatic truncation at `max_context_length`

For implementation details, see [Project State Overview](PROJECT_STATE_OVERVIEW.md). 