# Conversion Workflow

> **Scope note (2026-07).** This document describes the *mechanics* of the conversion pipeline.
> By default that pipeline produces a numerically faithful, **non-spiking** model (the LIF
> neurons in `SpikeAttention` are bypassed). Turning spiking on collapses a frozen converted
> model to near-constant output, and the projected energy of the current design is **worse** than
> the dense ANN (~7.6×) at its spike coverage. The multi-turn machinery below is real, but its
> conversational *quality* under genuine spiking does not survive without training. See
> [`coverage-quality.md`](coverage-quality.md), [`energy-crossover.md`](energy-crossover.md), and
> [`findings-summary.md`](findings-summary.md) for the measured position.

## Overview

The STAC framework provides two main conversion approaches:
1. **Simplified Conversion**: Fast, basic ANN→SNN transformation
2. **Full Pipeline**: Comprehensive conversion with quantization and calibration

## Conversion Process

### Step 1: Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
```

### Step 2: Architecture Conversion
The conversion process involves three main transformations:

1. **Activation Replacement**: GELU → ReLU
2. **Normalization Replacement**: LayerNorm → SpikeLayerNorm  
3. **Attention Replacement**: Standard Attention → SpikeAttention

### Step 3: Temporal Wrapper
```python
from smollm2_converter import TemporalSpikeProcessor

# Wrap with multi-turn capability
snn_model = TemporalSpikeProcessor(converted_model, T=16)
```

## Conversion Modes

### Simplified Mode
**Purpose**: Fast testing and development  
**Time**: 2-15 minutes  
**Features**:
- Basic layer replacement
- No quantization
- Minimal calibration

```bash
python scripts/run_conversion.py --model_name distilgpt2 --simplified --timesteps 8
```

### Full Pipeline Mode
**Purpose**: Research-grade conversion (experimental)  
**Time**: 1-3 hours  
**Features**:
- 8-bit quantization
- Extensive calibration
- Threshold optimization

```bash
python scripts/run_conversion.py --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct --timesteps 16
```

## Supported Models

### Currently Supported
- **DistilGPT-2**: Lightweight GPT-2 variant
- **SmolLM2-1.7B-Instruct**: Instruction-tuned language model

### Model Requirements
- Must be causal language models
- Transformer architecture
- HuggingFace compatible

## Conversion Parameters

### Key Parameters
- `--timesteps`: Number of SNN timesteps (8-64)
- `--simplified`: Use simplified conversion
- `--model_name`: Source model identifier
- `--output_dir`: Output directory

### Advanced Parameters
- `--surrogate_function`: Surrogate gradient function
- `--use_sparse`: Enable sparse tensor optimization
- `--verify`: Run post-conversion verification

## Multi-Turn Capability

> The `TemporalSpikeProcessor` machinery below (cache, positions, batching) works as described in
> the faithful non-spiking path. Under genuine spiking, a frozen converted model does not retain
> coherent multi-turn quality without training — the mechanism is present, the quality is not.

### TemporalSpikeProcessor Features
- **KV Cache Management**: Maintains context across turns
- **Position ID Handling**: Manages sequence positions
- **Batch Processing**: Supports multiple conversations

### Usage Example
```python
processor = TemporalSpikeProcessor(snn_model, T=16, max_context_length=512)

# Multi-turn conversation
for turn in conversation_turns:
    output = processor(input_ids, use_cache=True)
    # Process output...
```

## Validation and Testing

### Automatic Validation
The conversion process includes built-in validation:
- Position ID boundary testing
- Attention mask continuity
- Multi-turn cache/state checks (behavioral pinning, not a demonstration of conversational quality)
- Spike rate analysis

### Manual Testing
```bash
# Run comprehensive tests
python tests/test_conversational_snn.py --model_name distilgpt2 --test_all --timesteps 16

# Test specific components
python tests/test_conversational_snn.py --model_name distilgpt2 --test_multi_turn
```

## Output Format

### Saved Model Structure
```
output_dir/
├── snn_model.pt              # state_dict + metadata bundle (not a live nn.Module)
├── snn_config.json           # timesteps, conversion_mode, simplified flag, base_model
├── conversion_summary.json   # written by scripts/run_conversion.py
├── config.json               # model configuration
├── tokenizer.json            # tokenizer files are written at the top level,
├── tokenizer_config.json     #   not into a tokenizer/ subdirectory
├── vocab.json
├── merges.txt
└── special_tokens_map.json
```

Note: `snn_model.pt` holds `{"state_dict": ..., "config": ..., "model_type": ..., "T": ...,
"simplified": ...}`. Load it by constructing the base model and applying the state dict —
`torch.load` alone does not return a runnable module. `scripts/run_conversion.py --verify`
performs exactly that round-trip.

### Model Metadata
The saved model includes:
- Original model information
- Conversion parameters
- Timestep configuration
- Simplified/full mode flag

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or use CPU
2. **Conversion Failures**: Try simplified mode first
3. **Import Errors**: Verify SpikingJelly version >= 0.0.0.0.14

### Performance Tips
1. Start with simplified mode for testing
2. Use smaller timesteps (8-16) for faster conversion
3. Ensure adequate GPU memory for large models

## Future Enhancements

### Planned Features
- Additional model architectures
- Hardware-specific optimizations
- Automated hyperparameter tuning
- Real-time conversion monitoring

### Research Directions
- Improved spike encoding methods
- Advanced calibration techniques
- Multi-modal SNN support 