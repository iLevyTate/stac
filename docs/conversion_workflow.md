# Conversion Workflow

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
python run_conversion.py --model_name distilgpt2 --simplified --timesteps 8
```

### Full Pipeline Mode
**Purpose**: Production-ready conversion  
**Time**: 1-3 hours  
**Features**:
- 8-bit quantization
- Extensive calibration
- Threshold optimization

```bash
python run_conversion.py --model_name SmolLM2-1.7B-Instruct --timesteps 16
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
- Multi-turn coherence verification
- Spike rate analysis

### Manual Testing
```bash
# Run comprehensive tests
python test_conversational_snn.py --test_all --timesteps 16

# Test specific components
python test_conversational_snn.py --test_multi_turn
```

## Output Format

### Saved Model Structure
```
output_dir/
├── snn_model.pt          # Converted SNN model
├── tokenizer/            # Tokenizer files
├── config.json           # Model configuration
└── conversion_log.txt    # Conversion details
```

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