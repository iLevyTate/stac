# API Reference

## Core Classes

### TemporalSpikeProcessor

Main class for multi-turn conversational SNN processing.

```python
class TemporalSpikeProcessor(nn.Module):
    def __init__(self, snn_model, T=16, max_context_length=512):
        """
        Initialize the temporal spike processor.
        
        Args:
            snn_model: The converted SNN model
            T: Number of timesteps for spike processing
            max_context_length: Maximum sequence length
        """
```

#### Methods

##### `forward(input_ids, attention_mask=None, use_cache=True, **kwargs)`
Process input through the SNN with temporal dynamics.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs
- `attention_mask` (torch.Tensor, optional): Attention mask
- `use_cache` (bool): Whether to use KV cache for multi-turn
- `**kwargs`: Additional model arguments

**Returns:**
- Model output with logits and optional past key values

##### `reset_cache(batch_id=None)`
Reset the KV cache for new conversations.

**Parameters:**
- `batch_id` (int, optional): Specific batch to reset

##### `get_position_ids()`
Get current position IDs for the conversation.

**Returns:**
- Dictionary with position ID information

### SpikeAttention

Spiking-compatible attention mechanism.

```python
class SpikeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, T=16, causal=True):
        """
        Initialize spike-based attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            T: Number of timesteps
            causal: Whether to use causal attention
        """
```

### SpikeLayerNorm

Spiking-compatible layer normalization.

```python
class SpikeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize spike-compatible layer normalization.
        
        Args:
            normalized_shape: Input shape to normalize
            eps: Small constant for numerical stability
        """
```

## Conversion Functions

### `replace_gelu_with_relu(model)`

Replace GELU activations with ReLU for SNN compatibility.

**Parameters:**
- `model` (torch.nn.Module): Model to modify

**Returns:**
- Modified model with ReLU activations

### `simplified_conversion(model, timesteps=32)`

Perform simplified ANN→SNN conversion.

**Parameters:**
- `model` (torch.nn.Module): Source model
- `timesteps` (int): Number of SNN timesteps

**Returns:**
- Converted SNN model

### `replace_layernorm_with_spikelayernorm(model)`

Replace LayerNorm with SpikeLayerNorm.

**Parameters:**
- `model` (torch.nn.Module): Model to modify

**Returns:**
- Modified model with spike-compatible normalization

### `replace_attention_with_spikeattention(model)`

Replace standard attention with SpikeAttention.

**Parameters:**
- `model` (torch.nn.Module): Model to modify

**Returns:**
- Modified model with spike-compatible attention

### `apply_surrogate_gradients(model, alpha=4.0)`

Apply surrogate gradient functions for SNN training.

**Parameters:**
- `model` (torch.nn.Module): SNN model
- `alpha` (float): Surrogate gradient scaling factor

**Returns:**
- Model with surrogate gradients

### `calibrate_timesteps(model, original_T, target_T)`

Calibrate spike timing for different timestep counts.

**Parameters:**
- `model` (torch.nn.Module): SNN model
- `original_T` (int): Original timestep count
- `target_T` (int): Target timestep count

**Returns:**
- Calibrated model

### `save_snn_model(model, tokenizer, path)`

Save the converted SNN model with metadata.

**Parameters:**
- `model` (torch.nn.Module): SNN model to save
- `tokenizer`: Associated tokenizer
- `path` (str): Save path

**Returns:**
- Success status

## Utility Functions

### `create_calibration_data(tokenizer, num_samples=10, max_length=128)`

Create calibration data for SNN conversion.

**Parameters:**
- `tokenizer`: HuggingFace tokenizer
- `num_samples` (int): Number of calibration samples
- `max_length` (int): Maximum sequence length

**Returns:**
- Dictionary with calibration data

## Testing Functions

### `test_position_id_boundaries(model, tokenizer, args)`

Test position ID handling at sequence boundaries.

**Parameters:**
- `model`: SNN model to test
- `tokenizer`: Associated tokenizer
- `args`: Test configuration

**Returns:**
- Test results

### `test_attention_mask_continuity(model, tokenizer, args)`

Test attention mask continuity across conversation turns.

**Parameters:**
- `model`: SNN model to test
- `tokenizer`: Associated tokenizer
- `args`: Test configuration

**Returns:**
- Test results

### `test_multi_turn_coherence(model, tokenizer, args)`

Test multi-turn conversation coherence.

**Parameters:**
- `model`: SNN model to test
- `tokenizer`: Associated tokenizer
- `args`: Test configuration

**Returns:**
- Test results

### `simulate_conversation(model, tokenizer, turns=3, device="cpu")`

Simulate a multi-turn conversation for testing.

**Parameters:**
- `model`: SNN model
- `tokenizer`: Associated tokenizer
- `turns` (int): Number of conversation turns
- `device` (str): Computing device

**Returns:**
- Conversation results

## Command Line Interface

### `run_conversion.py`

Main CLI tool for model conversion.

**Usage:**
```bash
python run_conversion.py [OPTIONS]
```

**Options:**
- `--model_name`: Model to convert (distilgpt2, SmolLM2-1.7B-Instruct)
- `--output_dir`: Output directory
- `--timesteps`: Number of SNN timesteps
- `--simplified`: Use simplified conversion
- `--verify`: Run post-conversion verification

### `test_conversational_snn.py`

Testing and validation tool.

**Usage:**
```bash
python test_conversational_snn.py [OPTIONS]
```

**Options:**
- `--test_all`: Run all tests
- `--test_position_boundaries`: Test position ID boundaries
- `--test_attention_mask`: Test attention mask continuity
- `--test_multi_turn`: Test multi-turn capabilities
- `--test_energy`: Test energy consumption

## Configuration

### Model Parameters

**Supported Models:**
- `distilgpt2`: DistilGPT-2 (117M parameters)
- `SmolLM2-1.7B-Instruct`: SmolLM2 1.7B Instruct (1.7B parameters)

**Conversion Parameters:**
- `timesteps`: 8-64 (recommended: 16)
- `max_context_length`: 512-2048 (recommended: 512)
- `surrogate_function`: atan, sigmoid, stbif_plus

### Hardware Configuration

**GPU Memory Requirements:**
- DistilGPT-2: 4-8 GB
- SmolLM2-1.7B-Instruct: 20 GB

**CPU Requirements:**
- Multi-core processor recommended
- 16-32 GB RAM

## Error Handling

### Common Exceptions

**ImportError**: SpikingJelly version compatibility
```python
# Ensure SpikingJelly >= 0.0.0.0.14
pip install spikingjelly[cuda] -U --pre
```

**CUDA Out of Memory**: Insufficient GPU memory
```python
# Reduce batch size or use CPU
device = 'cpu'
```

**Position ID Errors**: Sequence length exceeds model limits
```python
# Reduce max_context_length
max_context_length = 512
```

## Examples

### Basic Conversion
```python
from smollm2_converter import *

# Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Convert to SNN
snn_model = simplified_conversion(model, timesteps=16)

# Wrap with temporal processor
processor = TemporalSpikeProcessor(snn_model, T=16)

# Test conversation
result = simulate_conversation(processor, tokenizer, turns=3)
```

### Advanced Usage
```python
# Full pipeline conversion
from convert import convert_model_to_spiking, create_calibration_data

# Create calibration data
calib_data = create_calibration_data(tokenizer, num_samples=10)

# Convert with calibration
snn_model = convert_model_to_spiking(model, calib_data, timesteps=32)

# Apply surrogate gradients
snn_model = apply_surrogate_gradients(snn_model, alpha=4.0)

# Save model
save_snn_model(snn_model, tokenizer, "./my_snn_model")
``` 