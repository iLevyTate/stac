# Hardware Requirements

This document outlines the hardware requirements for running the STAC conversion framework and testing multi-turn conversational SNN models.

## Conversion Requirements

### Fast Conversion (Simplified Pipeline)

**Recommended for testing and development**

- **CPU**: 4+ cores, 8GB RAM
- **GPU**: Optional (CPU conversion works well)
- **Models**: DistilGPT-2, GPT-2 small/medium
- **Time**: ~2-5 minutes

```bash
python run_conversion.py --model_name distilgpt2 --timesteps 8 --simplified
```

### Full SpikingJelly Conversion

**For production-quality models with calibration**

- **CPU**: 8+ cores, 16GB+ RAM
- **GPU**: 8GB+ VRAM (NVIDIA GTX 1070 or better)
- **Models**: DistilGPT-2, GPT-2 variants
- **Time**: ~10-30 minutes

### Large Model Conversion (SmolLM2-1.7B)

**For state-of-the-art conversational models**

- **CPU**: 16+ cores, 32GB+ RAM
- **GPU**: 20GB+ VRAM (NVIDIA RTX 3090/4090, A100)
- **Models**: SmolLM2-1.7B-Instruct, Llama-2-7B
- **Time**: ~1-3 hours

```bash
python smollm2_converter.py --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct --timesteps 32
```

## Inference Requirements

### CPU Inference

- **Minimum**: 4 cores, 8GB RAM
- **Recommended**: 8+ cores, 16GB RAM
- **Performance**: ~1-5 tokens/second for DistilGPT-2

### GPU Inference

- **Minimum**: 4GB VRAM (NVIDIA GTX 1050 Ti)
- **Recommended**: 8GB+ VRAM (NVIDIA RTX 3070+)
- **Performance**: ~10-50 tokens/second depending on model size

## Testing Requirements

### Comprehensive Test Suite

Running `test_conversational_snn.py --test_all`:

- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ for large context tests
- **Time**: ~10-30 minutes depending on model size

### Memory Usage by Model

| Model | Conversion RAM | Inference RAM | GPU VRAM |
|-------|----------------|---------------|----------|
| DistilGPT-2 | 4GB | 2GB | 2GB |
| GPT-2 Medium | 8GB | 4GB | 4GB |
| SmolLM2-1.7B | 20GB | 8GB | 12GB |

## Platform Compatibility

### Operating Systems

- ✅ **Windows 10/11** (tested)
- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+)
- ✅ **macOS** (Intel/Apple Silicon)

### Python Environment

- **Python**: 3.8-3.11
- **PyTorch**: 2.0+
- **SpikingJelly**: 0.0.0.0.14+
- **Transformers**: 4.20+

## Cloud Deployment

### Recommended Cloud Instances

| Provider | Instance Type | vCPUs | RAM | GPU | Use Case |
|----------|---------------|-------|-----|-----|----------|
| **AWS** | g4dn.xlarge | 4 | 16GB | T4 (16GB) | Development |
| **AWS** | p3.2xlarge | 8 | 61GB | V100 (16GB) | Production |
| **GCP** | n1-standard-8 | 8 | 30GB | T4 (16GB) | Development |
| **Azure** | Standard_NC6s_v3 | 6 | 112GB | V100 (16GB) | Production |

### Cost Optimization

- Use **CPU-only instances** for simplified conversion and testing
- Use **spot instances** for batch conversion jobs
- Use **preemptible VMs** on GCP for cost savings

## Performance Benchmarks

### Conversion Speed (DistilGPT-2)

- **CPU (simplified)**: ~2 minutes
- **GPU (simplified)**: ~1 minute  
- **GPU (full calibration)**: ~15 minutes

### Inference Speed (Multi-turn conversation)

- **CPU**: ~2-5 tokens/second
- **GPU (T4)**: ~15-25 tokens/second
- **GPU (V100)**: ~30-50 tokens/second

## Troubleshooting

**Out of Memory**: Use `--simplified` flag or reduce `--timesteps`
**Slow Conversion**: Enable GPU acceleration or use cloud instances
**CUDA Issues**: Ensure PyTorch CUDA version matches your driver

For detailed setup instructions, see [Conversion Workflow](conversion_workflow.md). 