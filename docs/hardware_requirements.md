# Hardware Requirements

## System Requirements

### Minimum Requirements
- **RAM**: 16 GB (32 GB recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080/4080/H100)
- **Storage**: 20 GB free space
- **CPU**: Multi-core processor (Intel i7/AMD Ryzen 7+)

### Recommended Requirements
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 20GB+ VRAM (RTX 4090/H100)
- **Storage**: 50 GB free space (SSD recommended)
- **CPU**: High-end multi-core processor

## Model-Specific Requirements

### DistilGPT-2
- **Conversion Time**: 2-5 minutes
- **VRAM**: 4-8 GB
- **Model Size**: ~500 MB

### SmolLM2-1.7B-Instruct  
- **Conversion Time**: 1-3 hours
- **VRAM**: 20 GB
- **Model Size**: ~3.5 GB

## Supported Hardware

### Current Support
- ✅ **Software Simulation**: Full support on CPU/GPU
- ✅ **NVIDIA GPUs**: CUDA 11.8+ or 12.1+
- ✅ **PyTorch**: 2.0.0 - 2.5.x

### Planned Support (Future Work)
- ⏳ **Intel Loihi-2**: Neuromorphic hardware deployment
- ⏳ **BrainChip Akida**: Edge neuromorphic processing
- ⏳ **SpiNNaker**: Large-scale spiking neural network platform

## Installation Notes

### CUDA Installation
```bash
# For CUDA 11.8
pip install torch==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 12.1  
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only
pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### SpikingJelly Installation
```bash
# Latest pre-release version required
pip install spikingjelly[cuda] -U --pre
```

## Performance Expectations

### Conversion Performance
- **Simplified Mode**: 2-15 minutes
- **Full Pipeline**: 1-3 hours (with quantization and calibration)

### Memory Usage
- **Peak VRAM**: 20GB (SmolLM2-1.7B-Instruct)
- **System RAM**: 16-32GB during conversion

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU fallback
2. **SpikingJelly Version**: Ensure version >= 0.0.0.0.14
3. **PyTorch Compatibility**: Use PyTorch 2.0.0 - 2.5.x range

### Performance Optimization
1. Use SSD storage for faster I/O
2. Close unnecessary applications during conversion
3. Use simplified mode for initial testing 