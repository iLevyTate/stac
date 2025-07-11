# STAC V1: End-to-End Training Pipeline

## Overview

STAC V1 represents the **original research approach** - a complete end-to-end training pipeline for spiking transformers. This version established the foundational concepts that were later adapted for the conversion-based approach in STAC V2.

## Key Differences: V1 vs V2

| Aspect | STAC V1 | STAC V2 |
|--------|---------|---------|
| **Approach** | End-to-end training from scratch | ANN→SNN conversion |
| **Architecture** | Learnable AdEx neurons | Converted transformer layers |
| **Memory** | Hyperdimensional Memory Module (HEMM) | Temporal Spike Processor (TSP) |
| **Training** | Surrogate gradient training | Pre-trained model conversion |
| **Scope** | Single-turn processing | Multi-turn conversations |
| **Status** | Complete research prototype | Experimental conversion framework |

## STAC V1 Contributions

### 🧠 **Neuromorphic Architecture**
- **Learnable AdEx Neurons**: Adaptive exponential neurons with biologically plausible parameters
- **Surrogate Gradient Training**: Successful training of spiking transformers using surrogate gradients
- **L1 Spike Regularization**: Energy-efficient spike patterns

### 🧩 **Memory Integration**
- **Hyperdimensional Memory Module (HEMM)**: 1024-dimensional memory projection
- **Spike Pooling**: Temporal aggregation of spike trains
- **Memory Bias**: Context-aware processing

### 📊 **Validation Suite**
- **Comprehensive Testing**: Position ID boundaries, attention masks, spike rates
- **Energy Analysis**: Theoretical energy savings projections
- **Quality Metrics**: Perplexity and coherence measurements

## Implementation Details

### Model Architecture
```python
# Key components in stacv1.ipynb:
- AdEx neurons with learnable parameters (τ_m=20.0, τ_w=144.0, etc.)
- HEMM with 1024-dim projection matrix
- L1 regularization for energy efficiency
- Surrogate gradient training on WikiText-2
```

### Training Process
1. **Data Loading**: WikiText-2 raw dataset
2. **Model Initialization**: Learnable AdEx parameters
3. **Forward Pass**: Spike accumulation and memory integration
4. **Loss Computation**: Cross-entropy + L1 spike penalty
5. **Backward Pass**: Surrogate gradient updates

## Research Impact

STAC V1 demonstrated several key innovations:
- ✅ **First successful surrogate gradient training** of spiking transformers
- ✅ **Learnable neuromorphic dynamics** with AdEx neurons
- ✅ **Hyperdimensional memory integration** in spiking networks
- ✅ **Energy-efficient spike regularization** techniques

## Usage

```bash
# Open the Jupyter notebook
jupyter notebook stac-v1/stacv1.ipynb

# Or view in VS Code
code stac-v1/stacv1.ipynb
```

## Evolution to STAC V2

STAC V2 evolved from V1 by:
1. **Shifting to conversion-based approach** for practical deployment
2. **Extending to multi-turn conversations** with Temporal Spike Processor
3. **Focusing on hardware compatibility** for neuromorphic deployment
4. **Maintaining V1's energy efficiency principles** in conversion framework

---

**Note**: STAC V1 is a **complete research prototype** that has been validated and documented. STAC V2 builds upon these foundations with a different methodological approach focused on practical deployment. 