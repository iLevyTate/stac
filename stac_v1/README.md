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
| **Status** | Research prototype; spiking pathway inactive until the 2026-07 audit | Experimental conversion framework |

> **Correction (2026-07).** In every release up to and including `3.0.0-beta`, the AdEx
> neurons here emitted **zero** spikes for every input. Two coupled defects were
> responsible — see [`docs/corrigendum-2026-07.md`](../docs/corrigendum-2026-07.md) for the
> full account and the reproduction. Descriptions of this component published before that
> date, including in the accompanying paper, describe intended rather than observed
> behaviour. The claims below are the post-fix ones.

## STAC V1 Components

### 🧠 **Neuromorphic Architecture**
- **Learnable AdEx Neurons**: adaptive exponential neurons; `tau_m`, `tau_w`, `a`, `b` and
  `delta_T` are trained, while `V_th`/`V_reset`/`V_rest` stay fixed by design
- **Surrogate Gradient Training**: Gaussian surrogate, width tied to `delta_T` so the
  gradient survives the millivolt scale of `V - V_th`. At unit width it underflowed float32
  to exactly zero and no gradient reached the spiking layer
- **L1 Spike Regularization**: active at a measured ~0.147 spike rate. Identically zero
  before the neurons were made excitable, so it exerted no pressure

### 🧩 **Memory Integration**
- **Hyperdimensional Memory Module (HEMM)**: 1024-dimensional memory projection
- **Spike Pooling**: causal aggregation of spike trains. Pooling over the whole sequence
  (the earlier behaviour) leaked future tokens into a next-token objective
- **Memory Bias**: Context-aware processing

### 📊 **Validation Suite**
- **Regression baseline**: `tests/test_v1_baseline.py` against `docs/baselines/stac_v1_smoke.json`
- **Spike telemetry**: real spike and synaptic-operation counts via `spike_metrics.py`
- **Reported metrics**: training loss and perplexity on a random-weight smoke model — these
  pin behaviour, not language quality

## Implementation Details

### Model Architecture
```python
# Key components in stac_v1/model.py:
- AdEx neurons with learnable parameters (τ_m=20.0, τ_w=144.0, etc.)
- HEMM with 1024-dim projection matrix
- L1 regularization for energy efficiency
- Surrogate gradient training (`--dataset wikitext2` when the optional `datasets`
  package is installed; built-in sample texts otherwise)
```

### Training Process
1. **Data Loading**: built-in sample texts, `--texts_file`, or WikiText-2 via
   `--dataset wikitext2` (needs the optional `datasets` package)
2. **Model Initialization**: Learnable AdEx parameters
3. **Forward Pass**: Spike accumulation and memory integration
4. **Loss Computation**: Cross-entropy + L1 spike penalty
5. **Backward Pass**: Surrogate gradient updates

## Research Impact

> Claims below are limited to what this repository measures. See
> `docs/baselines/stac_v1_smoke.json` for the numbers and the command that produces them.

STAC V1 implements:
- **Surrogate-gradient training** of a spiking layer on top of a frozen transformer
  backbone, end to end
- **Learnable neuromorphic dynamics** with AdEx neurons (tau_m, tau_w, a, b, delta_T are
  trained; V_th/V_reset/V_rest stay fixed by design)
- **Hyperdimensional memory integration** over spike trains, pooled causally
- **L1 spike regularization**, active at a measured ~0.147 spike rate / 86.8% sparsity
  (see `docs/baselines/stac_v1_smoke.json`). This term was identically zero until the
  neurons were made excitable.

## Usage

```bash
# Run the repo-native STAC V1 pipeline smoke (recommended)
# Demonstrates *hybrid fine-tuning*: frozen GPT-2 backbone + trained spiking/memory head
# Writes a simulation-time Loihi constraints report under stac_v1_output/
python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 3

# Provide your own texts (either a file, or repeated --text args)
python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --texts_file local/paper_excerpts.txt --steps 5
python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --text "Hello." --text "Neuromorphic edge is constrained." --steps 3

# Optional checkpointing (weights + config + last_run_summary snapshot)
python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 3 --checkpoint_out stac_v1_output/checkpoint.pth
python scripts/run_stac_v1.py --model_name sshleifer/tiny-gpt2 --steps 1 --checkpoint_in  stac_v1_output/checkpoint.pth
```

## Evolution to STAC V2

STAC V2 evolved from V1 by:
1. **Shifting to conversion-based approach** for practical deployment
2. **Extending to multi-turn conversations** with Temporal Spike Processor
3. **Focusing on hardware compatibility** for neuromorphic deployment
4. **Maintaining V1's energy efficiency principles** in conversion framework

---

**Note**: STAC V1 is a research prototype. Its spiking pathway has been functional and
covered by a regression baseline since the 2026-07 audit; before that it was structurally
present but inert. STAC V2 builds on the same foundations with a conversion-based approach
aimed at practical deployment.
