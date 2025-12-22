# STAC Dissertation Defense Q&A

## Aligned Minds, Efficient Machines: Integrating Neuromorphic Computing for Personalized AI

**Author:** Ben Kennedy
**Institution:** Capitol Technology University
**Publication:** IGI Global Scientific Publishing, 2026
**DOI:** 10.4018/979-8-3373-5702-7.ch005

---

## Table of Contents

1. [Technical Architecture Questions](#1-technical-architecture-questions)
2. [V1 to V2 Methodology Questions](#2-v1-to-v2-methodology-questions)
3. [Energy Efficiency & Hardware Questions](#3-energy-efficiency--hardware-questions)
4. [Limitations & Future Work Questions](#4-limitations--future-work-questions)
5. [Ethical & Comparative Questions](#5-ethical--comparative-questions)

---

## 1. Technical Architecture Questions

### Q1: Why did you choose AdEx neurons over simpler LIF neurons in STAC V1?

**Answer:** AdEx (Adaptive-Exponential) neurons capture a richer repertoire of biologically plausible dynamics than LIF neurons, including:

- **Spike-frequency adaptation**: Neurons fire less frequently with sustained input
- **Bursting behavior**: Rapid successive spikes followed by quiescence
- **Learnable parameters**: Time constants (tau_m, tau_w) and slope factors (delta_T) can be optimized during training

This biological plausibility was central to V1's research contribution. The paper states: "The choice of the Adaptive-Exponential (AdEx) integrate-and-fire neuron model, based on the foundational work of Brette & Gerstner (2005), was deliberate" (Page 132).

---

### Q2: How does the surrogate gradient technique enable training of spiking networks?

**Answer:** Spiking is inherently non-differentiable (step function at threshold). The surrogate gradient technique replaces the true gradient with a smooth approximation during backpropagation:

- **Forward pass**: Binary spike output (non-differentiable step function)
- **Backward pass**: Gaussian bump surrogate (smooth, differentiable proxy)

This allows standard gradient-based optimization (Adam, SGD) to train spiking networks end-to-end. As noted in the paper: "This proxy provides a useful learning signal that allows gradients to flow back through the spiking neurons" (Page 133).

---

### Q3: What is the purpose of the Hyperdimensional Memory Module (HEMM)?

**Answer:** HEMM provides context-aware memory beyond immediate spike dynamics through three steps:

1. **Temporal Aggregation**: Pools spike trains across the time dimension via mean operation
2. **Hyperdimensional Encoding**: Projects pooled vector to 1024-dimensional space via fixed random projection matrix
3. **Memory Bias Generation**: MLP transforms high-dimensional representation into bias added to token representations

This enables the model's own recent activity to influence current processing. The paper notes: "This mechanism allows the model's own recent past activity to influence its current processing in a structured, recurrent, and computationally efficient manner" (Page 133).

---

### Q4: Why does STAC V2 disable spiking neurons in attention despite having them implemented?

**Answer:** Enabling spiking neurons in attention degraded text generation to unusable levels (repeated tokens, gibberish output). This is a pragmatic trade-off documented in the paper: "The resulting SNN, while stateful, suffers from a clear degradation in conversational coherence and nuance compared to its source ANN" (Page 136-137).

The framework maintains the infrastructure for future hardware deployment where fine-tuning can recover quality. This represents a conscious design choice to prioritize usability during the research phase while preserving the path to full neuromorphic deployment.

---

### Q5: How does the Temporal Spike Processor (TSP) maintain conversational context?

**Answer:** TSP implements three complementary mechanisms:

1. **KV Cache**: Standard transformer key-value caching for efficient incremental generation
2. **Batch-Specific Caches**: Separate cache dictionaries for multi-conversation batching scenarios
3. **Token Cache**: Incremental token storage for turn-by-turn processing

The paper states: "The key innovation in V2 is the novel Temporal Spike Processor (TSP), a state-management module designed to preserve conversational context across multiple turns" (Page 119).

---

### Q6: What is the role of the L1 spike regularization term?

**Answer:** L1 regularization encourages sparse spiking activity by adding a penalty term to the loss function:

**Total Loss = Cross-Entropy Loss + λ × mean(|spikes|)**

Where λ = 1e-5 by default. By penalizing the mean absolute spike value, the optimizer finds solutions with more zeros (no spikes), which:

- **Reduces energy consumption** on neuromorphic hardware (fewer synaptic operations)
- **Aligns with biological efficiency** (only ~1% of biological neurons active at any time)
- **Enables event-driven computation** (compute only when spikes occur)

---

## 2. V1 to V2 Methodology Questions

### Q7: Why did the research pivot from V1's hybrid fine-tuning to V2's conversion approach?

**Answer:** Two primary factors motivated the pivot:

1. **Scalability**: V1's end-to-end training is computationally expensive and doesn't scale to modern LLMs with billions of parameters
2. **Accessibility**: V2 can leverage the vast ecosystem of existing pretrained models without requiring neuromorphic-aware training from scratch

The paper explicitly states: "While STAC V1 was a successful, self-contained research project, its immense computational cost and the inherent difficulty of scaling its bespoke training-based approach to compete with state-of-the-art LLMs prompted a strategic methodological shift" (Page 118-119).

---

### Q8: What are the trade-offs between V1 and V2 approaches?

**Answer:**

| Aspect | V1 (Fine-Tuning) | V2 (Conversion) |
|--------|------------------|-----------------|
| **Biological Plausibility** | Higher (AdEx, learnable dynamics) | Lower (LIF, currently disabled) |
| **Computational Cost** | High (full training required) | Low (post-hoc conversion) |
| **Model Compatibility** | Limited to small models | Any pretrained model |
| **Quality Control** | Integrated L1 regularization | Post-hoc spike telemetry |
| **Conversation Support** | Single-turn only | Multi-turn with TSP |
| **Research Focus** | Architectural innovation | Practical deployment |

---

### Q9: How does STAC V2 handle the "coherence-fidelity trade-off"?

**Answer:** The paper acknowledges this fundamental challenge:

> "The degradation in conversational quality observed in STAC V2 is a direct and predictable consequence of the information loss inherent in the ANN-to-SNN conversion process" (Page 135-136).

Mitigation strategies include:

1. **Skip GELU replacement**: Option to preserve original activation for quality testing
2. **Knowledge distillation**: Train SNN to match ANN output distributions
3. **LoRA adapters**: Fine-tune small adapter layers post-conversion
4. **Post-conversion fine-tuning**: Targeted training on small datasets

---

### Q10: Why is HEMM not implemented in V2?

**Answer:** V2 uses a fundamentally different architectural paradigm:

- **V1**: Custom spiking architecture built from scratch with integrated HEMM
- **V2**: Converted standard transformer maintaining native mechanisms

The TSP provides equivalent functionality through:
- Standard transformer KV caching (preserves attention context)
- Batch-specific cache management (multi-conversation support)
- Incremental token caching (turn-by-turn processing)

This aligns with V2's core goal of converting existing models rather than building custom architectures from scratch.

---

## 3. Energy Efficiency & Hardware Questions

### Q11: What theoretical energy savings does STAC project?

**Answer:** The paper projects 3-4x energy savings based on three principles:

1. **Event-driven computation**: SNNs compute only when spikes arrive (vs. continuous computation in ANNs)
2. **Temporal sparsity**: L1 regularization encourages 50-90% zero activations
3. **Synaptic operations**: Single SOP (synaptic operation) requires far less energy than dense MAC (multiply-accumulate) operations

However, as explicitly stated: "Its projected three- to four-fold energy savings are purely theoretical, derived from post-hoc spike telemetry" (Page 136). Hardware validation is required to confirm these projections.

---

### Q12: Why hasn't STAC been validated on Loihi hardware yet?

**Answer:** Several technical blockers remain:

1. **Dense Attention**: Standard transformer attention uses full QK^T softmax over the entire context, which is not directly mappable to Loihi primitives
2. **Incomplete Operator Coverage**: Some transformer operations lack neuromorphic hardware equivalents
3. **Hardware Access**: Loihi-2 access requires Intel Neuromorphic Research Community membership

The paper proposes: "A one-month study to benchmark energy consumption on neuromorphic hardware" as a concrete next step (Page 138).

---

### Q13: What is the LoihiCausalContextMixer and when should it be used?

**Answer:** The LoihiCausalContextMixer is a Loihi-oriented attention replacement designed to remove the dense QK^T/softmax operation that blocks Loihi export.

It implements a recurrent context mechanism where context is updated incrementally at each timestep, avoiding the need for full attention computation across all positions.

**Use cases:**
- Testing Loihi export readiness
- Research on event-driven attention alternatives
- Applications accepting quality trade-off for hardware compatibility

---

### Q14: How does spike sparsity contribute to energy efficiency?

**Answer:** Energy efficiency derives from sparse computation:

| Computation Type | Energy Profile |
|-----------------|----------------|
| Dense MAC (ANN) | Compute every element every clock cycle |
| Sparse SOP (SNN) | Compute only for non-zero spikes |

With 90% sparsity (90% of values are zero), SNNs perform approximately 10x fewer operations. The L1 regularization in V1 directly optimizes for this sparsity:

- Penalizes non-zero spike values during training
- Results in networks that naturally produce sparse activations
- Enables significant energy savings on event-driven hardware

---

### Q15: What validation metrics will be used for hardware benchmarking?

**Answer:** Proposed metrics from the paper (Page 131):

1. **Spike-count telemetry**: Total spikes per inference (proxy for energy)
2. **Watt-hour consumption**: Direct power measurement on hardware
3. **Synaptic operations (SOPs)**: Standard neuromorphic compute metric
4. **Latency**: Time per token generation
5. **Quality metrics**: Perplexity, coherence scores, task accuracy

These metrics will enable direct comparison between theoretical projections and empirical measurements.

---

## 4. Limitations & Future Work Questions

### Q16: What are the "hard blocks" for Loihi export?

**Answer:** The Loihi constraints validator identifies several blocking issues:

1. **Dense Attention**: QK^T softmax computation over full context windows cannot be directly mapped to Loihi's event-driven primitives
2. **Floating-point Parameters**: Loihi requires int8 quantization; floating-point weights must be converted
3. **Non-spike-compatible Activations**: GELU and complex normalizations have no direct neuromorphic equivalent

These blockers are documented and tracked, with proposed alternatives (like LoihiCausalContextMixer) provided for future resolution.

---

### Q17: How will you address the generation quality degradation in V2?

**Answer:** Multiple approaches are planned:

1. **Knowledge Distillation**: Train SNN to match ANN output probability distributions using KL divergence loss at elevated temperature

2. **LoRA Adapters**: Fine-tune small low-rank adapter layers post-conversion to recover task-specific performance

3. **Alternative Activations**: Explore SiLU, Leaky ReLU, and other activations that are more spike-compatible than GELU

4. **Quantization-Aware Training**: Train source models with lower precision expectations to reduce conversion loss

---

### Q18: What is the proposed timeline for hardware validation?

**Answer:** The paper proposes two concrete studies:

1. **One-month hardware study**: Benchmark energy consumption on Intel Loihi-2
   - Direct power measurements
   - Spike telemetry validation
   - Latency profiling

2. **Three-month longitudinal trial**: Evaluate user alignment stability
   - SCANAQ-based personalization metrics
   - Cognitive augmentation effectiveness
   - User experience data collection

These studies would transform theoretical projections into empirical validation.

---

## 5. Ethical & Comparative Questions

### Q19: What ethical considerations does the framework address?

**Answer:** The paper addresses four key ethical concerns (Pages 138-139):

1. **Cognitive Over-reliance**: Users may become dependent on AI guidance
   - *Mitigation*: Periodic prompts encouraging independent reflection

2. **Psychometric Data Privacy**: SCANAQ captures sensitive cognitive profiles
   - *Mitigation*: ISO-27001 compliant storage, encryption, explicit consent protocols

3. **Assessment Bias**: Cultural and linguistic biases in psychological scales
   - *Mitigation*: Cross-cultural validation studies, Differential Item Functioning (DIF) analysis

4. **Accountability**: Clear governance for AI-assisted decisions
   - *Mitigation*: Independent ethics review boards, public failure case reports

---

### Q20: How does STAC compare to other neuromorphic approaches?

**Answer:**

| Approach | STAC | SpikingJelly | Lava (Intel) |
|----------|------|--------------|--------------|
| **Focus** | Conversational AI | General SNN toolkit | Loihi programming |
| **Conversion** | Custom pipeline with TSP | Standard ANN-SNN Converter | Native Loihi compilation |
| **Memory** | HEMM (V1) / TSP (V2) | Standard SNN mechanisms | Hardware-native |
| **Personalization** | SCANAQ integration | None | None |
| **Target Models** | Transformers/LLMs | General architectures | Loihi-specific |

**STAC's unique contributions:**
- Psychometric alignment (SCANAQ → SCANUE → STAC pipeline)
- Multi-turn conversation support (TSP innovation)
- Hybrid fine-tuning approach with learnable dynamics (V1)

---

### Q21: What distinguishes STAC from standard ANN-to-SNN conversion?

**Answer:** STAC provides several unique capabilities beyond standard conversion:

1. **Conversational State Management**: TSP handles multi-turn context preservation
2. **Loihi-Aware Conversion**: Embedding bucketing, pruning, and quantization utilities
3. **Hybrid Fine-Tuning Option**: V1's end-to-end learnable spiking approach
4. **Integrated Validation**: Export readiness checking for neuromorphic deployment
5. **SCANUE Integration Path**: Connection to personalized cognitive agent architecture

Standard converters typically focus on single-inference scenarios without conversational context or personalization integration.

---

### Q22: How does the framework ensure reproducibility?

**Answer:** Multiple mechanisms ensure research reproducibility:

1. **Seed Control**: Deterministic random seeds for training and evaluation
2. **Configuration Dataclasses**: All hyperparameters captured in frozen dataclasses
3. **Checkpoint System**: Model weights + configuration + metrics saved together
4. **Comprehensive Test Suites**: 8+ automated tests validating component behavior
5. **Documentation**: Complete API reference, workflow guides, and architecture diagrams
6. **Version Control**: Public GitHub repository with tagged releases

---

### Q23: What real-world applications are envisioned?

**Answer:** The paper identifies four primary application domains (Page 139):

1. **Clinical Decision Support**: Energy-efficient on-device medical AI for diagnostics and treatment recommendations

2. **Personalized Adaptive Education**: SCANAQ-aligned tutoring systems that adapt to individual cognitive profiles

3. **Private Edge Computing**: Secure, responsive cognitive assistance without cloud dependency

4. **Accessibility Tools**: Low-power assistive devices for individuals with cognitive or physical disabilities

Each application benefits from the combination of neuromorphic efficiency, conversational capability, and psychometric personalization.

---

## Summary

This Q&A document covers the key questions a dissertation committee may ask about the STAC framework:

| Category | Questions | Key Themes |
|----------|-----------|------------|
| Technical Architecture | Q1-Q6 | AdEx neurons, surrogate gradients, HEMM, TSP, L1 regularization |
| V1 to V2 Methodology | Q7-Q10 | Research pivot rationale, trade-offs, coherence-fidelity balance |
| Energy & Hardware | Q11-Q15 | Theoretical savings, Loihi blockers, validation metrics |
| Limitations & Future | Q16-Q18 | Export blockers, quality recovery, hardware timeline |
| Ethics & Comparison | Q19-Q23 | Privacy, bias, reproducibility, real-world applications |

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Publication DOI:** 10.4018/979-8-3373-5702-7.ch005
