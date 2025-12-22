# STAC: Paper vs Implementation Comparison

This document compares the STAC approach described in the "Aligned Minds, Efficient Machines" paper with the actual implementation in the codebase.

---

## Executive Summary

| Aspect | Paper Claims | Implementation Reality |
|--------|--------------|------------------------|
| Spiking Neurons in Attention | Learnable AdEx neurons with surrogate gradients | **Disabled** - LIF neurons exist but are bypassed |
| Temperature Scaling | T=16 for spike accumulation | **Disabled** - T=1.0 forced in SpikeSoftmax |
| Attention Mechanism | Event-driven spiking computation | Dense QK^T softmax (standard transformer) |
| HEMM Memory Module | Hyperdimensional memory with random projections | **Not implemented** in V2 |
| Energy Efficiency | 3-4x savings claimed | **Theoretical only** - no hardware validation |
| Multi-turn Conversations | TSP enables stateful interactions | Implemented but with KV cache bugs (now fixed) |

---

## Detailed Comparison

### 1. Spiking Neuron Architecture

#### Paper Description (STAC V1)
- **Learnable AdEx Neurons** (Adaptive-Exponential Integrate-and-Fire)
- More biologically plausible than simple LIF
- Captures spike-frequency adaptation and bursting behaviors
- Parameters (τm, τw) made learnable via `torch.nn.Parameter`
- Surrogate gradient technique for backpropagation through spikes

#### Current Implementation
```python
# SpikeAttention lines 165-167 - NEURONS ARE DISABLED
q_spikes = q  # self.q_spk(q)  <- Commented out!
k_spikes = k  # self.k_spk(k)  <- Commented out!
v_spikes = v  # self.v_spk(v)  <- Commented out!
```

**Reality**:
- Uses simple LIF neurons (not AdEx)
- Neurons are **completely bypassed** in forward pass
- No actual spiking computation occurs in attention
- This is a pragmatic choice to preserve generation quality

---

### 2. SpikeSoftmax Temperature Scaling

#### Paper Description
- Accumulate spikes over T=16 timesteps
- Temperature scaling enables spike rate coding
- Theoretical foundation for event-driven computation

#### Current Implementation
```python
# SpikeSoftmax - Temperature scaling DISABLED
def forward(self, x):
    # Was: return torch.softmax(x / self.T, dim=self.dim)  # T=16
    return torch.softmax(x, dim=self.dim)  # T=1.0 effectively
```

**Reality**:
- Temperature scaling caused **near-uniform attention distributions**
- Led to degenerate output (repeated commas: ",,,,,,,,")
- Fix: Disabled temperature, using standard softmax
- This violates the spike rate coding principle from the paper

---

### 3. Hyperdimensional Memory Module (HEMM)

#### Paper Description (STAC V1)
- Custom memory system departing from standard attention
- Process:
  1. Pool spike train via `torch.mean`
  2. Project into high-dimensional space (random projection matrix)
  3. Create sparse, distributed representation
  4. MLP generates "memory bias" added to token representations
- Inspired by hyperdimensional computing literature

#### Current Implementation
**Not implemented in V2**

The codebase has no HEMM module. Memory is handled solely through:
- Standard transformer KV caching
- TemporalSpikeProcessor's `_token_cache_input_ids`
- Batch-specific `batch_kv_caches` dictionary

---

### 4. L1 Spike Regularization

#### Paper Description
- Energy efficiency built into training loss
- Formula: `L_total = L_cross_entropy + λ * torch.mean(torch.abs(spike_tensor))`
- Promotes temporal sparsity during optimization

#### Current Implementation
**Not implemented**

No spike regularization loss in the codebase. The implementation focuses on:
- Post-hoc conversion of pretrained models
- No fine-tuning with spike-aware objectives
- Sparsity only achieved through magnitude pruning (optional Loihi mode)

---

### 5. Attention Mechanism

#### Paper Description
- Event-driven spiking computation
- Spike-based Q, K, V representations
- Neuromorphic-native attention patterns

#### Current Implementation

| Mode | What Paper Says | What Code Does |
|------|-----------------|----------------|
| Standard | Spiking attention | Dense QK^T + standard softmax |
| Loihi | Event-driven | LoihiCausalContextMixer (recurrent EMA) |

```python
# Standard mode uses full dense attention
attn_weights = torch.matmul(q_spikes, k_spikes.transpose(-1, -2)) / (self.head_dim ** 0.5)
# ... followed by standard softmax
attn_probs = self.spike_softmax(attn_weights)  # Just torch.softmax()
```

**Reality**: The "spiking" attention is standard transformer attention with:
- Spiking neurons disabled
- Temperature scaling disabled
- No event-driven computation

---

### 6. Loihi Hardware Compatibility

#### Paper Description
- Deployment on Intel Loihi neuromorphic hardware
- 3-4x energy savings
- Event-driven, sparse computation

#### Current Implementation

**loihi_constraints.py** identifies blockers:
```python
# Dense attention is HARD_BLOCK for Loihi export
if "attention" in name.lower() or "attn" in name.lower():
    issues.append({
        "type": "HARD_BLOCK",
        "module": name,
        "reason": "Dense attention (softmax over full context) not directly mappable"
    })
```

**Reality**:
- Standard SpikeAttention cannot export to Loihi
- LoihiCausalContextMixer is a research approximation
- No actual hardware validation has been performed
- Energy savings are **theoretical projections only**

---

### 7. Multi-Turn Conversation Support

#### Paper Description (STAC V2)
- Temporal Spike Processor (TSP) enables stateful interactions
- Addresses single-turn limitation of V1
- Preserves conversational context

#### Current Implementation
**Implemented with bugs (now fixed)**

The TSP exists and works, but had critical issues:
1. **KV Cache Contamination**: State leaked between forward passes
2. **Weight Transpose Bug**: Conv1D->Linear weights copied incorrectly
3. **Temperature Scaling Bug**: Caused uniform attention

After fixes, TSP achieves **100% parity** with ANN on fidelity tests.

---

### 8. Conversion Methodology

#### Paper Description
- Uses SpikingJelly library's official Converter
- Systematic layer-by-layer conversion
- Surrogate gradient support

#### Current Implementation
```python
def simplified_conversion(model, timesteps=32, skip_gelu_replacement=False):
    """Perform simplified conversion WITHOUT relying on SpikingJelly."""
```

**Reality**:
- Manual, pragmatic layer replacements
- SpikingJelly used only for LIF neurons (which are disabled anyway)
- No official Converter class usage
- Selective component replacement for quality preservation

---

## Key Divergences Summary

### What the Paper Promised vs What Exists

| Feature | Paper | Code | Status |
|---------|-------|------|--------|
| AdEx Neurons | Learnable, biologically plausible | Simple LIF, disabled | Not implemented |
| HEMM | Hyperdimensional memory | None | Not implemented |
| Spike Regularization | L1 loss term | None | Not implemented |
| Event-Driven Attention | Spiking Q/K/V | Dense matmul | Disabled |
| Temperature Scaling | T=16 accumulation | T=1.0 (standard) | Disabled |
| Hardware Validation | Loihi deployment | Software only | Pending |
| Energy Savings | 3-4x claimed | Unknown | Unvalidated |

### Why These Divergences Exist

1. **Pragmatic Quality Preservation**: Enabling spiking components degraded text generation to unusable levels (repeated tokens, gibberish)

2. **V1 -> V2 Pivot**: Paper describes V1's sophisticated approach, but V2 pivoted to "pragmatic conversion" due to V1's computational cost

3. **Research Prototype Status**: Paper explicitly acknowledges V2 is experimental with "known limitations"

4. **Hardware Gap**: No Loihi hardware available for validation, so theoretical claims remain unproven

---

## Recommendations for Alignment

### If the goal is to align implementation with paper claims:

1. **Re-enable spiking neurons** with proper fine-tuning to recover quality
2. **Implement HEMM** for hyperdimensional memory
3. **Add L1 spike regularization** to training pipeline
4. **Validate on Loihi hardware** to prove energy claims
5. **Implement AdEx neurons** instead of simple LIF
6. **Use SpikingJelly Converter** officially instead of manual replacement

### If the goal is practical deployment:

1. **Document the divergences** clearly (this comparison)
2. **Focus on quality** with current pragmatic approach
3. **Add fine-tuning pipeline** for ReLU-converted models
4. **Benchmark software energy** as proxy for hardware

---

## References

- Paper: "Aligned Minds, Efficient Machines: Integrating Neuromorphic Computing for Personalized AI"
- Implementation: `smollm2_converter.py`, `test_conversational_snn.py`, `loihi_constraints.py`
- Related docs: `docs/snn_conversion_fixes.md`
