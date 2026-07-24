#!/usr/bin/env python3
"""
STAC: Spiking Transformer for Conversational AI
Copyright (C) 2024 STAC Authors

Licensed under the MIT License. See LICENSE file for details.

SmolLM2 Converter: Convert SmolLM2-1.7B-Instruct to a Spiking Neural Network
Specialized script for creating a conversational spiking language model.

# NOTE: -------------------------------------------------------------------
# This specialized SmolLM2 conversion pipeline is a **work in progress**.
# While the TemporalSpikeProcessor enables multi-turn state retention in
# software, true hardware-level validation (e.g., Intel Loihi-2) is still
# pending.  Expect API changes and incomplete operator coverage.
# ---------------------------------------------------------------------------
"""
import argparse
import inspect
import torch
import torch.nn as nn
import os
import json
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Tuple, Optional, Union

# Import and check SpikingJelly version first
import importlib.metadata
from packaging.version import parse
import spikingjelly
min_version = '0.0.0.0.14'
try:
    sj_version = importlib.metadata.version("spikingjelly")
except importlib.metadata.PackageNotFoundError:
    error_msg = (
        f"SpikingJelly not found. Version >= {min_version} is required. "
        f"Please install/upgrade SpikingJelly: pip install spikingjelly[cuda] -U --pre"
    )
    logging.error(error_msg)
    raise ImportError(error_msg)

# Use a proper version comparison (string comparison is lexicographic and wrong here).
if parse(sj_version) < parse(min_version):
    error_msg = (
        f"SpikingJelly version {sj_version} is older than required {min_version}. "
        f"Please upgrade SpikingJelly: pip install spikingjelly[cuda] -U --pre"
    )
    logging.error(error_msg)
    raise ImportError(error_msg)
logging.info(f"Using SpikingJelly version: {sj_version}")

# Direct imports from SpikingJelly. Only `functional` is used directly here; the
# neuron/surrogate/converter/quantizer components come through the compatibility layer.
from spikingjelly.activation_based import functional
# Cannot directly import Quantizer - using compatibility layer
from spikingjelly_compat import get_neuron, get_converter, get_quantizer, get_surrogate



# Get components from compatibility layer
LIFNode = get_neuron()
SurrogateModule = get_surrogate()
Converter = get_converter()
Quantizer = get_quantizer()

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('snn_conversion.log')
        ]
    )
logger = logging.getLogger("smollm2_converter")

# Spike-compatible layer normalization
class SpikeLayerNorm(nn.Module):
    """Spiking-compatible layer normalization."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        # Match nn.LayerNorm: normalize by sqrt(var + eps), not (std + eps).
        # Adding eps to the standard deviation changes the scale and diverges from
        # the reference LayerNorm the weights were trained with.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

# Spike-compatible softmax
class SpikeSoftmax(nn.Module):
    """Spiking-compatible softmax implementation using spike rates.

    Note: Temperature scaling is disabled (T=1.0) to preserve sharp attention
    distributions. The original T-based scaling caused near-uniform attention,
    leading to degenerate text generation (repeated commas/tokens).
    """
    def __init__(self, T=16, dim=-1):
        super().__init__()
        # Store T for compatibility but use T=1.0 for actual softmax
        # to preserve attention sharpness and text generation quality
        self._T_stored = T
        self.dim = dim

    def forward(self, x):
        # Use standard softmax without temperature scaling
        # Temperature scaling by T caused near-uniform attention distributions
        return torch.softmax(x, dim=self.dim)

class SpikeAttention(nn.Module):
    """Spiking-compatible self-attention implementation."""
    def __init__(self, embed_dim, num_heads, T=16, causal=True, layer_idx=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.T = T
        self.causal = causal
        # Controls the return-tuple layout expected by the host transformer block.
        # "gpt2": (output, present[, attn]); "llama": (output, attn, present).
        self.return_mode = "gpt2"
        # Required to update a transformers Cache object (Llama-style decoders index the
        # cache per layer). Set when replacing an attention module that carries one.
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        # Re-enable spiking dynamics on projected Q / K / V
        # Using lower thresholds to make neurons more sensitive and generate more spikes
        self.q_spk = LIFNode(v_threshold=0.1, v_reset=0.0, detach_reset=True)
        self.k_spk = LIFNode(v_threshold=0.1, v_reset=0.0, detach_reset=True)
        self.v_spk = LIFNode(v_threshold=0.1, v_reset=0.0, detach_reset=True)
        
        self.spike_softmax = SpikeSoftmax(T=T, dim=-1)
    
    def forward(self, hidden_states, attention_mask=None, layer_past=None,
               head_mask=None, use_cache=False, output_attentions=False,
               past_key_value=None, **kwargs):
        batch_size, seq_length = hidden_states.shape[:2]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # GPT-2 blocks pass the cache as `layer_past`; Llama-style decoders pass it as
        # `past_key_value`, and it may be a transformers Cache object rather than a
        # (k, v) tuple. Reading only `layer_past` meant the cache branch never ran on
        # Llama/SmolLM2, and returning a bare tuple made LlamaModel crash later with
        # "'tuple' object has no attribute 'to_legacy_cache'".
        cache_obj = past_key_value if past_key_value is not None else layer_past
        present = None

        if cache_obj is not None and hasattr(cache_obj, "update"):
            # transformers Cache: it stores the new keys/values and returns the full history.
            k, v = cache_obj.update(k, v, self.layer_idx)
            present = cache_obj if use_cache else None
        else:
            if cache_obj is not None:
                past_key, past_value = cache_obj[0], cache_obj[1]
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            present = (k, v) if use_cache else None
        
        # Reset neuron states to handle dynamic input shapes
        functional.reset_net(self.q_spk)
        functional.reset_net(self.k_spk) 
        functional.reset_net(self.v_spk)
        
        # For now, skip spiking neurons in attention to preserve text generation quality
        # Pass Q and K through spiking neurons (disabled for better generation)
        q_spikes = q  # self.q_spk(q)
        k_spikes = k  # self.k_spk(k)
        v_spikes = v  # self.v_spk(v)
        
        attn_weights = torch.matmul(q_spikes, k_spikes.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Always enforce causal masking when this is a causal attention layer.
        # Previously the causal mask was skipped whenever an attention_mask was supplied
        # (which is almost always the case during generation), silently disabling
        # autoregressive masking and letting each position attend to future tokens.
        if self.causal:
            kv_len = k.size(-2)
            # Number of cached (past) key positions preceding the current query block.
            past_length = kv_len - seq_length
            causal_mask = torch.triu(
                torch.ones(seq_length, kv_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=past_length + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, -10000.0)

        if attention_mask is not None:
            # A mask reaching this point is one of two kinds:
            #   * multiplicative / "keep" mask: 1 = attend, 0 = ignore  (HF's 2D input mask)
            #   * additive mask: 0 = attend, large negative = ignore    (what HF hands to
            #     attention modules, e.g. a [B, 1, tgt, src] float mask with -3.4e38)
            # Distinguish them by sign, NOT by `max() <= 1`: an additive mask also has
            # max 0, so that test re-inverted it into (1 - (-3.4e38)) * -1e4 = -inf on
            # every masked position. Whole rows became -inf and softmax returned NaN, so
            # any batch containing padding produced NaN logits.
            is_keep_mask = bool(attention_mask.min() >= 0)

            if attention_mask.dim() == 2:
                # [B, src] -> [B, 1, 1, src]
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # [B, tgt, src] -> [B, 1, tgt, src]
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 4:
                extended_attention_mask = attention_mask
            else:
                logger.warning(f"Unexpected attention_mask shape: {attention_mask.shape}")
                extended_attention_mask = attention_mask

            if is_keep_mask:
                extended_attention_mask = (1.0 - extended_attention_mask.to(attn_weights.dtype)) * -10000.0
            else:
                extended_attention_mask = extended_attention_mask.to(attn_weights.dtype)

            # Guard against a mask whose key axis disagrees with the score matrix (e.g. a
            # mask covering only the new tokens while attending over a KV cache).
            if extended_attention_mask.size(-1) != attn_weights.size(-1):
                logger.warning(
                    f"attention_mask key length {extended_attention_mask.size(-1)} != "
                    f"score length {attn_weights.size(-1)}; ignoring the mask for this step."
                )
            else:
                attn_weights = attn_weights + extended_attention_mask
        
        attn_probs = self.spike_softmax(attn_weights)
        
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        
        context = torch.matmul(attn_probs, v_spikes)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.o_proj(context)

        # Always return a tuple so downstream transformer blocks can unpack consistently.
        # A bare tensor return broke callers that do `attn_output = attn_outputs[0]`.
        attn_weights_out = attn_probs if output_attentions else None
        if getattr(self, "return_mode", "gpt2") == "llama":
            # LlamaDecoderLayer unpacks (hidden_states, self_attn_weights, present_key_value).
            return output, attn_weights_out, present
        # GPT-2 style block: (hidden_states, present[, attentions]).
        if output_attentions:
            return output, present, attn_probs
        return output, present


class LoihiCausalContextMixer(nn.Module):
    """
    Loihi-oriented attention replacement.

    Goal: remove dense QK^T / softmax attention (a hard Loihi export blocker) and replace it with
    a causal, recurrent context mechanism that is closer to event-driven primitives.

    This is a research approximation intended for export-readiness checks, not a drop-in quality match.
    """

    def __init__(self, hidden_size: int, *, context_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = int(context_size) if context_size is not None else hidden_size

        self.ctx_in = nn.Linear(hidden_size, self.context_size, bias=True)
        self.ctx_out = nn.Linear(self.context_size, hidden_size, bias=True)

        # alpha in (0,1); higher alpha = longer memory. Initialize near slow decay.
        self._logit_alpha = nn.Parameter(torch.tensor(2.0))

        self.mix = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        alpha = torch.sigmoid(self._logit_alpha)

        c = torch.zeros(batch_size, self.context_size, device=hidden_states.device, dtype=hidden_states.dtype)
        outs = []

        for t in range(seq_len):
            x_t = hidden_states[:, t, :]
            u_t = torch.tanh(self.ctx_in(x_t))
            c = alpha * c + (1.0 - alpha) * u_t
            ctx_t = self.ctx_out(c)
            y_t = self.mix(torch.cat([x_t, ctx_t], dim=-1))
            outs.append(y_t.unsqueeze(1))

        output = torch.cat(outs, dim=1)
        present = layer_past if use_cache else None

        # Always return a tuple; the host GPT-2 block does `attn_output = attn_outputs[0]`
        # and `outputs = attn_outputs[1:]`, which breaks on a bare tensor return.
        # LlamaDecoderLayer instead unpacks exactly three values
        # (hidden_states, self_attn_weights, present_key_value), so a 2-tuple raised
        # "ValueError: not enough values to unpack (expected 3, got 2)" on SmolLM2.
        if getattr(self, "return_mode", "gpt2") == "llama":
            return output, None, present
        if output_attentions:
            return output, present, None
        return output, present


def _fake_int8_quantize_tensor(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-tensor symmetric fake int8 quantization.
    Returns (qweight_int8, scale_float32) where dequant = qweight.float() * scale.
    """
    with torch.no_grad():
        w_fp32 = w.detach().to(dtype=torch.float32)
        max_abs = float(w_fp32.abs().max().item()) if w_fp32.numel() > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        q = torch.clamp(torch.round(w_fp32 / scale), -127, 127).to(torch.int8)
        return q, torch.tensor(scale, dtype=torch.float32, device=w.device)


class QuantizedLinearLike(nn.Module):
    """
    Fake-quantized linear-like module.
    - Stores int8 weights + float scale as buffers.
    - Computes in float at runtime (simulation), but demonstrates representability.
    """

    def __init__(self, *, qweight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor], transpose_weight: bool):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.transpose_weight = bool(transpose_weight)
        if bias is not None:
            self.register_buffer("bias", bias.detach())
        else:
            self.bias = None
        self._stac_fake_quant_bits = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
        if self.transpose_weight:
            y = torch.matmul(x, w.transpose(-1, -2))
        else:
            y = torch.matmul(x, w)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y


class QuantizedEmbedding(nn.Module):
    """Fake-quantized embedding table."""

    def __init__(self, *, qweight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self._stac_fake_quant_bits = 8

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Preserve the scale's dtype so a quantized fp16/bf16 model stays in its dtype
        # instead of being silently forced to float32 (which breaks dtype-matched matmuls).
        dtype = self.scale.dtype
        w = self.qweight.to(dtype=dtype) * self.scale.to(dtype=dtype)
        out = torch.index_select(w, 0, input_ids.view(-1)).view(*input_ids.shape, -1)
        return out


class HashedEmbedding(nn.Module):
    """
    Token embedding compression via hashing/bucketing.

    This reduces the embedding table size from vocab_size x hidden to num_buckets x hidden.
    Token IDs are mapped to buckets by modulo (deterministic, cheap).
    """

    def __init__(self, weight: torch.Tensor, num_buckets: int):
        super().__init__()
        if num_buckets <= 0:
            raise ValueError("num_buckets must be > 0")
        self.num_buckets = int(num_buckets)
        self.hidden_size = int(weight.shape[1])
        # store bucket weights as a Parameter to participate in pruning if desired
        self.weight = nn.Parameter(weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bucket_ids = torch.remainder(input_ids, self.num_buckets)
        out = torch.index_select(self.weight, 0, bucket_ids.view(-1)).view(*bucket_ids.shape, -1)
        return out


class QuantizedHashedEmbedding(nn.Module):
    """Fake-quantized hashed embedding (int8 + scale)."""

    def __init__(self, *, qweight: torch.Tensor, scale: torch.Tensor, num_buckets: int):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.num_buckets = int(num_buckets)
        self._stac_fake_quant_bits = 8

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bucket_ids = torch.remainder(input_ids, self.num_buckets)
        w = self.qweight.to(dtype=torch.float32) * self.scale.to(dtype=torch.float32)
        out = torch.index_select(w, 0, bucket_ids.view(-1)).view(*bucket_ids.shape, -1)
        return out.to(dtype=torch.float32)


def apply_loihi_embedding_bucketing(model: nn.Module, *, num_buckets: int) -> nn.Module:
    """
    Replace token embedding table (GPT-2 wte) with bucketed/hashed embedding weights.
    Initializes bucket weights by averaging original embeddings that map to each bucket.
    """
    if num_buckets <= 0:
        return model

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Embedding):
            continue
        # Heuristic: only apply to token embeddings, not positional embeddings
        if not (name.endswith("wte") or ".wte" in name):
            continue

        w = module.weight.detach()
        vocab, hidden = w.shape
        nb = int(num_buckets)
        device = w.device

        with torch.no_grad():
            bucket_w = torch.zeros((nb, hidden), device=device, dtype=w.dtype)
            counts = torch.zeros((nb,), device=device, dtype=torch.float32)
            ids = torch.arange(vocab, device=device)
            buckets = torch.remainder(ids, nb)
            bucket_w.index_add_(0, buckets, w)
            counts.index_add_(0, buckets, torch.ones_like(ids, dtype=torch.float32))
            counts = torch.clamp(counts, min=1.0).unsqueeze(1)
            bucket_w = bucket_w / counts

        # Replace module
        path = name.split(".")
        parent = model
        for attr in path[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, path[-1], HashedEmbedding(bucket_w, nb))
        replaced += 1
        logger.info(f"Replaced token embedding {name} with HashedEmbedding(num_buckets={nb})")

    logger.info(f"Loihi embedding bucketing: replaced {replaced} embedding modules")
    return model


def apply_fake_int8_quantization_for_loihi(model: nn.Module) -> nn.Module:
    """
    Replace common heavy weight modules with fake-quantized wrappers:
    - nn.Linear
    - transformers Conv1D (class name 'Conv1D') and similar linear-like modules with weight+bias
    - nn.Embedding
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # find parent
        path = name.split(".")
        parent_path = ".".join(path[:-1])
        child = path[-1]
        parent = model
        if not name:
            continue
        for attr in path[:-1]:
            parent = getattr(parent, attr)

        cn = module.__class__.__name__

        # Embedding (standard)
        if isinstance(module, nn.Embedding):
            q, s = _fake_int8_quantize_tensor(module.weight)
            setattr(parent, child, QuantizedEmbedding(qweight=q.to(module.weight.device), scale=s.to(module.weight.device)))
            replaced += 1
            continue

        # Hashed embedding
        if isinstance(module, HashedEmbedding):
            q, s = _fake_int8_quantize_tensor(module.weight)
            setattr(
                parent,
                child,
                QuantizedHashedEmbedding(
                    qweight=q.to(module.weight.device),
                    scale=s.to(module.weight.device),
                    num_buckets=module.num_buckets,
                ),
            )
            replaced += 1
            continue

        # Linear
        if isinstance(module, nn.Linear):
            q, s = _fake_int8_quantize_tensor(module.weight)
            bias = module.bias if module.bias is not None else None
            setattr(
                parent,
                child,
                QuantizedLinearLike(
                    qweight=q.to(module.weight.device),
                    scale=s.to(module.weight.device),
                    bias=bias.detach() if bias is not None else None,
                    transpose_weight=True,
                ),
            )
            replaced += 1
            continue

        # Conv1D-like (Transformers) or other linear-like modules: has weight, optional bias, and forward uses x @ weight (+ bias)
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor) and cn == "Conv1D":
            w = getattr(module, "weight")
            b = getattr(module, "bias", None)
            q, s = _fake_int8_quantize_tensor(w)
            setattr(
                parent,
                child,
                QuantizedLinearLike(
                    qweight=q.to(w.device),
                    scale=s.to(w.device),
                    bias=b.detach() if isinstance(b, torch.Tensor) else None,
                    transpose_weight=False,
                ),
            )
            replaced += 1
            continue

    logger.info(f"Loihi fake-quant pass: replaced {replaced} modules with fake int8 wrappers")
    return model


def apply_magnitude_pruning_for_loihi(
    model: nn.Module,
    *,
    target_sparsity: float = 0.5,
    name_substrings: Tuple[str, ...] = ("mlp", "ctx_in", "ctx_out", "mix", "c_fc", "c_proj"),
) -> nn.Module:
    """
    Simple magnitude pruning to introduce sparsity for Loihi-readiness signals.

    - Prunes weights (not biases) in nn.Linear and Conv1D modules whose names contain any of name_substrings.
    - Sets smallest-magnitude weights to zero to reach target_sparsity.
    """
    target_sparsity = float(target_sparsity)
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"target_sparsity must be in [0,1). Got {target_sparsity}")

    pruned_tensors = 0
    total_pruned = 0
    total_numel = 0

    for name, module in model.named_modules():
        if not any(s in name for s in name_substrings):
            continue

        cn = module.__class__.__name__
        weight = None

        if isinstance(module, nn.Linear):
            weight = module.weight
        elif cn == "Conv1D" and hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
            weight = getattr(module, "weight")
        else:
            continue

        if weight is None or weight.numel() == 0:
            continue

        with torch.no_grad():
            w = weight.detach()
            total_numel += int(w.numel())
            k = int(target_sparsity * w.numel())
            if k <= 0:
                continue

            flat = w.abs().view(-1)
            # threshold at kth smallest magnitude
            thresh, _ = torch.kthvalue(flat, k)
            mask = (w.abs() > thresh).to(w.dtype)
            # Ensure exact sparsity isn't critical; this is heuristic
            new_w = w * mask
            pruned_now = int((w.numel() - torch.count_nonzero(new_w)).item())
            total_pruned += pruned_now
            pruned_tensors += 1

            if isinstance(module, nn.Linear):
                module.weight.copy_(new_w)
            else:
                # Transformers Conv1D uses nn.Parameter for weight; keep it a Parameter and update data in-place.
                if isinstance(module.weight, torch.nn.Parameter):
                    module.weight.data.copy_(new_w)
                else:
                    module.weight = new_w

    setattr(model, "_stac_loihi_pruned", True)
    setattr(model, "_stac_loihi_prune_target", target_sparsity)
    logger.info(f"Loihi prune pass: pruned {total_pruned}/{total_numel} weights across {pruned_tensors} tensors (target={target_sparsity:.2f})")
    return model

# SNN Temporal Container for Autoregressive Processing
class TemporalSpikeProcessor(nn.Module):
    """Processes input through SNN model over multiple timesteps."""
    def __init__(self, snn_model, T=16, max_context_length=512):
        super().__init__()
        # Store the model directly - no need for Converter here since
        # simplified_conversion already does the layer replacements
        self.snn_model = snn_model
        # Expose the inner model's config so metadata helpers (e.g. save_snn_model)
        # and HF-style callers can read `.config` off the wrapper.
        self.config = getattr(snn_model, 'config', None)
        self.T = T
        self.kv_cache = None
        self.max_context_length = max_context_length
        # NOTE: `device` is a *live* property, not a snapshot. Caching it here meant a
        # later `.to('cuda')` left it pointing at CPU, and every tensor built from it
        # (position ids, empty KV placeholders) landed on the wrong device.
        # Initialize dictionary to store batch-specific KV caches
        self.batch_kv_caches = {}
        # Placeholder for last computed position IDs (for testing)
        self._last_position_ids = None
        # Optional token-cache for incremental (turn-by-turn) usage
        self._token_cache_input_ids = None
        self._token_cache_attention_mask = None
        # Learnable scalar to align student logit magnitudes with ANN teacher during distillation
        self.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        # logger.info(f"Created temporal spike processor with T={T}, max_context_length={max_context_length}, device={self.device}")
    
    @property
    def device(self):
        """Device the inner model currently lives on."""
        for param in self.snn_model.parameters():
            return param.device
        for buf in self.snn_model.buffers():
            return buf.device
        return torch.device("cpu")

    def _max_position_embeddings(self):
        """Positional capacity of the inner model, or None when it has no limit."""
        config = getattr(self.snn_model, 'config', None)
        if config is None:
            return None
        max_pos = getattr(config, 'max_position_embeddings', None)
        if max_pos is None:
            max_pos = getattr(config, 'n_positions', None)
        return int(max_pos) if max_pos else None

    def _context_limit(self):
        """
        Largest context this processor may feed the inner model.

        Bounded by BOTH the configured max_context_length and the model's positional
        capacity: exceeding the latter makes the position embedding lookup raise
        `IndexError: index out of range in self`.
        """
        limit = int(self.max_context_length)
        max_pos = self._max_position_embeddings()
        if max_pos:
            limit = min(limit, max_pos)
        return max(1, limit)

    @staticmethod
    def _cache_length(past_key_values):
        """Number of cached positions, or 0 when there is no usable cache."""
        try:
            return int(past_key_values[0][0].size(-2))
        except Exception:
            return 0

    @staticmethod
    def _trim_cache(past_key_values, keep):
        """
        Drop the oldest entries so at most `keep` cached positions remain.

        Trimming from the left keeps the cache aligned with a left-truncated context
        window and stops `past_length` from growing past the model's position limit.
        """
        if keep <= 0:
            return None
        trimmed = []
        for layer in past_key_values:
            key, value = layer[0], layer[1]
            trimmed.append((key[..., -keep:, :], value[..., -keep:, :]))
        return trimmed

    def _create_position_ids(self, input_shape, past_length=0):
        """
        HF-style position ID creation with cache support.
        Aligns with HuggingFace's create_position_ids_from_input_ids method.
        """
        batch_size, seq_length = input_shape

        # Create position IDs that continue from past_length
        position_ids = torch.arange(
            past_length,
            past_length + seq_length,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # Apply clamping with fallback for models using relative position embeddings
        max_pos = self._max_position_embeddings() or 32768
        position_ids = position_ids.clamp(0, max_pos-1)

        # Expand to match batch size
        return position_ids.expand(batch_size, -1)

    def _accepts_position_ids(self):
        """Whether the inner model's forward takes an explicit `position_ids` kwarg."""
        cached = getattr(self, "_position_ids_supported", None)
        if cached is None:
            try:
                params = inspect.signature(self.snn_model.forward).parameters
                cached = "position_ids" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                cached = False
            self._position_ids_supported = bool(cached)
        return self._position_ids_supported


    def forward(self, input_ids, attention_mask=None, use_cache=True, batch_ids=None, incremental=False, **kwargs):
        """
        Process input through the SNN model using temporal processing with batch support.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_length]
            attention_mask: Optional attention mask
            use_cache: Whether to use and update KV cache for efficient conversation
            batch_ids: Optional list/tensor of unique conversation IDs for multi-conversation batching
            incremental: If True, treat input_ids as *new tokens only* and append to an internal token cache
            
        Returns:
            Tensor with accumulated logits
        """
        batch_size, seq_length = input_ids.shape
        # Length the caller asked about. `seq_length` is rebound below when the context
        # is truncated, so keep the original for restoring the output shape at the end.
        original_seq_length = seq_length

        # Optional incremental token cache behavior for multi-turn state retention
        if incremental:
            if self._token_cache_input_ids is None:
                self._token_cache_input_ids = input_ids
                if attention_mask is None:
                    self._token_cache_attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                else:
                    self._token_cache_attention_mask = attention_mask
            else:
                self._token_cache_input_ids = torch.cat([self._token_cache_input_ids, input_ids], dim=1)
                if self._token_cache_attention_mask is None:
                    self._token_cache_attention_mask = torch.ones_like(self._token_cache_input_ids, dtype=torch.long)
                else:
                    add_mask = torch.ones_like(input_ids, dtype=torch.long) if attention_mask is None else attention_mask
                    self._token_cache_attention_mask = torch.cat([self._token_cache_attention_mask, add_mask], dim=1)

            # Enforce max context length on the cache
            token_cache_limit = self._context_limit()
            if self._token_cache_input_ids.size(1) > token_cache_limit:
                self._token_cache_input_ids = self._token_cache_input_ids[:, -token_cache_limit:]
                if self._token_cache_attention_mask is not None:
                    self._token_cache_attention_mask = self._token_cache_attention_mask[:, -token_cache_limit:]

            # Use cached full context for the actual forward pass
            input_ids = self._token_cache_input_ids
            attention_mask = self._token_cache_attention_mask
            batch_size, seq_length = input_ids.shape
        
        # Ensure input doesn't exceed the usable context. The bound is the smaller of the
        # configured max_context_length and the model's positional capacity — feeding
        # positions beyond the latter raises IndexError inside the embedding lookup.
        context_limit = self._context_limit()
        if seq_length > context_limit:
            logger.warning(f"Input sequence length {seq_length} exceeds max context length {context_limit}. Truncating.")
            input_ids = input_ids[:, -context_limit:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -context_limit:]
            batch_size, seq_length = input_ids.shape
        
        # In Loihi mode, avoid HuggingFace KV-cache semantics; rely on token cache / sequential processing instead.
        # Some attention replacements (e.g., LoihiCausalContextMixer) do not produce valid past_key_values and can
        # poison the cache with None entries.
        if bool(getattr(self.snn_model, "_stac_loihi_mode", False)):
            use_cache = False

        # Handle batch-specific KV caches if batch_ids provided
        if batch_ids is not None:
            # Convert batch_ids to list if it's a tensor
            if isinstance(batch_ids, torch.Tensor):
                batch_ids = batch_ids.tolist()
                
            # Initialize past_key_values based on batch_ids
            past_key_values_list = []
            for i, batch_id in enumerate(batch_ids):
                if use_cache and batch_id in self.batch_kv_caches:
                    # Extract single-batch cache for this conversation
                    single_batch_cache = self.batch_kv_caches[batch_id]
                    
                    # Add this conversation's cache to the list
                    past_key_values_list.append(single_batch_cache)
                else:
                    # No cache for this conversation yet
                    past_key_values_list.append(None)
                    
            # Create a combined past_key_values appropriate for batched processing
            past_key_values = []
            # Determine total layers reliably across model types
            total_layers = None
            if past_key_values_list and past_key_values_list[0] is not None:
                total_layers = len(past_key_values_list[0])
            else:
                total_layers = getattr(self.snn_model.config, 'num_hidden_layers', None)
                if total_layers is None:
                    total_layers = getattr(self.snn_model.config, 'n_layer', 0)

            # The per-conversation caches can only be batched when they all cover the same
            # number of positions. Concatenating a zero-length placeholder (a conversation
            # with no cache yet) with a populated one raised "Sizes of tensors must match
            # except in dimension 0" from torch.cat, so any batch mixing a fresh
            # conversation with an ongoing one crashed.
            cache_lengths = {
                (c[0][0].size(-2) if c is not None else 0) for c in past_key_values_list
            }
            if len(cache_lengths) > 1:
                logger.warning(
                    f"Conversations in this batch have different cache lengths {sorted(cache_lengths)}; "
                    "processing them without a KV cache for this step."
                )
                past_key_values = None
            elif cache_lengths == {0}:
                past_key_values = None
            else:
                for layer_idx in range(total_layers):
                    key_layer = []
                    value_layer = []
                    # Collect keys and values for each batch item
                    for batch_idx, batch_cache in enumerate(past_key_values_list):
                        key_layer.append(batch_cache[layer_idx][0])
                        value_layer.append(batch_cache[layer_idx][1])
                    # Stack along batch dimension
                    keys = torch.cat(key_layer, dim=0)
                    values = torch.cat(value_layer, dim=0)
                    past_key_values.append((keys, values))
        else:
            # Standard non-batched processing using global KV cache
            past_key_values = self.kv_cache if use_cache else None

        # Keep the KV cache inside the context window. Truncating input_ids alone left the
        # cache growing by one position per call forever: it silently exceeded
        # max_context_length and then crashed with "index out of range in self" once
        # past_length reached the model's max_position_embeddings. The cache must also stay
        # a strict prefix of the current input, otherwise the model is fed only its last
        # token against an unrelated history.
        # A cache built for a different batch size cannot be reused: concatenating it with
        # the new keys raised a bare "Sizes of tensors must match except in dimension 2"
        # from inside attention. Drop it instead.
        if past_key_values is not None:
            try:
                cache_batch = int(past_key_values[0][0].size(0))
            except Exception:
                cache_batch = batch_size
            if cache_batch != batch_size:
                logger.warning(
                    f"Discarding KV cache built for batch size {cache_batch}; "
                    f"this call has batch size {batch_size}."
                )
                past_key_values = None
                if batch_ids is None:
                    self.kv_cache = None

        if past_key_values is not None:
            cache_len = self._cache_length(past_key_values)
            max_cache = min(context_limit - 1, seq_length - 1)
            if cache_len > max_cache:
                logger.debug(
                    f"Trimming KV cache from {cache_len} to {max(max_cache, 0)} entries "
                    f"(context limit {context_limit}, current input {seq_length})."
                )
                past_key_values = self._trim_cache(past_key_values, max_cache)
                if batch_ids is None:
                    self.kv_cache = past_key_values

        # Reset all neuron states in the model before processing
        functional.reset_net(self.snn_model)
        
        # Process over T timesteps to accumulate spikes
        spike_accum = None
        present_key_values = None
        
        effective_T = max(1, int(self.T))
        for _ in range(effective_T):
            # Allow gradients when caller enables them (needed for distillation / adapter finetune).
            # Do not wrap in torch.no_grad(); the caller controls grad mode.
            model_kwargs = {}

            # -----------------  KV Cache & Position Handling -----------------
            # Validate cache structure to avoid None poisoning
            using_kv_cache = (
                past_key_values is not None
                and isinstance(past_key_values, (list, tuple))
                and len(past_key_values) > 0
                and isinstance(past_key_values[0], (list, tuple))
                and len(past_key_values[0]) >= 1
                and past_key_values[0][0] is not None
            )
            model_input_ids = input_ids  # By default feed full sequence
            past_length = 0
            if using_kv_cache:
                # Only feed the NEW tokens to the model to avoid size mismatch. The cache
                # was trimmed above so it is always a strict prefix of `input_ids`.
                past_length = min(self._cache_length(past_key_values), max(seq_length - 1, 0))
                model_input_ids = input_ids[:, past_length:]
            # -----------------------------------------------------------------

            # Ensure attention_mask is valid
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=input_ids.device)

            # The mask must cover cached + new positions; keep it aligned with the window
            # actually being attended over.
            expected_mask_len = past_length + model_input_ids.size(1)
            if attention_mask.dim() == 2 and attention_mask.size(1) != expected_mask_len:
                if attention_mask.size(1) > expected_mask_len:
                    attention_mask = attention_mask[:, -expected_mask_len:]
                else:
                    pad = torch.ones(
                        (attention_mask.size(0), expected_mask_len - attention_mask.size(1)),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([pad, attention_mask], dim=1)

            # Add attention mask to kwargs
            model_kwargs["attention_mask"] = attention_mask

            # Pass explicit, clamped position IDs. Without this the inner model derives
            # them from the cache length alone, with nothing keeping them below
            # max_position_embeddings (_create_position_ids existed but was never called).
            if self._accepts_position_ids():
                model_kwargs["position_ids"] = self._create_position_ids(
                    (model_input_ids.size(0), model_input_ids.size(1)), past_length=past_length
                )

            # Always pass an explicit use_cache. If we leave it unset the inner HF model
            # falls back to config.use_cache (usually True) and builds its own cache
            # object (e.g. a Llama DynamicCache), which is incompatible with the legacy
            # (k, v) tuples SpikeAttention emits and crashes on `.to_legacy_cache()`.
            model_kwargs["use_cache"] = bool(use_cache)
            if use_cache and past_key_values is not None:
                model_kwargs["past_key_values"] = past_key_values

            # Forward pass through the model
            outputs = self.snn_model(model_input_ids, **model_kwargs)

            # Get the logits from output structure
            if hasattr(outputs, "logits"):
                # Standard HF model output
                current_logits = outputs.logits
                if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                    present_key_values = outputs.past_key_values
            else:
                # Tuple output (logits, past_key_values)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    current_logits = outputs[0]
                    present_key_values = outputs[1]
                else:
                    # Direct logits output
                    current_logits = outputs

            if spike_accum is None:
                spike_accum = current_logits
            else:
                spike_accum = spike_accum + current_logits
        
        # Update cache if needed
        if use_cache and present_key_values is not None:
            if batch_ids is not None:
                # Update batch-specific KV caches
                for i, batch_id in enumerate(batch_ids):
                    # Extract and store single-batch cache for this conversation
                    single_batch_cache = []
                    for layer_idx in range(len(present_key_values)):
                        # Extract batch slice from key and value
                        key_slice = present_key_values[layer_idx][0][i:i+1]  # Keep batch dimension
                        value_slice = present_key_values[layer_idx][1][i:i+1]  # Keep batch dimension
                        single_batch_cache.append((key_slice, value_slice))
                    
                    # Store in batch-specific cache
                    self.batch_kv_caches[batch_id] = single_batch_cache
            else:
                # Update global KV cache
                self.kv_cache = present_key_values
        
        # Store last position ids for external inspection (testing utilities)
        try:
            total_seq_len = 0
            if self.kv_cache is not None and len(self.kv_cache) > 0:
                total_seq_len = self.kv_cache[0][0].size(-2)
            else:
                total_seq_len = input_ids.size(1)
            self._last_position_ids = torch.arange(0, total_seq_len, device=self.device).unsqueeze(0)
        except Exception:
            self._last_position_ids = None
        
        # Scale accumulated spikes to restore original logit magnitudes
        # SNN conversion typically reduces magnitudes significantly, so we need strong scaling
        if spike_accum is None:
            vocab = getattr(self.snn_model.config, "vocab_size", None)
            if vocab is None:
                raise RuntimeError("TemporalSpikeProcessor could not produce logits and vocab_size is unknown.")
            final_logits = torch.zeros(
                (batch_size, original_seq_length, vocab),
                device=input_ids.device,
                dtype=torch.float32,
            )
        else:
            final_logits = spike_accum / effective_T  # Normalize accumulated spikes by timestep count

        # Apply learnable logit scaling (helps distillation/parity)
        final_logits = final_logits * self.logit_scale.to(dtype=final_logits.dtype)

        # Ensure logits sequence length matches the ORIGINAL input_ids length so callers
        # comparing shapes do not fail, even if the model shortened the sequence via
        # context truncation or KV-cache reuse. `seq_length` is rebound by truncation, so
        # comparing against it silently returned a shorter tensor than the caller passed.
        if final_logits.shape[1] != original_seq_length:
            if final_logits.shape[1] < original_seq_length:
                # Left-pad with zeros (model ignored some positions)
                pad_len = original_seq_length - final_logits.shape[1]
                pad_tensor = torch.zeros(
                    final_logits.size(0), pad_len, final_logits.size(-1),
                    dtype=final_logits.dtype, device=final_logits.device
                )
                final_logits = torch.cat([pad_tensor, final_logits], dim=1)
            else:
                # Truncate to expected length
                final_logits = final_logits[:, -seq_length:]

        # Build an output object that supports both `.logits` access and Tensor-style indexing used
        # elsewhere in the test suite.
        class _CompatOutput:
            def __init__(self, logits_tensor, pkv):
                self.logits = logits_tensor
                self.past_key_values = pkv
            # Allow `output[0]` or `output[0, -1, :]` to access logits as if it were a tensor
            def __getitem__(self, item):
                return self.logits.__getitem__(item)
            # Make it iterable so tuple(output) works
            def __iter__(self):
                yield self.logits
                yield self.past_key_values
            # For printing
            def __repr__(self):
                return f"_CompatOutput(logits_shape={tuple(self.logits.shape)})"

        return _CompatOutput(final_logits, self.kv_cache if use_cache else None)
    
    def reset_cache(self, batch_id=None):
        """Reset the KV cache (e.g., at the start of a new conversation)
        
        Args:
            batch_id: Optional batch ID to reset only a specific conversation cache
        """
        if batch_id is not None:
            # Drop the specific batch's cache entirely. Zeroing it in-place kept a
            # non-empty (stale past_length) KV tensor, which corrupts the next turn's
            # position/length bookkeeping. Removing the entry means the conversation
            # starts fresh on its next forward pass.
            self.batch_kv_caches.pop(batch_id, None)
        else:
            # Clear global cache entirely
            self.kv_cache = None
            self._token_cache_input_ids = None
            self._token_cache_attention_mask = None
            
            # Also reset all batch-specific caches
            self.batch_kv_caches = {}
        
    def get_position_ids(self):
        """Return the last computed position IDs tensor for validation."""
        if hasattr(self, '_last_position_ids') and self._last_position_ids is not None:
            return self._last_position_ids.clone().detach()
        # Fallback: return zero tensor
        return torch.zeros(1, dtype=torch.long, device=self.device)

    def get_cached_input_length(self) -> int:
        """Return cached context length in tokens (incremental mode), or 0 if none."""
        if self._token_cache_input_ids is None:
            return 0
        return int(self._token_cache_input_ids.size(1))

    # ---- HuggingFace/PEFT compatibility helpers ----
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegate to inner model when available. PEFT expects CausalLM models to implement this.
        """
        if hasattr(self.snn_model, "prepare_inputs_for_generation"):
            return self.snn_model.prepare_inputs_for_generation(*args, **kwargs)
        raise AttributeError("Inner model does not implement prepare_inputs_for_generation")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert SmolLM2 to a Spiking Neural Network')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-1.7B-Instruct',
                        help='The model to convert (default: HuggingFaceTB/SmolLM2-1.7B-Instruct)')
    parser.add_argument('--output_dir', type=str, default='./snn_converted_model',
                        help='Directory to save the converted model')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of calibration samples')
    parser.add_argument('--timesteps', type=int, default=32,
                        help='Number of timesteps for SNN')
    parser.add_argument('--quantize_bits', type=int, default=8,
                        help='Number of bits for quantization')
    parser.add_argument('--simplified', action='store_true',
                        help='Use simplified conversion (no SpikingJelly)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for conversion')
    parser.add_argument('--max_context_length', type=int, default=512,
                        help='Maximum context length for the model')
    parser.add_argument('--calibrate_timesteps', action='store_true',
                        help='After conversion, halve T and rescale LIF thresholds accordingly. '
                             'Off by default so --timesteps is honoured exactly.')
    return parser.parse_args()

def replace_gelu_with_relu(model):
    """Replace GeLU activations with ReLU for SNN compatibility."""
    logger.info("Replacing GeLU activations with ReLU")
    gelu_count = 0
    gelu_new_count = 0

    # Replace any GELU-family activation module by swapping it out on its parent.
    # Iterate over a snapshot so mutating the module tree mid-iteration is safe, and
    # use proper parent-setattr replacement. Reassigning `mod.__class__` in place left
    # a torch.nn.ReLU instance without an `inplace` attribute, which raises
    # AttributeError on the next forward.
    gelu_class_names = {"GELU", "GELUActivation", "NewGELUActivation", "FastGELUActivation", "QuickGELUActivation"}
    for name, mod in list(model.named_modules()):
        if mod.__class__.__name__ not in gelu_class_names:
            continue
        path = name.split('.')
        child_name = path[-1]
        parent_path = '.'.join(path[:-1])
        if parent_path:
            parent = model
            for attr in parent_path.split('.'):
                parent = getattr(parent, attr)
            setattr(parent, child_name, torch.nn.ReLU())
        elif child_name:
            setattr(model, child_name, torch.nn.ReLU())
        else:
            # Model itself is the activation (unusual); nothing to reparent.
            continue
        if mod.__class__.__name__ == "NewGELUActivation":
            gelu_new_count += 1
        else:
            gelu_count += 1

    # Update config if it exists
    if hasattr(model, 'config') and hasattr(model.config, 'activation_function'):
        model.config.activation_function = "relu"
    
    logger.info(f"Replaced {gelu_count} GELU and {gelu_new_count} NewGELUActivation modules with ReLU")
    return model

def create_calibration_data(tokenizer, num_samples=10, max_length=128):
    """Create simple calibration data for SNN conversion."""
    logger.info(f"Creating {num_samples} calibration samples")
    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "The purpose of neural networks is",
        "Quantum computing uses",
        "Machine learning models can",
        "The future of technology looks",
        "Climate change affects",
        "The human brain processes",
        "Space exploration has revealed",
        "Renewable energy sources include"
    ]
    
    # Use available prompts or generate random tokens if more needed
    if num_samples > len(prompts):
        # Extend with random data
        for _ in range(num_samples - len(prompts)):
            random_length = torch.randint(5, 15, (1,)).item()
            random_ids = torch.randint(100, tokenizer.vocab_size, (random_length,))
            random_text = tokenizer.decode(random_ids)
            prompts.append(random_text)
    
    # Tokenize all prompts
    inputs = tokenizer(
        prompts[:num_samples], 
        return_tensors="pt", 
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    # Format as dataloader-compatible list
    calib_data_list = []
    for i in range(len(inputs["input_ids"])):
        sample = {
            "input_ids": inputs["input_ids"][i].unsqueeze(0),
            "attention_mask": inputs["attention_mask"][i].unsqueeze(0)
        }
        calib_data_list.append((sample, None))
    
    return calib_data_list

def replace_layernorm_with_spikelayernorm(model):
    """Replace LayerNorm with spike-compatible SpikeLayerNorm."""
    logger.info("Replacing LayerNorm with spike-compatible SpikeLayerNorm")
    ln_count = 0
    
    # Find and replace layer norms. Snapshot the module list first so replacing
    # submodules mid-iteration cannot disturb the traversal.
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.LayerNorm):
            shape = module.normalized_shape
            new_ln = SpikeLayerNorm(shape, module.eps)
            
            # Copy parameters
            new_ln.weight.data.copy_(module.weight.data)
            new_ln.bias.data.copy_(module.bias.data)
            
            # Find parent module
            path = name.split('.')
            parent_path = '.'.join(path[:-1])
            child_name = path[-1]
            
            if parent_path:
                parent = model
                for attr in parent_path.split('.'):
                    parent = getattr(parent, attr)
                setattr(parent, child_name, new_ln)
            else:
                setattr(model, child_name, new_ln)
            
            ln_count += 1
    
    logger.info(f"Replaced {ln_count} LayerNorm modules with SpikeLayerNorm")
    return model

def replace_attention_with_spikeattention(model):
    """Replace self-attention mechanisms with spike-compatible versions."""
    logger.info("Replacing attention blocks with SpikeAttention")
    attn_count = 0
    
    # Detect model architecture type for appropriate attention handling
    model_type = ""
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type.lower()
        logger.info(f"Detected model type: {model_type}")
    
    # For Llama-style decoder-only architectures (Llama, Mistral, SmolLM2, ...).
    # These expose `model.model.layers[i].self_attn` with separate q/k/v/o_proj Linears.
    if (
        model_type
        and ('llama' in model_type or 'mistral' in model_type or 'smollm' in model_type)
        and hasattr(model, 'model') and hasattr(model.model, 'layers')
    ):
        logger.info(f"Using Llama-style attention handling for {model_type}")
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)

        if num_kv_heads != num_heads:
            # Grouped-query attention: q_proj and k/v_proj have different output dims and
            # SpikeAttention assumes full multi-head (q==kv heads). Refuse honestly rather
            # than silently copying mismatched weights.
            raise NotImplementedError(
                f"Grouped-query attention (num_key_value_heads={num_kv_heads} != "
                f"num_attention_heads={num_heads}) is not supported by SpikeAttention yet."
            )

        logger.warning(
            "Llama-style conversion: SpikeAttention does not apply rotary position "
            "embeddings (RoPE). Positional information from RoPE is therefore lost; "
            "expect reduced fidelity relative to the original model."
        )

        for layer in model.model.layers:
            if not hasattr(layer, 'self_attn'):
                continue
            attn = layer.self_attn
            spike_attn = SpikeAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                T=model.T if hasattr(model, 'T') else 16,
                causal=True,
                # Carry the layer index over so the replacement can update a Cache object.
                layer_idx=getattr(attn, 'layer_idx', None),
            )
            spike_attn.return_mode = "llama"
            try:
                spike_attn.q_proj.weight.data.copy_(attn.q_proj.weight.data)
                spike_attn.k_proj.weight.data.copy_(attn.k_proj.weight.data)
                spike_attn.v_proj.weight.data.copy_(attn.v_proj.weight.data)
                spike_attn.o_proj.weight.data.copy_(attn.o_proj.weight.data)
                # Llama projections are typically bias-free; copy biases only if present.
                for src_name, dst in (
                    ('q_proj', spike_attn.q_proj), ('k_proj', spike_attn.k_proj),
                    ('v_proj', spike_attn.v_proj), ('o_proj', spike_attn.o_proj),
                ):
                    src = getattr(attn, src_name)
                    if getattr(src, 'bias', None) is not None and dst.bias is not None:
                        dst.bias.data.copy_(src.bias.data)
                    elif dst.bias is not None:
                        dst.bias.data.zero_()
            except Exception as e:
                logger.warning(f"Error copying Llama attention weights: {e}. Using default initialization.")
            layer.self_attn = spike_attn
            attn_count += 1

        if attn_count == 0:
            raise NotImplementedError(
                f"Could not find self_attn modules in Llama-style model '{model_type}'."
            )
        logger.info(f"Replaced {attn_count} attention blocks with SpikeAttention")
        return model

    # For GPT and similar decoder-only architectures
    if model_type and ('gpt' in model_type or 'opt' in model_type or 'pythia' in model_type):
        logger.info(f"Using GPT-style attention handling for {model_type}")
        
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT2-style architecture
            hidden_size = model.config.hidden_size
            num_heads = model.config.num_attention_heads
            
            for block in model.transformer.h:
                if hasattr(block, 'attn'):
                    # Create SpikeAttention module
                    spike_attn = SpikeAttention(
                        embed_dim=hidden_size,
                        num_heads=num_heads,
                        T=model.T if hasattr(model, 'T') else 16,
                        causal=True
                    )
                    
                    # Store original weights for initialization
                    orig_weights = {
                        'q_weight': None,
                        'k_weight': None,
                        'v_weight': None,
                        'q_bias': None,
                        'k_bias': None,
                        'v_bias': None,
                        'o_weight': None,
                        'o_bias': None
                    }
                    
                    # Try different GPT-style attention formats
                    try:
                        # Check if it's using a combined QKV projection (c_attn)
                        if hasattr(block.attn, 'c_attn'):
                            # GPT2-style: combined QKV projection
                            qkv_weight = block.attn.c_attn.weight.data
                            qkv_bias = block.attn.c_attn.bias.data
                            
                            # Handle different weight dimensions - GPT2 uses a single matrix for QKV
                            head_dim = hidden_size // num_heads
                            
                            # Check the shape format - GPT-2 uses [hidden_size, 3*hidden_size]
                            if qkv_weight.size(0) == hidden_size and qkv_weight.size(1) == 3 * hidden_size:
                                # GPT-2 style format - Conv1D stores weights as [in, out]
                                # but Linear expects [out, in], so we need to transpose
                                logger.info(f"Detected GPT-2 style QKV format: {qkv_weight.shape}")

                                # Split the weights along dimension 1 - GPT-2 has them as [h, 3h]
                                q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=1)
                                q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)

                                # Copy to new layers - MUST transpose from Conv1D [in, out] to Linear [out, in]
                                spike_attn.q_proj.weight.data.copy_(q_weight.t())
                                spike_attn.k_proj.weight.data.copy_(k_weight.t())
                                spike_attn.v_proj.weight.data.copy_(v_weight.t())

                                spike_attn.q_proj.bias.data.copy_(q_bias)
                                spike_attn.k_proj.bias.data.copy_(k_bias)
                                spike_attn.v_proj.bias.data.copy_(v_bias)
                            
                            # Standard format with [3*hidden_size, hidden_size]
                            elif qkv_weight.size(0) == 3 * hidden_size and qkv_weight.size(1) == hidden_size:
                                logger.info(f"Detected standard QKV format: {qkv_weight.shape}")
                                
                                # Split into separate Q, K, V (along first dimension)
                                q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
                                q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
                                
                                # Copy to new layers
                                spike_attn.q_proj.weight.data.copy_(q_weight)
                                spike_attn.k_proj.weight.data.copy_(k_weight)
                                spike_attn.v_proj.weight.data.copy_(v_weight)
                                
                                spike_attn.q_proj.bias.data.copy_(q_bias)
                                spike_attn.k_proj.bias.data.copy_(k_bias)
                                spike_attn.v_proj.bias.data.copy_(v_bias)
                            else:
                                # For SmolLM2 which may have a different attention structure
                                logger.info(f"SmolLM2 QKV weight shape: {qkv_weight.shape}, attempting to adapt")
                                
                                # Try to infer the format
                                if qkv_weight.dim() == 2:
                                    # See if it's a transposed version or other format
                                    if qkv_weight.size(1) % 3 == 0:
                                        # Conv1D format [hidden_size, something*3] - needs transpose
                                        split_size = qkv_weight.size(1) // 3
                                        q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size, dim=1)

                                        # Try to split bias similarly
                                        if qkv_bias.size(0) % 3 == 0:
                                            bias_split = qkv_bias.size(0) // 3
                                            q_bias, k_bias, v_bias = torch.split(qkv_bias, bias_split, dim=0)
                                        else:
                                            # Just duplicate bias if we can't split
                                            q_bias = k_bias = v_bias = qkv_bias

                                        # Copy to new layers - transpose from Conv1D [in, out] to Linear [out, in]
                                        spike_attn.q_proj.weight.data.copy_(q_weight.t())
                                        spike_attn.k_proj.weight.data.copy_(k_weight.t())
                                        spike_attn.v_proj.weight.data.copy_(v_weight.t())

                                        spike_attn.q_proj.bias.data.copy_(q_bias)
                                        spike_attn.k_proj.bias.data.copy_(k_bias)
                                        spike_attn.v_proj.bias.data.copy_(v_bias)
                                    
                                    elif qkv_weight.size(0) % 3 == 0:
                                        # Probably [something*3, hidden_size]
                                        split_size = qkv_weight.size(0) // 3
                                        q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size, dim=0)
                                        
                                        # Try to split bias similarly
                                        if qkv_bias.size(0) % 3 == 0:
                                            bias_split = qkv_bias.size(0) // 3
                                            q_bias, k_bias, v_bias = torch.split(qkv_bias, bias_split, dim=0)
                                        else:
                                            # Just duplicate bias if we can't split
                                            q_bias = k_bias = v_bias = qkv_bias
                                            
                                        # Copy to new layers
                                        spike_attn.q_proj.weight.data.copy_(q_weight)
                                        spike_attn.k_proj.weight.data.copy_(k_weight)
                                        spike_attn.v_proj.weight.data.copy_(v_weight)
                                        
                                        spike_attn.q_proj.bias.data.copy_(q_bias)
                                        spike_attn.k_proj.bias.data.copy_(k_bias)
                                        spike_attn.v_proj.bias.data.copy_(v_bias)
                                    else:
                                        # Can't determine, use default initialization
                                        logger.warning(f"Couldn't determine QKV split for shape: {qkv_weight.shape}. Using default initialization.")
                                else:
                                    logger.warning(f"Unexpected QKV weight tensor dimension: {qkv_weight.dim()}. Using default initialization.")
                            
                            # Copy output projection if available
                            # c_proj is also Conv1D, so transpose weights from [in, out] to [out, in]
                            if hasattr(block.attn, 'c_proj'):
                                spike_attn.o_proj.weight.data.copy_(block.attn.c_proj.weight.data.t())
                                spike_attn.o_proj.bias.data.copy_(block.attn.c_proj.bias.data)
                        
                        # Check if using separate Q, K, V projections
                        elif hasattr(block.attn, 'q_proj') and hasattr(block.attn, 'k_proj') and hasattr(block.attn, 'v_proj'):
                            # Separate projections like in many modern Transformer models
                            spike_attn.q_proj.weight.data.copy_(block.attn.q_proj.weight.data)
                            spike_attn.k_proj.weight.data.copy_(block.attn.k_proj.weight.data)
                            spike_attn.v_proj.weight.data.copy_(block.attn.v_proj.weight.data)
                            
                            if hasattr(block.attn.q_proj, 'bias') and block.attn.q_proj.bias is not None:
                                spike_attn.q_proj.bias.data.copy_(block.attn.q_proj.bias.data)
                                spike_attn.k_proj.bias.data.copy_(block.attn.k_proj.bias.data)
                                spike_attn.v_proj.bias.data.copy_(block.attn.v_proj.bias.data)
                            
                            # Copy output projection if available
                            if hasattr(block.attn, 'out_proj'):
                                spike_attn.o_proj.weight.data.copy_(block.attn.out_proj.weight.data)
                                if hasattr(block.attn.out_proj, 'bias') and block.attn.out_proj.bias is not None:
                                    spike_attn.o_proj.bias.data.copy_(block.attn.out_proj.bias.data)
                        else:
                            # For other attention implementations, just use the default initialization
                            logger.warning(f"Unknown attention structure in block. Using default initialization.")
                    
                    except Exception as e:
                        logger.warning(f"Error during attention weight copying: {e}. Using default initialization.")
                    
                    # Replace the attention block
                    block.attn = spike_attn
                    attn_count += 1
        else:
            logger.warning(f"Model has GPT-style architecture but couldn't find transformer.h structure")
    
    # For BERT and other encoder-only architectures
    elif model_type and ('bert' in model_type or 'roberta' in model_type or 'distilbert' in model_type):
        logger.info(f"Using BERT-style attention handling for {model_type}")
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style architecture
            hidden_size = model.config.hidden_size
            num_heads = model.config.num_attention_heads
            
            for layer in model.encoder.layer:
                if hasattr(layer, 'attention'):
                    attn_block = layer.attention
                    # Get the self-attention component
                    if hasattr(attn_block, 'self'):
                        attn_self = attn_block.self
                        
                        # Create SpikeAttention module
                        spike_attn = SpikeAttention(
                            embed_dim=hidden_size,
                            num_heads=num_heads,
                            T=model.T if hasattr(model, 'T') else 16,
                            causal=False  # BERT uses bidirectional attention
                        )
                        
                        try:
                            # BERT typically has separate query, key, value projections
                            if hasattr(attn_self, 'query') and hasattr(attn_self, 'key') and hasattr(attn_self, 'value'):
                                # Copy weights
                                spike_attn.q_proj.weight.data.copy_(attn_self.query.weight.data)
                                spike_attn.k_proj.weight.data.copy_(attn_self.key.weight.data)
                                spike_attn.v_proj.weight.data.copy_(attn_self.value.weight.data)
                                
                                # Copy biases if they exist
                                if hasattr(attn_self.query, 'bias') and attn_self.query.bias is not None:
                                    spike_attn.q_proj.bias.data.copy_(attn_self.query.bias.data)
                                    spike_attn.k_proj.bias.data.copy_(attn_self.key.bias.data)
                                    spike_attn.v_proj.bias.data.copy_(attn_self.value.bias.data)
                                
                                # Copy output projection
                                if hasattr(attn_block, 'output') and hasattr(attn_block.output, 'dense'):
                                    spike_attn.o_proj.weight.data.copy_(attn_block.output.dense.weight.data)
                                    if hasattr(attn_block.output.dense, 'bias'):
                                        spike_attn.o_proj.bias.data.copy_(attn_block.output.dense.bias.data)
                            else:
                                logger.warning("Could not find query/key/value projections in BERT attention")
                        except Exception as e:
                            logger.warning(f"Error during BERT attention weight copying: {e}")
                        
                        # Replace the self-attention component
                        attn_block.self = spike_attn
                        attn_count += 1
        else:
            logger.warning(f"Model has BERT-style architecture but couldn't find encoder.layer structure")
    
    # For other model architectures with unknown structure
    else:
        # Try a generic approach by looking for attention modules
        logger.warning("Unknown model architecture type. Trying generic approach to find attention blocks...")
        
        # Look for transformer blocks with attention
        for name, module in model.named_modules():
            if any(attn_name in name.lower() for attn_name in ['attention', 'attn']) and isinstance(module, nn.Module):
                logger.info(f"Found potential attention module at {name}")
                
                # Try to determine parent module to replace the attention
                parent_path = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_path:
                    try:
                        parent = model
                        for attr in parent_path.split('.'):
                            parent = getattr(parent, attr)
                        
                        # Get model dimensions
                        if hasattr(model, 'config'):
                            if hasattr(model.config, 'hidden_size') and hasattr(model.config, 'num_attention_heads'):
                                hidden_size = model.config.hidden_size
                                num_heads = model.config.num_attention_heads
                                
                                # Create and set the spike attention
                                spike_attn = SpikeAttention(
                                    embed_dim=hidden_size,
                                    num_heads=num_heads,
                                    T=model.T if hasattr(model, 'T') else 16,
                                    causal=True  # Default to causal for safety
                                )
                                
                                # Replace the attention module
                                setattr(parent, child_name, spike_attn)
                                attn_count += 1
                                logger.info(f"Replaced attention at {name}")
                    except Exception as e:
                        logger.warning(f"Failed to replace attention at {name}: {e}")
    
    if attn_count == 0:
        raise NotImplementedError(f"Could not find compatible attention structure in model type '{model_type}'. "
                                 "Please implement specific handling for this architecture.")
    
    logger.info(f"Replaced {attn_count} attention blocks with SpikeAttention")
    return model


def replace_attention_with_loihi_mixer(model):
    """
    Replace attention modules with LoihiCausalContextMixer to avoid dense QK^T softmax attention.
    """
    model_type = ""
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        model_type = str(model.config.model_type).lower()
    attn_count = 0

    # Only replace the attention module itself, matched by its exact child name.
    # Matching any name *containing* "attn"/"attention" also matched children such as
    # `c_attn` / `q_proj` inside the attention block, over-replacing and corrupting it.
    attn_child_names = {"attn", "self_attn", "attention", "self_attention"}
    replaced_prefixes: List[str] = []

    for name, module in list(model.named_modules()):
        child_name = name.split(".")[-1]
        if child_name not in attn_child_names:
            continue
        if not hasattr(module, "forward"):
            continue
        # Skip modules living inside an already-replaced attention subtree.
        if any(name == p or name.startswith(p + ".") for p in replaced_prefixes):
            continue

        parent_path = ".".join(name.split(".")[:-1])
        if not parent_path:
            continue

        try:
            parent = model
            for attr in parent_path.split("."):
                parent = getattr(parent, attr)

            hidden_size = None
            if hasattr(model, "config"):
                hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                # fallback: try to infer from common GPT-2 naming
                hidden_size = getattr(getattr(model, "transformer", None), "embed_dim", None)

            if hidden_size is None:
                raise RuntimeError("Could not infer hidden_size for Loihi mixer replacement.")

            mixer = LoihiCausalContextMixer(hidden_size=int(hidden_size))
            # Llama-style decoder layers unpack three values from their attention module;
            # GPT-2 blocks index the returned tuple. Tell the mixer which layout to emit.
            if child_name == "self_attn" or 'llama' in model_type or 'mistral' in model_type or 'smollm' in model_type:
                mixer.return_mode = "llama"
            setattr(parent, child_name, mixer)
            replaced_prefixes.append(name)
            attn_count += 1
            logger.info(f"Replaced attention at {name} with LoihiCausalContextMixer")
        except Exception as e:
            logger.warning(f"Failed to replace attention at {name} for model_type={model_type}: {e}")

    if attn_count == 0:
        raise NotImplementedError(
            f"Could not find compatible attention structure in model type '{model_type}' "
            "for Loihi mixer replacement."
        )

    logger.info(f"Replaced {attn_count} attention blocks with LoihiCausalContextMixer")
    return model

def simplified_conversion(model, timesteps=32, skip_gelu_replacement=False):
    """Perform simplified conversion without relying on SpikingJelly.

    Args:
        model: The model to convert.
        timesteps: Number of timesteps for SNN simulation.
        skip_gelu_replacement: If True, skip GELU->ReLU replacement. This preserves
            text generation quality but sacrifices spike-compatibility. Set to True
            for inference testing; set to False for actual neuromorphic deployment.
    """
    logger.info(f"Using simplified conversion with T={timesteps}")

    # 1. Optionally replace GELU/NewGELUActivation with ReLU
    # Note: GELU->ReLU significantly degrades text generation quality
    # but is required for true spike-based neuromorphic execution
    if not skip_gelu_replacement:
        model = replace_gelu_with_relu(model)
    else:
        logger.info("Skipping GELU->ReLU replacement (better generation quality, less spike-compatible)")
    
    # 2. Store timesteps attribute
    model.T = timesteps
    
    # 3. Replace standard LayerNorm with SpikeLayerNorm
    model = replace_layernorm_with_spikelayernorm(model)
    
    # 4. Replace Attention with Loihi-friendly mixer or SpikeAttention
    loihi_mode = bool(getattr(model, "_stac_loihi_mode", False))
    if loihi_mode:
        logger.info("Loihi mode enabled: replacing attention with LoihiCausalContextMixer")
        model = replace_attention_with_loihi_mixer(model)
        embed_buckets = int(getattr(model, "_stac_loihi_embed_buckets", 0) or 0)
        if embed_buckets > 0:
            logger.info(f"Loihi embedding bucketing enabled: num_buckets={embed_buckets}")
            model = apply_loihi_embedding_bucketing(model, num_buckets=embed_buckets)
        if bool(getattr(model, "_stac_loihi_prune", False)):
            target = float(getattr(model, "_stac_loihi_prune_target", 0.5))
            logger.info(f"Loihi prune enabled: applying magnitude pruning (target_sparsity={target:.2f})")
            model = apply_magnitude_pruning_for_loihi(model, target_sparsity=target)
        if bool(getattr(model, "_stac_loihi_quantize", False)):
            logger.info("Loihi quantize enabled: applying fake int8 quantization pass")
            model = apply_fake_int8_quantization_for_loihi(model)
    else:
        model = replace_attention_with_spikeattention(model)
    
    # 5. Add a wrapper for temporal processing
    model = TemporalSpikeProcessor(model, T=timesteps)
    
    logger.info("Simplified SNN conversion completed")
    return model

def _clip_grad_hook(mod, grad_input, grad_output):
    """Clamp input gradients for stability, safely handling None entries.

    A full backward hook may receive ``None`` grad-input entries (e.g. for inputs
    that do not require grad). ``torch.clamp(None)`` raises, so leave those as-is.
    """
    if not grad_input:
        return grad_input
    clipped = tuple(
        torch.clamp(g, -1.0, 1.0) if g is not None else None
        for g in grad_input
    )
    return clipped


def apply_surrogate_gradients(model, alpha=4.0):
    """Apply surrogate gradients for spike backpropagation."""
    logger.info(f"Applying surrogate gradients with alpha={alpha}")

    # Find all LIF neurons and apply surrogate gradient
    count = 0
    atan_surrogate_fn = SurrogateModule.ATan(alpha=alpha) # Use aliased surrogate module
    for module in model.modules():
        if hasattr(module, 'neuron') and hasattr(module.neuron, 'surrogate_function'): # Check if it's a SpikingJelly neuron wrapper
            # This case might be for older SpikingJelly structures or custom wrappers.
            # Official LIFNode usually has surrogate_function directly on it.
            module.neuron.surrogate_function = atan_surrogate_fn
            count += 1
            module.neuron.register_full_backward_hook(_clip_grad_hook)
        elif isinstance(module, LIFNode): # Direct check for official LIFNode
            module.surrogate_function = atan_surrogate_fn
            count += 1
            # Add gradient clipping hook for stability
            module.register_full_backward_hook(_clip_grad_hook)

    logger.info(f"Applied ATan surrogate gradient to {count} LIFNode modules.")
    return model

def calibrate_timesteps(model, original_T, target_T):
    """Calibrate the model to run with fewer timesteps."""
    logger.info(f"Calibrating model: {original_T} -> {target_T} timesteps")

    if original_T <= 0:
        logger.warning(f"Invalid original_T={original_T}; skipping timestep calibration.")
        return model

    # Apply threshold scaling: v_th_new = v_th_old * (target_T / original_T)
    scale_factor = target_T / original_T
    count = 0
    
    # LIFNode is already defined from direct imports
    
    for module in model.modules():
        if isinstance(module, LIFNode):
            if hasattr(module, 'v_threshold') and module.v_threshold is not None:
                 module.v_threshold *= scale_factor
            count += 1
    
    # Update T attribute in TemporalSpikeProcessor and potentially in custom SpikeAttention/SpikeSoftmax
    if isinstance(model, TemporalSpikeProcessor):
        model.T = target_T
    elif hasattr(model, 'T'): # If it's the inner SNN model directly
        model.T = target_T

    # Also update T for custom spiking components within the SNN model
    for module in model.modules():
        if isinstance(module, (SpikeAttention, SpikeSoftmax)):
            if hasattr(module, 'T'): # If they have a T attribute
                module.T = target_T
            # If SpikeAttention has SpikeSoftmax internally, it should also be updated if not handled by parent T.
            if isinstance(module, SpikeAttention) and hasattr(module, 'spike_softmax') and hasattr(module.spike_softmax, 'T'):
                module.spike_softmax.T = target_T
    
    logger.info(f"Calibrated {count} LIF neurons and relevant T attributes for T={target_T}")
    return model

def save_snn_model(model, tokenizer, path):
    """Save the SNN model with metadata."""
    os.makedirs(path, exist_ok=True)

    # Read config from the wrapper if present, otherwise fall back to the inner model.
    config = getattr(model, 'config', None)
    if config is None:
        config = getattr(getattr(model, 'snn_model', None), 'config', None)

    # Extract/create metadata
    snn_config = {
        "timesteps": getattr(model, 'T', 16),
        "base_model": getattr(config, '_name_or_path', "") if config is not None else "",
        "model_type": getattr(config, 'model_type', "") if config is not None else "",
        "activation": "relu",
        "surrogate_gradient": "atan",
        "is_snn": True
    }

    # Save tokenizer
    tokenizer.save_pretrained(path)

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "T": getattr(model, 'T', 16),
        "snn_config": snn_config
    }, os.path.join(path, "snn_model.pt"))
    
    # Save SNN config as separate file
    with open(os.path.join(path, "snn_config.json"), "w") as f:
        json.dump(snn_config, f, indent=2)
    
    logger.info(f"Saved SNN model to {path}")
    return True

def main():
    """Main conversion function."""
    args = parse_args()
    device = args.device
    logger.info(f"Using device: {device}")
    logger.info(f"SpikingJelly version from main: {importlib.metadata.version('spikingjelly')}") # Confirm version

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading base model: {args.model_name}")
    # Load with BitsAndBytes if specified/possible, otherwise standard load
    quant_cfg = None
    torch_dtype_load = torch.float32 # Default
    if args.quantize_bits == 8:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head"] 
            )
            torch_dtype_load = torch.float16 # BNB 8bit usually used with fp16
            logger.info("8-bit quantization selected via BitsAndBytesConfig.")
        except Exception as e:
            logger.warning(f"Failed to create BitsAndBytesConfig for 8-bit: {e}. Will load in fp32/fp16.")
            quant_cfg = None
    elif args.quantize_bits == 4:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            torch_dtype_load = torch.bfloat16 # Often used with 4bit
            logger.info("4-bit quantization selected via BitsAndBytesConfig.")
        except Exception as e:
            logger.warning(f"Failed to create BitsAndBytesConfig for 4-bit: {e}. Will load in fp32/fp16.")
            quant_cfg = None
    
    model_load_args = {"torch_dtype": torch_dtype_load}
    if quant_cfg:
        model_load_args["quantization_config"] = quant_cfg
        # device_map="auto" is often used with BitsAndBytes
        # However, for SNN conversion, explicit device control might be better.
        # Let's stick to args.device for now unless device_map is critical.
        # model_load_args["device_map"] = device 

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_args)
    except Exception as e:
        logger.error(f"Failed to load model {args.model_name} with specified config: {e}. Trying with default float32.")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    model.to(device) # Ensure model is on the target device after loading

    calib_data = create_calibration_data(tokenizer, args.num_samples) # Assumed to be [(sample_dict, None), ...]
    
    model_for_snn = replace_gelu_with_relu(model)
    
    if args.quantize_bits != 0 and not quant_cfg: # If BitsAndBytes wasn't used but quantization is desired
        logger.info(f"Applying SpikingJelly {args.quantize_bits}-bit quantization (official Quantizer)...")
        # Quantizer is now the official one
        quantizer_instance = Quantizer(n_bits_w=args.quantize_bits, n_bits_a=args.quantize_bits)
        
        try:
            model_for_snn = quantizer_instance(model_for_snn)
            logger.info("SpikingJelly Quantization applied.")
        except Exception as e:
            logger.error(f"SpikingJelly Quantizer failed: {e}. Proceeding without it if possible.")

    logger.info(f"Converting to SNN components with T={args.timesteps} (simplified_conversion wrapper)...")
    # simplified_conversion prepares the model by replacing layers, sets model.T
    snn_parts_model = simplified_conversion(model_for_snn, args.timesteps)

    logger.info("Applying surrogate gradients using official SpikingJelly ATan...")
    snn_parts_model = apply_surrogate_gradients(snn_parts_model, alpha=4.0)

    # Now, use the official SpikingJelly Converter for the final step if its specific logic is desired
    # (e.g. data-based scaling, specific layer replacements it handles beyond simplified_conversion)
    # If simplified_conversion already does everything, this Converter step might be redundant or for refinement.
    # The prompt implied using official Converter. Let's assume it applies some final touches.
    logger.info(f"Applying official SpikingJelly Converter (T={args.timesteps})...")
    # Converter now comes from direct import and is the official one
    # It needs calibration data in a specific format (typically a DataLoader)
    # Our create_calibration_data returns a list of tuples. We might need to adapt.
    
    # Create a simple dataloader for the SpikingJelly Converter
    from torch.utils.data import DataLoader, Dataset
    class CalibrationDataset(Dataset):
        def __init__(self, calib_data_list):
            self.data = calib_data_list
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            # SpikingJelly converter expects input tensor directly, not dict or tuple usually
            sample_dict, _ = self.data[idx]
            return sample_dict['input_ids'].squeeze(0) # Return tensor [seq_len]

    if calib_data:
        sj_calib_dataset = CalibrationDataset(calib_data)
        # SpikingJelly converter usually expects batch_size 1 for this type of calibration data
        sj_calib_dataloader = DataLoader(sj_calib_dataset, batch_size=1) 
    else:
        sj_calib_dataloader = None
        logger.warning("No calibration data for SpikingJelly Converter. Some features might not work optimally.")

    try:
        # Converter is the class from direct import. Its signature is
        # (dataloader, device=None, mode='Max', momentum=0.1, fuse_flag=True) — there is
        # no `spiking_neuron_type` parameter, and passing one raised TypeError before any
        # conversion happened, so this branch always fell through to the except below.
        converter_instance = Converter(
            dataloader=sj_calib_dataloader,
            mode='max',
            device=device,
        )
        converted_snn_model = converter_instance(snn_parts_model)
        logger.info("Official SpikingJelly Converter applied.")
    except Exception as e:
        logger.error(f"Official SpikingJelly Converter failed: {e}. Using model from simplified_conversion.")
        converted_snn_model = snn_parts_model 
    
    # Wrap with TemporalSpikeProcessor for multi-step processing.
    # simplified_conversion() already returns a TemporalSpikeProcessor, so re-wrapping
    # would nest T x T timestep loops and apply the logit scaling twice. Only wrap when
    # the SpikingJelly Converter step replaced it with a bare model.
    logger.info("Wrapping with TemporalSpikeProcessor...")
    max_context = getattr(args, 'max_context_length', 512)  # Default fallback
    if isinstance(converted_snn_model, TemporalSpikeProcessor):
        logger.info("Model is already a TemporalSpikeProcessor; updating T/max_context_length in place.")
        final_snn_model = converted_snn_model
        final_snn_model.T = args.timesteps
        final_snn_model.max_context_length = max_context
    else:
        final_snn_model = TemporalSpikeProcessor(converted_snn_model, T=args.timesteps, max_context_length=max_context)
    final_snn_model.to(device)

    # Timestep calibration halves T, so leaving it on by default meant `--timesteps 32`
    # silently produced a model running at T=16. It is now opt-in.
    if getattr(args, 'calibrate_timesteps', False) and args.timesteps > 1:
        target_T = max(1, args.timesteps // 2)
        logger.info(f"Calibrating SNN timesteps: {args.timesteps} -> {target_T}")
        final_snn_model = calibrate_timesteps(final_snn_model, args.timesteps, target_T)
    
    logger.info(f"Saving SNN model to {args.output_dir}")
    save_snn_model(final_snn_model, tokenizer, args.output_dir)
    
    logger.info("SNN Conversion completed successfully.")
    return 0

if __name__ == "__main__":
    main() 