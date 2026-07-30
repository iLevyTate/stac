"""
Conversion must preserve rotary position embeddings.

`SpikeAttention` replaces the whole attention block. Llama-family models (every SmolLM2
variant this repo targets) keep ALL of their positional information in RoPE, applied
inside that block — unlike GPT-2, whose learned positional embedding lives in the
embedding layer and survives replacement untouched. The original replacement accepted
`position_embeddings` via **kwargs and dropped it: converted SmolLM2 lost position
entirely, measured at 19-28x worse perplexity with spiking OFF.

These tests pin the fix at three levels: the kwarg is honoured, position actually
influences the output, and the non-spiking conversion tracks the source model's logits.
Everything runs offline against the generated grouped-query Llama fixture.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smollm2_converter import SpikeAttention, TemporalSpikeProcessor  # noqa: E402


def _load_by_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


offline = _load_by_path("_stac_offline_models", ROOT / "tests" / "_offline_models.py")


def _load_llama():
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(offline.model_path("tiny-llama-gqa"))
    except Exception as e:  # pragma: no cover - fixture generation failed
        pytest.skip(f"could not load llama fixture: {e}")
    return model.eval()


# --------------------------------------------------------------------------------------
# Module level: the kwarg changes the output
# --------------------------------------------------------------------------------------

def test_position_embeddings_kwarg_is_applied():
    """
    Passing rotated vs. unrotated position embeddings must change the attention output.
    Before the fix, `position_embeddings` was swallowed by **kwargs and the output was
    identical — which is exactly the assertion that would have caught it.
    """
    torch.manual_seed(0)
    attn = SpikeAttention(embed_dim=64, num_heads=4, T=4, num_kv_heads=2).eval()
    hidden = torch.randn(1, 10, 64)
    head_dim = 16

    # cos/sin in the shape transformers hands down: [batch, seq, head_dim]
    positions = torch.arange(10, dtype=torch.float32)
    freqs = torch.outer(positions, 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim)))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
    identity = (torch.ones_like(cos), torch.zeros_like(sin))  # rotation by zero

    with torch.no_grad():
        rotated = attn(hidden, position_embeddings=(cos, sin))[0]
        unrotated = attn(hidden, position_embeddings=identity)[0]
        dropped = attn(hidden)[0]

    assert not torch.allclose(rotated, unrotated), (
        "position_embeddings had no effect on the output — RoPE is being dropped"
    )
    # Rotation by zero must be a true no-op, so `identity` doubles as a correctness check.
    assert torch.allclose(unrotated, dropped, atol=1e-5)


# --------------------------------------------------------------------------------------
# Model level: position information survives conversion
# --------------------------------------------------------------------------------------

def test_converted_llama_is_position_sensitive():
    """
    With RoPE lost, attention is position-blind: permuting the input tokens permutes the
    value mixture but cannot change *how* positions attend, so the logits at a fixed
    position barely move. With RoPE intact, permutation must change them substantially.
    """
    converted = offline.convert("tiny-llama-gqa", timesteps=4, real_spiking=False)
    inner = converted.snn_model if isinstance(converted, TemporalSpikeProcessor) else converted

    torch.manual_seed(0)
    ids = torch.randint(0, 250, (1, 24))
    swapped = ids.clone()
    swapped[0, :12] = ids[0, 12:24]
    swapped[0, 12:24] = ids[0, :12]

    with torch.no_grad():
        base = inner(ids, use_cache=False)
        perm = inner(swapped, use_cache=False)
    base_logits = (base.logits if hasattr(base, "logits") else base[0])[0, -1]
    perm_logits = (perm.logits if hasattr(perm, "logits") else perm[0])[0, -1]

    # The final position sees the same *set* of tokens either way; only their order
    # differs. A position-blind model gives (near-)identical logits here.
    delta = (base_logits - perm_logits).abs().mean() / base_logits.abs().mean().clamp(min=1e-6)
    assert delta > 0.01, (
        f"reordering the context moved the final logits by only {delta:.2e} — "
        "the converted model is position-blind (RoPE lost in conversion)"
    )


def test_conversion_preserves_llama_logits():
    """
    The end-to-end guarantee: non-spiking conversion of a RoPE model must track the
    source model's logits closely. This is the model-level regression that fails if any
    future attention rewrite drops position information again, whatever the mechanism.
    """
    from transformers import AutoModelForCausalLM

    source = _load_llama()
    converted = offline.convert("tiny-llama-gqa", timesteps=4, real_spiking=False)
    inner = converted.snn_model if isinstance(converted, TemporalSpikeProcessor) else converted

    torch.manual_seed(1)
    ids = torch.randint(0, 250, (1, 32))
    with torch.no_grad():
        ref = source(ids).logits[0].float()
        got_out = inner(ids, use_cache=False)
        got = (got_out.logits if hasattr(got_out, "logits") else got_out[0])[0].float()

    cos = torch.nn.functional.cosine_similarity(ref, got, dim=-1).mean()
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean()

    # SpikeLayerNorm/SpikeRMSNorm and the attention rewrite introduce small numerical
    # differences, so exact equality is not expected — but position loss shows up as
    # near-zero agreement, nowhere near these floors.
    assert cos > 0.98, f"cosine {cos:.4f}: conversion is not tracking the source model"
    assert agree > 0.80, f"top-1 agreement {agree:.4f}: conversion changed the predictions"
