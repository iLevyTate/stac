#!/usr/bin/env python3
"""
Build tiny local models so the STAC test suite can run without network access.

The suites default to Hugging Face hub ids (``distilgpt2``, ``sshleifer/tiny-gpt2``).
In a sandbox, an air-gapped CI runner, or behind a proxy that blocks huggingface.co,
nothing in this repository can be executed at all — which is exactly how several
long-standing bugs stayed invisible.

This script writes randomly-initialised checkpoints with real architectures and a
self-contained byte-level tokenizer, so every code path (GPT-2, Llama, grouped-query
Llama) can be exercised offline. The weights are random: use these for *behavioural*
testing (shapes, dtypes, caching, causality, spike rates, crashes), not for measuring
generation quality.

Usage:
    python scripts/make_test_models.py --out local/test-models
    STAC_TEST_MODEL=local/test-models/tiny-gpt2 python -m pytest tests/ -q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this file directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
)
import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization


# Every byte is its own token plus one special token. Small, deterministic, and needs no
# downloaded merges/vocab files.
BYTE_VOCAB_SIZE = 257
EOS_TOKEN = "<|endoftext|>"


# Written into every generated model directory. Tests that measure *language quality*
# (rather than behaviour) check for it and skip: random weights cannot produce coherent
# text, so asserting on their output would be meaningless.
FIXTURE_MARKER = "stac_fixture.json"


def is_random_weight_fixture(model_path) -> bool:
    """True if `model_path` is a generated fixture rather than a trained checkpoint."""
    try:
        return (Path(model_path) / FIXTURE_MARKER).exists()
    except (TypeError, OSError):
        return False


def _write_fixture_marker(out_dir: Path, architecture: str, seed: int) -> None:
    (out_dir / FIXTURE_MARKER).write_text(
        json.dumps(
            {
                "generated_by": "scripts/make_test_models.py",
                "architecture": architecture,
                "seed": seed,
                "random_weights": True,
                "note": (
                    "Randomly initialised. Suitable for behavioural tests (shapes, dtypes, "
                    "caching, causality, spike rates, crashes); not for generation quality."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_byte_tokenizer(out_dir: Path) -> GPT2TokenizerFast:
    """Write a self-contained byte-level BPE tokenizer with no merges."""
    out_dir.mkdir(parents=True, exist_ok=True)

    byte_to_unicode = gpt2_tokenization.bytes_to_unicode()
    vocab = {byte_to_unicode[i]: i for i in range(256)}
    vocab[EOS_TOKEN] = 256

    (out_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (out_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    tokenizer = GPT2TokenizerFast(
        vocab_file=str(out_dir / "vocab.json"),
        merges_file=str(out_dir / "merges.txt"),
        unk_token=EOS_TOKEN,
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(out_dir))
    return tokenizer


def build_tiny_gpt2(out_dir: Path, *, seed: int, n_positions: int = 1024) -> Path:
    """
    GPT-2 architecture: Conv1D attention, learned position embeddings, tied lm_head.

    `n_positions` defaults to 1024 to match distilgpt2, so position-boundary behaviour
    matches the real model the docs describe.
    """
    torch.manual_seed(seed)
    build_byte_tokenizer(out_dir)
    config = GPT2Config(
        vocab_size=BYTE_VOCAB_SIZE,
        n_positions=n_positions,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=BYTE_VOCAB_SIZE - 1,
        eos_token_id=BYTE_VOCAB_SIZE - 1,
    )
    GPT2LMHeadModel(config).save_pretrained(str(out_dir))
    _write_fixture_marker(out_dir, "gpt2", seed)
    return out_dir


def build_tiny_llama(out_dir: Path, *, seed: int, num_key_value_heads: int = 4) -> Path:
    """
    Llama architecture: separate q/k/v/o projections, RMSNorm, SiLU, rotary embeddings.

    Set `num_key_value_heads` < `num_attention_heads` for a grouped-query-attention
    model, matching SmolLM2-135M/360M.
    """
    torch.manual_seed(seed)
    build_byte_tokenizer(out_dir)
    config = LlamaConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=512,
        bos_token_id=BYTE_VOCAB_SIZE - 1,
        eos_token_id=BYTE_VOCAB_SIZE - 1,
    )
    LlamaForCausalLM(config).save_pretrained(str(out_dir))
    _write_fixture_marker(out_dir, "llama", seed)
    return out_dir


# name -> builder. `tiny-gpt2` is the default target for STAC_TEST_MODEL.
MODEL_BUILDERS = {
    "tiny-gpt2": build_tiny_gpt2,
    "tiny-llama": build_tiny_llama,
    "tiny-llama-gqa": lambda out, *, seed: build_tiny_llama(out, seed=seed, num_key_value_heads=1),
}


def ensure_test_model(name: str = "tiny-gpt2", *, out_root: Path | str, seed: int = 0) -> Path:
    """
    Return the path to a generated model, building it only if absent.

    Used by tests/conftest.py so the suite is runnable with no network and no manual
    setup step.
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown test model {name!r}; choose from {sorted(MODEL_BUILDERS)}")
    out_dir = Path(out_root) / name
    if (
        (out_dir / "config.json").exists()
        and (out_dir / "tokenizer_config.json").exists()
        and (out_dir / FIXTURE_MARKER).exists()
    ):
        return out_dir
    return MODEL_BUILDERS[name](out_dir, seed=seed)


def hub_reachable(timeout: float = 3.0) -> bool:
    """Cheap probe for Hugging Face hub availability."""
    import os
    import socket
    import urllib.error
    import urllib.request

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/gpt2", timeout=timeout)
        return True
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


def resolve_test_model(default: str, *, name: str = "tiny-gpt2",
                       out_root: Path | str | None = None) -> str:
    """
    Decide which model a test suite should use.

    Order: an explicit STAC_TEST_MODEL, then `default` if the hub is reachable, then a
    locally generated model. Shared by tests/conftest.py and the standalone test runners
    so `python tests/test_v1.py` behaves the same offline as `pytest` does.
    """
    import os

    configured = os.environ.get("STAC_TEST_MODEL")
    if configured:
        return configured
    if hub_reachable():
        return default

    root = Path(out_root) if out_root else Path(__file__).resolve().parents[1] / "local" / "test-models"
    try:
        path = str(ensure_test_model(name, out_root=root))
    except Exception:
        return default  # let the caller fail with its own message
    os.environ.setdefault("STAC_TEST_MODEL", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tiny local models for offline testing")
    parser.add_argument("--out", default="local/test-models", help="Output root directory")
    parser.add_argument(
        "--models",
        nargs="*",
        default=sorted(MODEL_BUILDERS),
        choices=sorted(MODEL_BUILDERS),
        help="Which models to build (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Weight-initialisation seed")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the model exists")
    args = parser.parse_args()

    out_root = Path(args.out)
    for name in args.models:
        out_dir = out_root / name
        if args.force and out_dir.exists():
            for path in sorted(out_dir.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
        path = ensure_test_model(name, out_root=out_root, seed=args.seed)
        print(f"✓ {name:16} -> {path}")

    print(f"\nRun the suite against one with:\n  STAC_TEST_MODEL={out_root / 'tiny-gpt2'} python -m pytest tests/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
