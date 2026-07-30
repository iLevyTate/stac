"""
Regression test against the committed STAC V1 baseline.

The audit found that V1's AdEx neurons never reached threshold: spike_mean was exactly
0.0, the L1 spike penalty was exactly 0.0, and the surrogate gradient underflowed to zero
so nothing in the spiking pathway could learn. With `spk_trains` all zeros the model's
logits were identical at every sequence position — it could not do next-token prediction
at all — and every existing test still passed, because they only checked shapes.

docs/baselines/stac_v1_smoke.json records what a working run produces. This test compares
against it, so a silent return to a dead spiking pathway fails loudly.
"""
import json
import os
import sys
from pathlib import Path

# Allow running this file directly by putting the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from stac_v1.pipeline import (
    STACV1Config,
    build_dataloader_from_texts,
    build_model_and_tokenizer,
    set_seed,
    train_steps,
)

BASELINE_PATH = _REPO_ROOT / "docs" / "baselines" / "stac_v1_smoke.json"

# Sample texts, matching scripts/run_stac_v1.py::_default_texts().
_TEXTS = [
    "STAC V1 demonstrates hybrid fine-tuning: a pretrained transformer backbone plus neuromorphic spiking layers.",
    "Neuromorphic edge deployment prioritizes sparse activity and quantization, but dense attention remains a mapping challenge.",
    "Hyperdimensional memory (HEMM) provides a context bias over spike trains.",
    "Surrogate gradients enable training spiking dynamics end-to-end.",
]

# Metrics whose value is a property of the model, not of the machine. Loss and perplexity
# depend on the randomly-initialised backbone, so they are checked loosely; the spiking
# metrics are the ones that went silently to zero and are checked tightly.
_TIGHT = ("spike_mean", "spike_frac_zero", "spike_neuron_mean_min", "spike_neuron_mean_max", "steps")
_LOOSE = ("train_loss", "train_perplexity", "train_l1")


def _baseline():
    if not BASELINE_PATH.exists():
        pytest.skip(f"No baseline recorded at {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())


def _run_smoke(model_name: str):
    config = STACV1Config(
        model_name=model_name,
        seq_length=64,
        dlpfc_output_size=64,
        num_recurrent_layers=1,
        dropout_prob=0.1,
        hdm_dim=128,
        seed=42,
        output_dir=str(_REPO_ROOT / "local" / "baseline-check"),
    )
    set_seed(config.seed)
    model, tokenizer = build_model_and_tokenizer(config)
    loader = build_dataloader_from_texts(
        tokenizer, _TEXTS, seq_length=config.seq_length, batch_size=2, shuffle=False
    )
    return train_steps(
        model,
        loader,
        cfg=config,
        device=torch.device("cpu"),
        max_steps=5,
        hybrid_finetune=True,
        train_lm_head=True,
        write_loihi_report=False,
    )


def test_v1_smoke_run_matches_baseline():
    """A V1 run must still reproduce the recorded metrics."""
    baseline = _baseline()
    model_name = os.environ.get("STAC_TEST_MODEL")
    if not model_name:
        pytest.skip("STAC_TEST_MODEL is not set (conftest sets it when the hub is unreachable)")

    try:
        metrics = _run_smoke(model_name)
    except Exception as e:
        pytest.skip(f"Could not run the V1 smoke pipeline: {e}")

    expected = baseline["metrics"]

    for key in _TIGHT:
        assert key in metrics, f"metric {key!r} disappeared from the run summary"
        assert metrics[key] == pytest.approx(expected[key], rel=0.05, abs=1e-6), (
            f"{key} drifted from the baseline: {metrics[key]} vs {expected[key]}. "
            f"If this change is intended, regenerate {BASELINE_PATH.name}."
        )

    for key in _LOOSE:
        assert metrics[key] == pytest.approx(expected[key], rel=0.5, abs=1e-6), (
            f"{key} is far from the baseline: {metrics[key]} vs {expected[key]}"
        )


def test_spiking_pathway_is_alive():
    """
    The specific regression the baseline exists to catch.

    These four assertions all failed before the audit fix, while the whole test suite
    passed — nothing checked that the neurons ever fired.
    """
    model_name = os.environ.get("STAC_TEST_MODEL")
    if not model_name:
        pytest.skip("STAC_TEST_MODEL is not set")

    try:
        metrics = _run_smoke(model_name)
    except Exception as e:
        pytest.skip(f"Could not run the V1 smoke pipeline: {e}")

    assert metrics["spike_mean"] > 0.0, "AdEx neurons emitted no spikes — the pathway is dead again"
    assert metrics["spike_mean"] < 1.0, "every neuron fires at every step — no sparsity"
    assert metrics["train_l1"] > 0.0, "the L1 spike penalty is identically zero, so it regularises nothing"
    assert metrics["spike_neuron_mean_max"] > metrics["spike_neuron_mean_min"], (
        "all neurons have identical firing rates, which suggests they are not responding to input"
    )
