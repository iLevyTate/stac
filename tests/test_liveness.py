"""
Liveness assertions: catch networks that report success while doing nothing.

Every major defect this project's audit found shared one signature — a convenient metric
reading its best possible value on a dead network:

* STAC V1's L1 spike penalty sat at exactly 0.0 for its entire life because the neurons
  never fired (docs/corrigendum-2026-07.md);
* cosine similarity to teacher logits ROSE from 0.978 to 0.986 while the converted model
  collapsed to emitting commas (docs/coverage-quality.md);
* the spiking attention path ran for months while `q_spikes = q` bypassed it entirely.

These tests assert the three liveness properties directly, so that class of failure trips
CI instead of surviving for a release cycle:

1. spiking modules actually spike (rate > 0),
2. they are not saturated either (rate < 1 — a neuron firing every step carries nothing),
3. the model's predictions vary by position (a near-constant argmax is a dead model even
   when its perplexity is finite and its cosine similarity is high).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spike_metrics import measure_spikes  # noqa: E402


def _load_by_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


offline = _load_by_path("_stac_offline_models", ROOT / "tests" / "_offline_models.py")


@pytest.fixture(scope="module")
def spiking_model():
    return offline.convert("tiny-gpt2", timesteps=8, real_spiking=True)


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 200, (1, 48))


# --------------------------------------------------------------------------------------
# 1 + 2: the spiking pathway is alive and not saturated
# --------------------------------------------------------------------------------------

def test_spiking_modules_emit_spikes(spiking_model, input_ids):
    """
    Zero spikes means the spiking pathway is dead, whatever the loss curves say.
    This exact assertion, had it existed, would have caught the V1 defect in 2025.
    """
    report = measure_spikes(spiking_model, input_ids, use_cache=False)
    assert report.spiking_modules > 0, "conversion produced no spiking modules at all"
    assert report.invoked_modules > 0, (
        "spiking modules exist but none ran — the forward path bypasses them "
        "(the q_spikes = q defect)"
    )
    assert report.total_spikes > 0, (
        "spiking modules ran and emitted zero spikes — the operating point is "
        "below threshold (the V1 defect)"
    )


def test_spike_rate_not_saturated(spiking_model, input_ids):
    """
    A neuron that fires on every timestep carries no information: the rate code's
    codebook is {0, 1} and saturation pins everything to 1. Saturation reads as
    healthy activity to any "are there spikes?" check, which is why the band is
    asserted from both sides.
    """
    report = measure_spikes(spiking_model, input_ids, use_cache=False)
    assert 0.0 < report.spike_mean < 0.95, (
        f"spike rate {report.spike_mean:.3f} is outside the informative band; "
        "~1.0 means saturated inputs (threshold far too low), ~0.0 means near-dead"
    )


# --------------------------------------------------------------------------------------
# 3: predictions vary by position
# --------------------------------------------------------------------------------------

def _unique_prediction_fraction(model, ids) -> float:
    with torch.no_grad():
        out = model(ids, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    preds = logits[0].argmax(-1)
    return preds.unique().numel() / preds.numel()


def test_ann_baseline_has_diverse_predictions(input_ids):
    """
    Calibrate the diversity floor against the unconverted model, so the spiking test
    below fails because conversion killed diversity, not because a random-weight
    fixture was never diverse.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(offline.model_path("tiny-gpt2")).eval()
    assert _unique_prediction_fraction(model, input_ids) > 0.05


def test_converted_model_predictions_vary_by_position(input_ids):
    """
    The collapse signature: a converted model whose argmax is near-constant across
    positions. distilgpt2 under full spike coverage produced 6 distinct predictions
    over 256 positions (86.7% commas) while cosine similarity to the ANN read 0.978.
    Perplexity and similarity metrics do not catch this; prediction diversity does.

    The non-spiking conversion must preserve the ANN's diversity almost exactly —
    it is numerically faithful by design.
    """
    converted = offline.convert("tiny-gpt2", timesteps=8, real_spiking=False)
    fraction = _unique_prediction_fraction(converted, input_ids)
    assert fraction > 0.05, (
        f"converted model emits {fraction:.1%} unique predictions — near-constant "
        "output is a dead model regardless of any similarity metric"
    )
