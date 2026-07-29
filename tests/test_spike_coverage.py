"""
Tests for spike_coverage.py.

The module exists to answer whether the model survives the spike coverage an energy
advantage requires. That answer is worthless if the spike encoding is itself broken, so
these tests pin the encoding's fidelity before anything measures quality with it.

Two bugs found while building it motivate most of what is here:

* a *leaky* neuron with hard reset cannot rate-code at all -- sub-threshold input decays to
  a steady state below threshold and never fires, so the layer emits nothing;
* discarding negative activations (unsigned encoding) caps achievable fidelity around
  correlation 0.69 no matter how many timesteps are spent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spike_coverage import (  # noqa: E402
    COMPONENTS,
    SpikeLinear,
    apply_spike_coverage,
    calibrate_thresholds,
    spike_linears,
)


def _accumulate(layer: nn.Module, x: torch.Tensor, timesteps: int) -> torch.Tensor:
    """Run `timesteps` steps of the rate code and average, as TemporalSpikeProcessor does."""
    from spikingjelly.activation_based import functional

    functional.reset_net(layer)
    total = None
    for _ in range(timesteps):
        out = layer(x)
        total = out if total is None else total + out
    return total / timesteps


def _calibrated(dense: nn.Linear, x: torch.Tensor, signed: bool = True) -> SpikeLinear:
    layer = SpikeLinear(nn.Linear(dense.in_features, dense.out_features), signed=signed)
    layer.inner.load_state_dict(dense.state_dict())
    layer.calibrating = True
    layer(x)
    layer.finalize_calibration()
    return layer


@pytest.fixture(scope="module")
def dense_and_input():
    torch.manual_seed(0)
    dense = nn.Linear(64, 64)
    return dense, torch.randn(1, 32, 64)


# --------------------------------------------------------------------------------------
# Encoding fidelity
# --------------------------------------------------------------------------------------

def test_rate_code_converges(dense_and_input):
    """
    Averaged spiking output must approach the dense output as timesteps grow.

    This is the property the whole coverage/quality experiment rests on. It fails outright
    for a leaky neuron with hard reset, which is what the module used first.
    """
    dense, x = dense_and_input
    with torch.no_grad():
        reference = dense(x)
        layer = _calibrated(dense, x)

        correlations = []
        for timesteps in (8, 32, 128):
            approx = _accumulate(layer, x, timesteps)
            corr = torch.corrcoef(torch.stack([approx.flatten(), reference.flatten()]))[0, 1]
            correlations.append(float(corr))

    assert correlations[0] > 0.95, f"T=8 correlation {correlations[0]:.4f} too low"
    assert correlations[-1] > 0.99, f"T=128 correlation {correlations[-1]:.4f} too low"
    assert correlations == sorted(correlations), f"not monotonic in T: {correlations}"


def test_soft_reset_beats_hard_reset_and_leak():
    """
    Pin the neuron-model choice with a measurement rather than a comment.

    Input in [0,1] against threshold 1.0: a soft-reset IF tracks the input almost exactly,
    a hard-reset IF discards the excess above threshold, and a leaky hard-reset LIF never
    fires at all.
    """
    from spikingjelly.activation_based import functional
    from spikingjelly.activation_based.neuron import IFNode, LIFNode

    torch.manual_seed(0)
    x = torch.rand(1, 512)

    def mean_rate(neuron, timesteps=64):
        functional.reset_net(neuron)
        return sum(neuron(x) for _ in range(timesteps)) / timesteps

    soft = mean_rate(IFNode(v_threshold=1.0, v_reset=None, detach_reset=True))
    hard = mean_rate(IFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True))
    leaky = mean_rate(LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True))

    soft_err = float((soft - x).abs().mean() / x.abs().mean())
    hard_err = float((hard - x).abs().mean() / x.abs().mean())

    assert soft_err < 0.05, f"soft reset should track the input closely, got {soft_err:.4f}"
    assert hard_err > soft_err * 3, "hard reset should be markedly worse than soft reset"
    assert float(leaky.sum()) == 0.0, "a leaky hard-reset neuron should not fire here at all"


def test_unsigned_encoding_loses_negative_activations(dense_and_input):
    """
    Signed encoding is a real modelling choice, not decoration: dropping the negative
    population caps fidelity however many timesteps are spent.
    """
    dense, x = dense_and_input
    reference = dense(x)

    def correlation(signed):
        with torch.no_grad():
            approx = _accumulate(_calibrated(dense, x, signed=signed), x, 128)
            return float(torch.corrcoef(torch.stack([approx.flatten(), reference.flatten()]))[0, 1])

    assert correlation(True) > 0.99
    assert correlation(False) < 0.85


def test_calibration_tracks_input_scale():
    """Thresholds must follow the activation distribution, or the rate code saturates."""
    torch.manual_seed(0)
    dense = nn.Linear(32, 32)
    for scale in (0.1, 1.0, 10.0):
        x = torch.randn(1, 16, 32) * scale
        layer = _calibrated(dense, x)
        expected = float(x.abs().flatten().quantile(0.99))
        assert layer.threshold.item() == pytest.approx(expected, rel=0.2), (
            f"scale {scale}: threshold {layer.threshold.item():.4f} vs expected {expected:.4f}"
        )


# --------------------------------------------------------------------------------------
# Selective application
# --------------------------------------------------------------------------------------

def _tiny_gpt2():
    from transformers import AutoModelForCausalLM, AutoConfig

    cfg = AutoConfig.for_model("gpt2", n_layer=2, n_head=2, n_embd=32, vocab_size=128)
    return AutoModelForCausalLM.from_config(cfg)


@pytest.mark.parametrize("components", [["mlp"], ["lm_head"], ["mlp", "lm_head"]])
def test_apply_spike_coverage_wraps_only_requested_components(components):
    model = _tiny_gpt2()
    info = apply_spike_coverage(model, components)
    assert info["total"] > 0, f"nothing wrapped for {components}"
    assert set(info["wrapped"]) == set(components)


def test_apply_spike_coverage_rejects_unknown_component():
    with pytest.raises(ValueError, match="unknown component"):
        apply_spike_coverage(_tiny_gpt2(), ["not_a_real_component"])


def test_components_match_energy_analysis_labels():
    """Coverage labels must line up with the energy model, or the two disagree silently."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ea_labels", ROOT / "scripts" / "energy_analysis.py"
    )
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)

    known = set(ea.macs_from_arch(ea.ARCHS["distilgpt2"], seq_len=8))
    assert set(COMPONENTS) <= known, f"{set(COMPONENTS) - known} unknown to energy_analysis"


def test_spike_metrics_counts_wrapped_layers_as_spike_driven():
    """
    Coverage must actually move when layers are wrapped. If spike_metrics does not
    recognise SpikeLinear, the sweep reports a flat coverage curve and every quality
    conclusion drawn from it is meaningless.
    """
    from spike_metrics import count_macs

    model = _tiny_gpt2()
    before = count_macs(model, seq_len=8)
    apply_spike_coverage(model, ["mlp"])
    after = count_macs(model, seq_len=8)

    assert before["spike_replaceable"] == 0
    assert after["spike_replaceable"] > 0
    assert after["total"] == pytest.approx(before["total"]), (
        "wrapping must not change the dense MAC denominator"
    )


def test_calibrate_thresholds_visits_every_wrapped_layer():
    model = _tiny_gpt2()
    apply_spike_coverage(model, ["mlp"])
    for layer in spike_linears(model):
        layer.threshold.fill_(-1.0)  # sentinel; calibration must overwrite

    count = calibrate_thresholds(model, [torch.randint(0, 128, (1, 8))])

    assert count > 0
    for layer in spike_linears(model):
        assert layer.threshold.item() > 0, "threshold left at its sentinel"
