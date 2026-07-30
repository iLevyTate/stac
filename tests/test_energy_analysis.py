"""
Tests for scripts/energy_analysis.py.

The crossover analysis is only worth anything if its closed form agrees with what
spike_metrics.py actually measures. The first test pins that agreement; the rest check the
algebra is self-consistent, so a future edit to the energy model cannot quietly invalidate
the conclusions in docs/energy-crossover.md.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spike_metrics import measure_spikes


def _load_by_path(name: str, path: Path):
    """scripts/ is not a package, and tests/ is only a namespace package when the repo
    root happens to be on sys.path — see tests/_offline_models.py for why that bites."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ea = _load_by_path("energy_analysis", ROOT / "scripts" / "energy_analysis.py")
offline = _load_by_path("_stac_offline_models", ROOT / "tests" / "_offline_models.py")


@pytest.fixture(scope="module")
def spiking_report():
    """One measured forward pass of the spiking pipeline, shared across tests."""
    snn = offline.convert("tiny-gpt2", timesteps=8, real_spiking=True)
    return measure_spikes(snn, torch.randint(0, 200, (1, 32)), use_cache=False)


# --------------------------------------------------------------------------------------
# Agreement with the measured pipeline
# --------------------------------------------------------------------------------------

def test_closed_form_matches_spike_metrics(spiking_report):
    """
    ratio = T * (rho*f*r + 1 - f) must reproduce spike_metrics' measured E_SNN/E_ANN.

    Any divergence beyond a percent means the closed form and the instrumentation have
    drifted apart, and the crossover conclusions no longer describe this codebase.
    """
    rep = spiking_report
    assert rep.macs > 0, "no MACs counted; the harness is not exercising the model"
    assert rep.energy_ann_pj > 0

    coverage = rep.macs_spike_replaced / rep.macs
    measured = rep.energy_snn_pj / rep.energy_ann_pj
    predicted = ea.energy_ratio(coverage, rep.timesteps, rep.spike_mean)

    assert predicted == pytest.approx(measured, rel=0.01), (
        f"closed form {predicted:.4f} disagrees with measured {measured:.4f} "
        f"(coverage={coverage:.4f}, T={rep.timesteps}, rho={rep.spike_mean:.4f})"
    )


def test_current_architecture_has_no_winning_spiking_operating_point(spiking_report):
    """
    The load-bearing negative result: with only QK^T spike-driven, the affordable timestep
    count is barely above 1 -- and T=1 is not a spiking network. If this starts failing,
    coverage has genuinely improved and docs/energy-crossover.md needs updating.
    """
    rep = spiking_report
    coverage = rep.macs_spike_replaced / rep.macs
    assert coverage < 0.10, f"coverage rose to {coverage:.3f}; revisit docs/energy-crossover.md"
    assert ea.max_timesteps(coverage, rep.spike_mean) < 2.0


# --------------------------------------------------------------------------------------
# Algebra
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("timesteps", [2, 4, 8, 16, 32])
def test_required_coverage_and_max_timesteps_are_inverses(timesteps):
    """f* is exactly the coverage at which T_max equals T, and the ratio there is 1."""
    rho = 0.094
    f_star = ea.required_coverage(timesteps, rho)
    assert ea.max_timesteps(f_star, rho) == pytest.approx(timesteps, rel=1e-9)
    assert ea.energy_ratio(f_star, timesteps, rho) == pytest.approx(1.0, rel=1e-9)


def test_single_timestep_needs_no_coverage():
    """At T=1 the SNN pays the dense cost once, exactly like the ANN."""
    assert ea.required_coverage(1, 0.094) == pytest.approx(0.0)
    assert ea.energy_ratio(0.0, 1, 0.094) == pytest.approx(1.0)


def test_coverage_dominates_sparsity():
    """
    The central claim of the analysis: rho enters only via rho*r, so sparsity is a weak
    lever compared with coverage. Driving the spike rate to zero must move T_max less than
    a modest coverage gain does.
    """
    base = ea.max_timesteps(0.55, 0.094)
    perfect_sparsity = ea.max_timesteps(0.55, 0.0)      # rho -> 0, coverage unchanged
    more_coverage = ea.max_timesteps(0.75, 0.094)       # coverage up, rho unchanged

    assert perfect_sparsity - base < 0.15, "sparsity unexpectedly strong"
    assert more_coverage - base > 1.0, "coverage unexpectedly weak"
    assert more_coverage > perfect_sparsity


def test_lm_head_share_shrinks_with_model_width():
    """
    lm_head is d*V (linear in width); the body is L*d^2 (quadratic). The block that is
    hardest to spike therefore matters less as models grow -- which is why benchmarking
    on a tiny model understates the achievable advantage.
    """
    def head_fraction(arch):
        cats = ea.macs_from_arch(ea.ARCHS[arch], seq_len=512)
        return cats["lm_head"] / sum(cats.values())

    small = head_fraction("distilgpt2")
    large = head_fraction("smollm2-1.7b")
    assert small > 0.40, small
    assert large < 0.10, large


def test_large_model_body_spiking_wins_at_t8():
    """
    Spiking everything except lm_head on the 1.7B target projects cheaper than the ANN at
    T=8, while the same coverage on distilgpt2 does not. This asymmetry is the actionable
    result, so pin it.
    """
    rho = 0.094
    for arch, should_win in [("smollm2-1.7b", True), ("distilgpt2", False)]:
        cats = ea.macs_from_arch(ea.ARCHS[arch], seq_len=512)
        total = sum(cats.values())
        body = 1.0 - cats["lm_head"] / total
        wins = ea.energy_ratio(body, 8, rho) < 1.0
        assert wins is should_win, f"{arch}: expected win={should_win}"
