"""
Tests for genuine spiking mode, grouped-query attention, and the energy projection.

These cover behaviour that previously had no test at all, and which the audit found to be
silently broken:

* the T-timestep loop was an exact no-op because the spiking neurons were bypassed;
* grouped-query models raised NotImplementedError;
* the "spike-count analysis" the README refers to did not exist.

Everything here runs offline against generated models (see scripts/make_test_models.py).
"""
import importlib.util
import sys
from pathlib import Path

# Allow running this file directly by putting the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from smollm2_converter import SpikeAttention
from loihi_constraints import validate_loihi_export_readiness
from spike_metrics import measure_spikes


def _load_offline_helpers():
    """
    Load tests/_offline_models.py by path.

    Not `from tests._offline_models import ...`: whether `tests` is importable as a
    package depends on pytest's import mode and rootdir detection, and CI proved it is
    not (ModuleNotFoundError on a cross-import that resolved fine locally). A path load
    works regardless of how the suite is invoked.
    """
    name = "_stac_offline_models"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / "_offline_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_offline = _load_offline_helpers()
_model_path = _offline.model_path
_convert = _offline.convert


def test_spiking_mode_invokes_its_neurons():
    """
    The neurons must actually run.

    They were constructed and bypassed (`q_spikes = q  # self.q_spk(q)`), so the model
    reported spiking modules while performing no spiking computation.
    """
    ids = torch.randint(0, 200, (1, 12))

    non_spiking = _convert("tiny-gpt2", timesteps=4, real_spiking=False)
    _, report = validate_loihi_export_readiness(non_spiking, sample_input=ids)
    ids_found = {f["id"] for f in report["findings"]}
    assert "spiking_neurons_not_invoked" in ids_found, (
        "the default path does not spike; the validator must say so"
    )

    spiking = _convert("tiny-gpt2", timesteps=4, real_spiking=True)
    _, report = validate_loihi_export_readiness(spiking, sample_input=ids)
    invoked = [f for f in report["findings"] if f["id"] == "spiking_neurons_invoked"]
    assert invoked, f"expected spiking neurons to run; got {sorted(f['id'] for f in report['findings'])}"
    detail = invoked[0]["detail"]
    assert detail["invoked"] == detail["spiking_neuron_count"] > 0


def test_timestep_loop_is_not_a_no_op_when_spiking():
    """
    T must change the result.

    Without spiking the network is stateless, so running it T times and averaging returns
    the same logits at T times the cost — measured as bit-identical at T=1 vs T=8. With
    spiking, membrane state carries across timesteps and the outputs genuinely differ.
    """
    ids = torch.randint(0, 200, (1, 16))

    def logits_at(timesteps: int, real_spiking: bool) -> torch.Tensor:
        torch.manual_seed(0)
        model = _convert("tiny-gpt2", timesteps=timesteps, real_spiking=real_spiking)
        with torch.no_grad():
            return model(ids, use_cache=False).logits.clone()

    non_spiking_delta = (logits_at(1, False) - logits_at(8, False)).abs().max().item()
    spiking_delta = (logits_at(1, True) - logits_at(8, True)).abs().max().item()

    assert non_spiking_delta < 1e-5, (
        f"without spiking the T-loop should be a no-op up to float noise, got {non_spiking_delta:.2e}"
    )
    assert spiking_delta > 1e-3, (
        f"with spiking, T=1 and T=8 must differ; got {spiking_delta:.2e} — the membrane "
        "state is probably being reset every timestep again"
    )


def test_spiking_output_is_sparse_and_binary():
    """Spike trains must be binary and sparse, or nothing downstream is a real SNN."""
    model = _convert("tiny-gpt2", timesteps=4, real_spiking=True)
    ids = torch.randint(0, 200, (1, 16))
    report = measure_spikes(model, ids, use_cache=False)

    assert report.invoked_modules > 0, "no spiking neuron ran"
    assert report.total_spikes > 0, "spiking mode emitted no spikes at all"
    assert 0.0 < report.spike_mean < 1.0, f"implausible spike rate {report.spike_mean}"
    assert report.spike_frac_zero > 0.5, (
        f"spike train is not sparse (only {1 - report.spike_frac_zero:.1%} zeros)"
    )


def test_energy_projection_accounts_for_timesteps_and_dense_remainder():
    """
    The projection must not flatter the SNN.

    An early version compared attention-only SynOps against the whole model's MACs and
    produced a 1304x "advantage". Two facts have to survive: only spike-driven MACs
    become accumulates, and the dense remainder is paid once per timestep.
    """
    timesteps = 4
    model = _convert("tiny-gpt2", timesteps=timesteps, real_spiking=True)
    ids = torch.randint(0, 200, (1, 16))
    report = measure_spikes(model, ids, use_cache=False)

    assert report.timesteps == timesteps
    assert report.macs > 0
    assert 0 < report.macs_spike_replaced < report.macs, (
        "only part of the network is spike-driven; if this equals total MACs the "
        "accounting has stopped distinguishing them"
    )
    assert report.macs_remaining == pytest.approx(report.macs - report.macs_spike_replaced)

    # Energy must include the per-timestep dense cost.
    expected = timesteps * (
        report.synops * 0.9 + report.macs_remaining * 4.6
    )
    assert report.energy_snn_pj == pytest.approx(expected, rel=1e-6)
    assert report.energy_ann_pj == pytest.approx(report.macs * 4.6, rel=1e-6)
    assert any("projection" in n for n in report.notes), "the projection caveat must be reported"


@pytest.mark.parametrize("model_name", ["tiny-llama", "tiny-llama-gqa"])
@pytest.mark.parametrize("real_spiking", [False, True])
def test_llama_and_grouped_query_attention(model_name, real_spiking):
    """
    Llama-family conversion, including grouped-query attention.

    GQA (num_key_value_heads < num_attention_heads, as in SmolLM2-135M/360M) previously
    raised NotImplementedError. Exercises cached multi-turn generation, which is where
    the K/V head repetition interacts with the cache.
    """
    config = AutoConfig.from_pretrained(_model_path(model_name))
    model = _convert(model_name, timesteps=2, real_spiking=real_spiking)

    attns = [m for m in model.modules() if isinstance(m, SpikeAttention)]
    assert attns, "no SpikeAttention modules were installed"
    assert attns[0].num_kv_heads == config.num_key_value_heads
    assert attns[0].num_heads == config.num_attention_heads

    ids = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        for _ in range(4):
            out = model(ids, use_cache=True)
            assert torch.isfinite(out.logits).all(), "non-finite logits during generation"
            ids = torch.cat([ids, out.logits[:, -1:, :].argmax(-1)], dim=1)
    assert out.logits.shape[-1] == config.vocab_size


def test_default_path_is_unchanged_by_the_spiking_option():
    """The opt-in must stay opt-in: the default conversion still matches the ANN."""
    try:
        reference = AutoModelForCausalLM.from_pretrained(_model_path("tiny-gpt2"))
    except Exception as e:
        pytest.skip(f"Could not load the reference model: {e}")
    reference.eval()

    ids = torch.randint(0, 200, (1, 16))
    with torch.no_grad():
        expected = reference(ids).logits

    converted = _convert("tiny-gpt2", timesteps=4, real_spiking=False)
    with torch.no_grad():
        actual = converted(ids, use_cache=False).logits

    assert torch.allclose(expected, actual, atol=1e-4), (
        f"default conversion drifted from the reference: "
        f"max diff {(expected - actual).abs().max().item():.2e}"
    )
