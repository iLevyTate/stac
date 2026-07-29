"""
Extend spike coverage beyond Q/K/V, so the coverage/quality trade-off can be measured.

`docs/energy-crossover.md` shows that the energy question is settled by *coverage* — the
fraction of MACs whose operands are binary spikes — and that the current conversion covers
only ~5% of them (the QK^T product). It also shows coverage above ~90% is where an
advantage lives. What it deliberately does not answer is whether the model still works at
that coverage.

This module supplies the missing knob. It wraps individual linear layers so their *input*
is a binary spike train, which is what makes the downstream MACs accumulates:

    SpikeLinear(x) = inner(spikes(x / threshold) * threshold)

Three details carry the whole thing:

**Rate coding.** `TemporalSpikeProcessor` re-presents the same input on each of its T
timesteps and resets neuron state once before the loop, so a LIF neuron's membrane
integrates across timesteps and its firing rate encodes input magnitude. T timesteps
therefore give roughly log2(T+1) bits of activation precision — T=8 is about 3 bits. That
quantisation, not the spiking itself, is what costs quality.

**Threshold balancing.** A LIF maps input magnitude to a firing rate that saturates at 1
once the input reaches the threshold. Leave the threshold at its default and activations
either saturate (every neuron fires every step, carrying no information) or never fire.
Thresholds are therefore calibrated per layer from a percentile of observed input
magnitude — the same idea as Diehl et al. (2015) weight/threshold balancing, which this
project already cites.

**Signed input.** Transformer activations are signed; a LIF only fires on positive drive,
so a naive wrap silently discards every negative component. Two opposing populations encode
sign at 2x the spikes. Accumulates are cheap, so this is close to free energetically —
but it is a real modelling choice and `signed=False` is available to measure its cost.

Scale folding: `spikes * threshold` is written explicitly for clarity in simulation. On
hardware the threshold folds into the downstream weights and the wire stays binary; the
MAC-to-accumulate accounting in `spike_metrics.py` is unaffected either way.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Component labels, matching scripts/energy_analysis.py so coverage figures line up.
COMPONENTS = ("mlp", "attn_qkv_proj", "attn_out_proj", "lm_head")


class SpikeLinear(nn.Module):
    """
    Make a linear layer spike-driven by binarising its input.

    `inner` may be an `nn.Linear` or a HuggingFace `Conv1D`; only its call signature is
    used, so the wrapper is agnostic to which.
    """

    def __init__(self, inner: nn.Module, *, threshold: float = 1.0, signed: bool = True,
                 tau: float | None = None):
        super().__init__()
        from spikingjelly.activation_based.neuron import IFNode, LIFNode

        self.inner = inner
        self.signed = bool(signed)
        self.tau = tau

        if tau is not None:
            # Leaky variant. A no-leak IF integrates any input bias without bound, so its
            # firing rate creeps upward over a long timestep window. A leak bounds that,
            # at the cost of systematically under-counting (charge is lost between spikes).
            # Measured on constant input in [0,1] over T=64, soft reset throughout:
            #     tau=2    corr 0.929  err 0.501  drift +0.003
            #     tau=8    corr 0.998  err 0.136  drift +0.018
            #     tau=32   corr 1.000  err 0.045  drift +0.026
            #     no leak  corr 1.000  err 0.015  drift +0.031
            # decay_input=False so the leak acts on the membrane, not on the input current.
            def make():
                return LIFNode(tau=float(tau), v_threshold=1.0, v_reset=None,
                               decay_input=False, detach_reset=True)
        else:
            def make():
                return IFNode(v_threshold=1.0, v_reset=None, detach_reset=True)
        # Integrate-and-fire with SOFT reset (`v_reset=None` subtracts the threshold on a
        # spike instead of clearing the membrane). This choice is not cosmetic — it is what
        # makes the rate code work, and it is what ANN->SNN conversion has used since
        # Diehl et al. (2015) / Rueckauer et al. (2017).
        #
        # Measured, input in [0,1] against threshold 1.0 over T=64:
        #     LIF, hard reset:  fires not at all      (the leak pulls sub-threshold input
        #                                              to a steady state below theta)
        #     IF,  hard reset:  corr 0.93, 29.5% err  (the excess above theta is discarded)
        #     IF,  soft reset:  corr 0.9999, 1.6% err
        #
        # Pinned by tests/test_spike_coverage.py::test_rate_code_converges.
        self.pos = make()
        self.neg = make() if signed else None
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        # Set while calibrating: pass through unchanged and record input statistics.
        self.calibrating = False
        self._observed: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            with torch.no_grad():
                # 99th percentile of |x|: the largest magnitude the rate code should be
                # able to represent without saturating. The max would be set by outliers.
                flat = x.detach().abs().flatten().float()
                if flat.numel() > 100_000:  # quantile() has a tensor-size limit
                    flat = flat[torch.randperm(flat.numel(), device=flat.device)[:100_000]]
                if flat.numel():
                    self._observed.append(torch.quantile(flat, 0.99).item())
            return self.inner(x)

        thr = self.threshold.clamp(min=1e-6)
        normalised = x / thr
        if self.signed:
            spikes = self.pos(normalised.clamp(min=0.0)) - self.neg((-normalised).clamp(min=0.0))
        else:
            spikes = self.pos(normalised.clamp(min=0.0))
        return self.inner(spikes * thr)

    def finalize_calibration(self) -> None:
        if self._observed:
            self.threshold.fill_(float(sum(self._observed) / len(self._observed)))
        self._observed = []
        self.calibrating = False


# --------------------------------------------------------------------------------------
# Selective application
# --------------------------------------------------------------------------------------

def _component_of(name: str) -> str | None:
    """Bucket a module path into the component whose coverage decision it belongs to."""
    if "lm_head" in name:
        return "lm_head"
    if "mlp" in name or "feed_forward" in name:
        return "mlp"
    if "c_attn" in name or any(k in name for k in (".q_proj", ".k_proj", ".v_proj")):
        return "attn_qkv_proj"
    if "c_proj" in name or ".o_proj" in name:
        return "attn_out_proj"
    return None


def _is_linear_like(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or type(module).__name__ == "Conv1D"


def apply_spike_coverage(
    model: nn.Module,
    components: Sequence[str],
    *,
    signed: bool = True,
    tau: float | None = None,
) -> dict:
    """
    Wrap every linear layer belonging to `components` in a `SpikeLinear`.

    Returns a summary describing what was wrapped. Wrapping nothing is reported rather
    than passing silently — a coverage sweep whose knob does nothing looks exactly like a
    model that is robust to spiking, which is the failure mode this guards against.
    """
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown component(s) {sorted(unknown)}; expected {COMPONENTS}")

    targets: list[tuple[str, nn.Module, str]] = []
    for name, module in model.named_modules():
        if not _is_linear_like(module):
            continue
        comp = _component_of(name)
        # attn_out_proj matching must not swallow the MLP's own c_proj on GPT-2, where
        # both are named "c_proj"; the mlp check above already claimed those.
        if comp in components:
            targets.append((name, module, comp))

    wrapped: dict[str, int] = {}
    for name, module, comp in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, SpikeLinear(module, signed=signed, tau=tau))
        wrapped[comp] = wrapped.get(comp, 0) + 1

    if not wrapped:
        logger.warning(
            "apply_spike_coverage(%s) wrapped nothing — the model exposes no matching "
            "linear layers, so measured coverage will not change.", list(components)
        )
    else:
        logger.info("Wrapped %d layer(s) as spike-driven: %s", len(targets), wrapped)
    return {"wrapped": wrapped, "total": len(targets), "signed": signed, "tau": tau}


def spike_linears(model: nn.Module) -> Iterable[SpikeLinear]:
    for module in model.modules():
        if isinstance(module, SpikeLinear):
            yield module


@torch.no_grad()
def calibrate_thresholds(model: nn.Module, batches: Iterable[torch.Tensor]) -> int:
    """
    Set every SpikeLinear's threshold from observed input magnitudes.

    Runs the model with spiking disabled, so the statistics come from the real activation
    distribution rather than from an already-degraded one.
    """
    layers = list(spike_linears(model))
    if not layers:
        return 0
    for layer in layers:
        layer.calibrating = True
    was_training = model.training
    model.eval()
    for batch in batches:
        model(batch)
    for layer in layers:
        layer.finalize_calibration()
    if was_training:
        model.train()
    logger.info("Calibrated %d SpikeLinear threshold(s)", len(layers))
    return len(layers)
