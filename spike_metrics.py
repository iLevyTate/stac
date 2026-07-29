"""
Spike-count telemetry and an operation-level energy projection for STAC.

The README describes energy figures as "theoretical projections derived from spike-count
analysis". This module is that analysis: it counts what the network actually does during
a forward pass and converts it into an energy estimate using published per-operation
costs.

What it measures
----------------
* **Spikes** — emitted by every stateful spiking neuron, via forward hooks.
* **Synaptic operations (SynOps)** — for an SNN, work is proportional to *spikes times
  fan-out*: a silent neuron costs nothing downstream. This is the standard
  neuromorphic cost metric.
* **MACs** — for the dense ANN baseline, multiply-accumulates in the linear/attention
  layers, which happen regardless of activation values.

The projection
--------------
Accumulate (AC) operations are markedly cheaper than multiply-accumulates (MAC) because
they skip the multiplier. Using Horowitz's widely-cited 45nm 32-bit figures (ISSCC 2014),
which essentially every SNN energy paper builds on:

    E_MAC ~= 4.6 pJ      (32-bit float multiply-accumulate)
    E_AC  ~= 0.9 pJ      (32-bit float accumulate)

The accounting has to be honest about two things that flatter an SNN if ignored:

1. **Only operations *downstream of a spiking neuron* become accumulates.** In this
   architecture the LIF neurons sit on Q/K/V, so the QK^T product is spike-driven; the
   projections, MLP and lm_head still consume real-valued activations and remain MACs.
   Comparing attention-only SynOps against the whole model's MACs would overstate the
   advantage by orders of magnitude.
2. **The SNN evaluates the network once per timestep.** A T-step simulation does T times
   the work, so the dense remainder is paid T times over.

    E_ANN = MACs_total * E_MAC
    E_SNN = T * (SynOps * E_AC + MACs_not_replaced * E_MAC)

IMPORTANT: this is an operation-count projection on a 45nm reference process, not a
measurement. It ignores memory movement (often the dominant real cost) and every detail
of an actual neuromorphic part, and it assumes event-driven hardware that skips silent
neurons — which simulation on a CPU/GPU does not do. Report it as a projection.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from loihi_constraints import _is_stateful_spiking_neuron

# Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014.
# 45nm, 32-bit. Override for a different process/precision.
ENERGY_PER_MAC_PJ = 4.6
ENERGY_PER_AC_PJ = 0.9


@dataclass
class SpikeReport:
    """Everything one measured forward pass tells us."""

    # Spiking activity
    total_spikes: float = 0.0
    total_neuron_timesteps: float = 0.0
    spike_mean: float = 0.0          # mean activation value over all spiking outputs
    spike_frac_zero: float = 1.0     # sparsity: fraction of exact zeros
    spiking_modules: int = 0
    invoked_modules: int = 0

    # Operation counts (per timestep unless noted)
    synops: float = 0.0              # spike-driven accumulates
    macs: float = 0.0                # dense MACs for the ANN baseline (single pass)
    macs_spike_replaced: float = 0.0 # MACs the spiking path turns into accumulates
    macs_remaining: float = 0.0      # MACs the SNN still pays, per timestep
    timesteps: int = 1

    # Energy projection (picojoules)
    energy_snn_pj: float = 0.0
    energy_ann_pj: float = 0.0
    energy_ratio: float = 0.0        # ann / snn; > 1 means the SNN is projected cheaper

    per_module: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"spikes={self.total_spikes:,.0f} rate={self.spike_mean:.4f} "
            f"sparsity={self.spike_frac_zero:.3f} T={self.timesteps} "
            f"synops={self.synops:,.0f} macs(ann)={self.macs:,.0f} "
            f"macs(snn/step)={self.macs_remaining:,.0f} | projected energy: SNN "
            f"{self.energy_snn_pj:,.0f} pJ vs ANN {self.energy_ann_pj:,.0f} pJ "
            f"({self.energy_ratio:.2f}x)"
        )


def _fan_out(module: nn.Module) -> int:
    """
    Downstream synapses per spike for the layer a neuron feeds.

    Approximated by the output width of the module's parent projection. Without a
    traced graph we cannot know the true fan-out, so we use the neuron's own feature
    width, which is the standard approximation in the SNN literature.
    """
    for attr in ("out_features", "embed_dim", "hidden_size"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def count_macs(model: nn.Module, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
    """
    Dense multiply-accumulates for one forward pass.

    Counts nn.Linear and HuggingFace Conv1D (GPT-2's linear-like layer), plus the two
    batched matmuls inside each attention block. Embedding lookups are excluded: they are
    table reads, not arithmetic.

    Returns {"total", "spike_replaceable"}. `spike_replaceable` is the QK^T product of
    any SpikeAttention running in spiking mode — the only arithmetic here whose operands
    are binary spikes and which therefore becomes accumulation on neuromorphic hardware.
    Everything else consumes real-valued activations and stays a MAC.
    """
    total = 0.0
    spike_replaceable = 0.0
    for _name, module in model.named_modules():
        cls = type(module).__name__
        if cls == "SpikeLinear":
            # spike_coverage.SpikeLinear binarises its input, so the wrapped layer's MACs
            # become spike-driven accumulates. Only credit the replacement here: the inner
            # layer is a submodule and named_modules() visits it separately, where it is
            # added to `total`. Counting it in both places would double the denominator.
            inner = getattr(module, "inner", None)
            if isinstance(inner, nn.Linear):
                spike_replaceable += float(inner.in_features) * inner.out_features * seq_len * batch_size
            elif type(inner).__name__ == "Conv1D" and hasattr(inner, "weight"):
                in_f, out_f = inner.weight.shape
                spike_replaceable += float(in_f) * float(out_f) * seq_len * batch_size
        elif isinstance(module, nn.Linear):
            total += float(module.in_features) * float(module.out_features) * seq_len * batch_size
        elif cls == "Conv1D" and hasattr(module, "weight"):
            in_f, out_f = module.weight.shape  # Conv1D stores [in, out]
            total += float(in_f) * float(out_f) * seq_len * batch_size
        elif cls in ("QuantizedLinearLike",) and hasattr(module, "qweight"):
            a, b = module.qweight.shape[-2], module.qweight.shape[-1]
            total += float(a) * float(b) * seq_len * batch_size
        elif cls == "SpikeAttention":
            heads = getattr(module, "num_heads", 1)
            head_dim = getattr(module, "head_dim", 1)
            qk = float(heads) * head_dim * seq_len * seq_len * batch_size
            av = qk  # (attn) x V has the same shape cost
            total += qk + av
            if getattr(module, "spiking", False):
                # Only QK^T has binary operands on both sides. attn x V does not: the
                # attention scores are real-valued even in spiking mode.
                spike_replaceable += qk
    return {"total": total, "spike_replaceable": spike_replaceable}


class SpikeCounter:
    """
    Context manager that counts spikes emitted during forward passes.

    Hooks are attached to module objects (not resolved by name) so a wrapped model works:
    names differ between the wrapper's namespace and the inner model's.

        with SpikeCounter(model) as counter:
            model(input_ids)
        report = counter.report(seq_len=input_ids.shape[1])
    """

    def __init__(self, model: nn.Module, *, include: Optional[List[nn.Module]] = None):
        self.model = model
        inner = getattr(model, "snn_model", None)
        self._search_root = inner if isinstance(inner, nn.Module) else model
        if include is not None:
            self._neurons = [(f"module_{i}", m) for i, m in enumerate(include)]
        else:
            self._neurons = [
                (name, module)
                for name, module in self._search_root.named_modules()
                if _is_stateful_spiking_neuron(module)
            ]
        self._handles: List[Any] = []
        self._stats: Dict[str, Dict[str, float]] = {}

    def _make_hook(self, name: str, module: nn.Module):
        def hook(_mod, _inp, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            with torch.no_grad():
                detached = tensor.detach()
                entry = self._stats.setdefault(
                    name, {"spikes": 0.0, "elements": 0.0, "sum": 0.0, "zeros": 0.0,
                           "calls": 0.0, "fan_out": float(_fan_out(module))}
                )
                entry["spikes"] += float((detached > 0).sum().item())
                entry["elements"] += float(detached.numel())
                entry["sum"] += float(detached.sum().item())
                entry["zeros"] += float((detached == 0).sum().item())
                entry["calls"] += 1.0
        return hook

    def __enter__(self) -> "SpikeCounter":
        for name, module in self._neurons:
            self._handles.append(module.register_forward_hook(self._make_hook(name, module)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def report(self, *, seq_len: int, batch_size: int = 1, timesteps: Optional[int] = None) -> SpikeReport:
        """Build a report; `seq_len` is needed for the dense MAC baseline."""
        report = SpikeReport()
        if timesteps is None:
            timesteps = int(getattr(self.model, "T", 1) or 1)
        report.timesteps = max(1, int(timesteps))
        report.spiking_modules = len(self._neurons)
        report.invoked_modules = len(self._stats)

        total_spikes = sum(e["spikes"] for e in self._stats.values())
        total_elements = sum(e["elements"] for e in self._stats.values())
        total_sum = sum(e["sum"] for e in self._stats.values())
        total_zeros = sum(e["zeros"] for e in self._stats.values())

        report.total_spikes = total_spikes
        report.total_neuron_timesteps = total_elements
        report.spike_mean = (total_sum / total_elements) if total_elements else 0.0
        report.spike_frac_zero = (total_zeros / total_elements) if total_elements else 1.0
        # Spikes were counted across ALL timesteps of the measured pass; report SynOps
        # per timestep so it composes with the explicit timestep factor below.
        total_synops = sum(e["spikes"] * e["fan_out"] for e in self._stats.values())
        report.synops = total_synops / report.timesteps

        mac_counts = count_macs(self._search_root, seq_len=seq_len, batch_size=batch_size)
        report.macs = mac_counts["total"]
        report.macs_spike_replaced = mac_counts["spike_replaceable"]
        report.macs_remaining = max(0.0, report.macs - report.macs_spike_replaced)

        # The ANN runs once. The SNN runs the whole network once per timestep, paying the
        # un-replaced dense MACs every time; only the spike-driven part becomes cheap
        # accumulates. Ignoring the timestep factor is how an SNN gets an implausible
        # advantage on paper.
        report.energy_ann_pj = report.macs * ENERGY_PER_MAC_PJ
        report.energy_snn_pj = report.timesteps * (
            report.synops * ENERGY_PER_AC_PJ + report.macs_remaining * ENERGY_PER_MAC_PJ
        )
        report.energy_ratio = (
            report.energy_ann_pj / report.energy_snn_pj if report.energy_snn_pj > 0 else 0.0
        )

        report.per_module = {
            name: {
                "spikes": e["spikes"],
                "rate": (e["spikes"] / e["elements"]) if e["elements"] else 0.0,
                "synops": e["spikes"] * e["fan_out"],
                "calls": e["calls"],
            }
            for name, e in sorted(self._stats.items())
        }

        if report.invoked_modules == 0:
            report.notes.append(
                f"None of the {report.spiking_modules} spiking neurons ran: the model "
                "performs no spiking computation, so no energy advantage can be claimed."
            )
        elif report.invoked_modules < report.spiking_modules:
            report.notes.append(
                f"Only {report.invoked_modules}/{report.spiking_modules} spiking neurons ran."
            )
        if report.energy_ratio and report.energy_ratio < 1.0:
            report.notes.append(
                f"Projected energy is {1.0 / report.energy_ratio:.2f}x WORSE than the dense "
                f"ANN: only {report.macs_spike_replaced:,.0f} of {report.macs:,.0f} MACs are "
                f"spike-driven, and the remainder is paid on each of {report.timesteps} "
                "timesteps. An advantage requires spiking activations throughout the "
                "network, not just on Q/K/V."
            )
        report.notes.append(
            "Operation-count projection on a 45nm reference process (Horowitz ISSCC 2014); "
            "not a hardware measurement. Excludes memory movement, and assumes event-driven "
            "hardware that skips silent neurons."
        )
        return report


def measure_spikes(model: nn.Module, input_ids: torch.Tensor, **forward_kwargs) -> SpikeReport:
    """Convenience wrapper: run one forward pass under a SpikeCounter and report."""
    was_training = model.training
    model.eval()
    with SpikeCounter(model) as counter:
        with torch.no_grad():
            model(input_ids, **forward_kwargs)
    if was_training:
        model.train()
    return counter.report(
        seq_len=int(input_ids.shape[-1]),
        batch_size=int(input_ids.shape[0]),
        timesteps=int(getattr(model, "T", 1) or 1),
    )
