"""
Reproduction for docs/corrigendum-2026-07.md.

Runs the pre-fix STAC V1 neuron and layer code *verbatim* — copied from the original
stac-v1/stacv1.ipynb notebook, not imported from stac_v1/, so the fixes cannot mask the
defect — and reports how many spikes it emits.

Expected result: zero, out of 262,144 neuron-timesteps.

    python scripts/verify_v1_corrigendum.py

Exits non-zero if the historical behaviour fails to reproduce, so the corrigendum's central
claim stays checkable rather than asserted.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn

# Shipped V1 parameters, from the notebook's TEST_HPARAMS.
V_TH, V_RESET, V_REST = -50.0, -70.0, -65.0
TAU_M, TAU_W, A, B, DELTA_T = 20.0, 144.0, 4.0, 0.08, 2.0
L1_LAMBDA = 1e-5


class SurrogateSpikeFunction(torch.autograd.Function):
    """Verbatim from the notebook: unit-width Gaussian on a millivolt-scale argument."""

    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        spike_pseudo_grad = torch.exp(-(input_tensor**2) / 2.0) / math.sqrt(2 * math.pi)
        return grad_output * spike_pseudo_grad


surrogate_spike = SurrogateSpikeFunction.apply


class DLPFCAdExNeuron(nn.Module):
    """Verbatim from the notebook."""

    def __init__(self):
        super().__init__()
        self.tau_m = nn.Parameter(torch.tensor(TAU_M))
        self.tau_w = nn.Parameter(torch.tensor(TAU_W))
        self.a = nn.Parameter(torch.tensor(A))
        self.b = nn.Parameter(torch.tensor(B))
        self.V_th = nn.Parameter(torch.tensor(V_TH), requires_grad=False)
        self.V_reset = nn.Parameter(torch.tensor(V_RESET), requires_grad=False)
        self.V_rest = nn.Parameter(torch.tensor(V_REST), requires_grad=False)
        self.delta_T = nn.Parameter(torch.tensor(DELTA_T))

    def forward(self, input_current, V, w):
        dt = 1.0
        exp_term = torch.exp((V - self.V_th) / self.delta_T).clamp(max=50.0)
        dV = (dt / self.tau_m) * (
            -(V - self.V_rest) + self.delta_T * exp_term - w + input_current
        )
        V_new = V + dV
        dw = (dt / self.tau_w) * (self.a * (V - self.V_rest) - w)
        w_new = w + dw
        spike = surrogate_spike(V_new - self.V_th)
        V_final = torch.where(spike > 0.5, self.V_reset, V_new)
        w_final = w_new + self.b * spike
        return spike, V_final, w_final


def main() -> int:
    torch.manual_seed(0)

    batch, seq_len, hidden, width = 4, 128, 768, 512  # GPT-2 hidden size -> DLPFC width
    projection = nn.Linear(hidden, width)  # the notebook feeds the neuron directly
    neuron = DLPFCAdExNeuron()

    hidden_states = torch.randn(batch, seq_len, hidden)
    V = torch.full((batch, width), V_REST)
    w = torch.zeros(batch, width)

    spikes, current = [], None
    for t in range(seq_len):
        current = projection(hidden_states[:, t, :])
        spk, V, w = neuron(current, V, w)
        spikes.append(spk)
    spikes = torch.stack(spikes, dim=1)

    required = (1.0 + A) * (V_TH - V_REST)
    gap = V_TH - V.max().item()
    total = int(spikes.sum().item())
    l1 = L1_LAMBDA * spikes.abs().mean().item()

    x = torch.tensor(V.max().item() - V_TH)
    surrogate = (torch.exp(-(x**2) / 2.0) / math.sqrt(2 * math.pi)).item()

    print("STAC V1 pre-fix reproduction (docs/corrigendum-2026-07.md)")
    print("-" * 62)
    print(f"injected current      mean={current.mean():+.4f}  max={current.max():+.4f}")
    print(f"current to fire       (1+a)(V_th - V_rest) = {required:.1f}")
    print(f"membrane potential    max={V.max():+.4f}   threshold={V_TH:.1f}")
    print(f"gap to threshold      {gap:.2f} mV")
    print(f"spikes emitted        {total} of {spikes.numel():,} neuron-timesteps")
    print(f"L1 spike penalty      {l1:.8f}")
    print(f"surrogate gradient    {surrogate:.3e}  (fp32 min subnormal 1.401e-45)")
    print("-" * 62)

    ok = total == 0 and l1 == 0.0 and surrogate == 0.0
    print("REPRODUCED: the spiking pathway is inert" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
