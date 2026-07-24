from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2Model


class SurrogateSpikeFunction(torch.autograd.Function):
    """
    Binary spike with a smooth surrogate gradient for backprop.

    Forward: step(x > 0)
    Backward: Gaussian bump derivative approximation, with width `sigma`.

    `sigma` must match the units of the input. The membrane potential here is in mV and
    sits ~10-20 mV from threshold, where a unit-width bump evaluates to exp(-100)~1e-44
    and underflows float32 to exactly zero — no gradient reaches the spiking layer at
    all. Sizing sigma to the membrane scale (AdEx's delta_T) keeps the surrogate usable.
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.sigma = float(sigma) if float(sigma) > 0 else 1.0
        return (input_tensor > 0).to(dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_tensor,) = ctx.saved_tensors
        sigma = ctx.sigma
        scaled = input_tensor / sigma
        spike_pseudo_grad = torch.exp(-(scaled**2) / 2.0) / (sigma * math.sqrt(2 * math.pi))
        # No gradient for the non-tensor `sigma` argument.
        return grad_output * spike_pseudo_grad, None


surrogate_spike = SurrogateSpikeFunction.apply


@dataclass(frozen=True)
class AdExParams:
    tau_m: float = 20.0
    tau_w: float = 144.0
    a: float = 4.0
    b: float = 0.08
    V_th: float = -50.0
    V_reset: float = -70.0
    V_rest: float = -65.0
    delta_T: float = 2.0


class DLPFCAdExNeuron(nn.Module):
    """
    Minimal AdEx-inspired spiking neuron with learnable dynamics parameters.

    Note: This is a simplified AdEx-style update as used in the original V1 notebook.
    """

    def __init__(self, params: AdExParams):
        super().__init__()
        self.tau_m = nn.Parameter(torch.tensor(params.tau_m, dtype=torch.float32))
        self.tau_w = nn.Parameter(torch.tensor(params.tau_w, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(params.a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(params.b, dtype=torch.float32))
        # Keep thresholds/rest potentials fixed (not trained) by default.
        self.V_th = nn.Parameter(torch.tensor(params.V_th, dtype=torch.float32), requires_grad=False)
        self.V_reset = nn.Parameter(torch.tensor(params.V_reset, dtype=torch.float32), requires_grad=False)
        self.V_rest = nn.Parameter(torch.tensor(params.V_rest, dtype=torch.float32), requires_grad=False)
        self.delta_T = nn.Parameter(torch.tensor(params.delta_T, dtype=torch.float32))

    def forward(
        self,
        input_current: torch.Tensor,
        V: torch.Tensor,
        w: torch.Tensor,
        *,
        dt: float = 1.0,
        exp_clamp: float = 50.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # These dynamics parameters are learnable and unconstrained, so training can drive
        # them to ~0 (or negative), producing division by ~0 / NaNs. Clamp to a small
        # positive floor in the forward pass to keep the update numerically stable.
        tau_m = self.tau_m.clamp(min=1e-3)
        tau_w = self.tau_w.clamp(min=1e-3)
        delta_T = self.delta_T.clamp(min=1e-3)

        exp_term = torch.exp((V - self.V_th) / delta_T).clamp(max=exp_clamp)
        dV = (dt / tau_m) * (-(V - self.V_rest) + delta_T * exp_term - w + input_current)
        V_new = V + dV

        dw = (dt / tau_w) * (self.a * (V - self.V_rest) - w)
        w_new = w + dw

        # Width the surrogate gradient in membrane units, so gradients survive the mV
        # scale of (V - V_th) instead of underflowing to zero.
        spike = surrogate_spike(V_new - self.V_th, float(delta_T.detach().item()))
        V_final = torch.where(spike > 0.5, self.V_reset, V_new)
        w_final = w_new + self.b * spike
        return spike, V_final, w_final


class CurrentDrive(nn.Module):
    """
    Maps a projection's output into the AdEx neuron's excitable current range.

    An AdEx neuron settles at V ~= V_rest + I, so it only ever spikes when the injected
    current is on the order of (V_th - V_rest) — 15 mV for the shipped parameters. A raw
    nn.Linear over transformer hidden states produces currents of order 1, which pins the
    membrane ~14 mV below threshold: the layer emitted exactly zero spikes for every
    input, the L1 spike penalty was identically zero, and (since the surrogate gradient
    vanishes that far from threshold) no gradient could ever correct it.

    Normalising per feature and rescaling by a learnable gain/offset centres the
    population's steady state on threshold, so roughly half the neurons are active and
    training can move the rest either way.
    """

    def __init__(self, size: int, *, drive: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        # Offset puts the steady state at threshold; gain spreads the population around it.
        self.gain = nn.Parameter(torch.tensor(float(drive) / 2.0, dtype=torch.float32))
        self.offset = nn.Parameter(torch.tensor(float(drive), dtype=torch.float32))

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.gain * self.norm(current) + self.offset


class DLPFCLayer(nn.Module):
    """
    Sequentially processes transformer hidden states with an AdEx spiking layer,
    optionally followed by recurrent spiking sublayers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        num_recurrent_layers: int,
        adex_params: AdExParams,
        dropout_prob: float,
    ):
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be > 0")
        if num_recurrent_layers < 0:
            raise ValueError("num_recurrent_layers must be >= 0")

        self.output_size = int(output_size)
        self.num_recurrent_layers = int(num_recurrent_layers)

        # Current needed to bring the membrane from rest to threshold. The adaptation
        # current settles at w = a * (V - V_rest), so it cancels a factor (1 + a) of the
        # injected current: the steady state is V_rest + I / (1 + a). Sizing the drive
        # accordingly is what actually puts the neuron at threshold.
        drive = float((1.0 + adex_params.a) * (adex_params.V_th - adex_params.V_rest))

        self.projection = nn.Linear(input_size, self.output_size)
        self.input_drive = CurrentDrive(self.output_size, drive=drive)
        self.adex0 = DLPFCAdExNeuron(adex_params)

        self.recurrent_projections = nn.ModuleList(
            [nn.Linear(self.output_size, self.output_size) for _ in range(self.num_recurrent_layers)]
        )
        self.recurrent_drives = nn.ModuleList(
            [CurrentDrive(self.output_size, drive=drive) for _ in range(self.num_recurrent_layers)]
        )
        self.recurrent_neurons = nn.ModuleList(
            [DLPFCAdExNeuron(adex_params) for _ in range(self.num_recurrent_layers)]
        )
        self.dropout = nn.Dropout(p=float(dropout_prob))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device

        V0 = torch.full((batch_size, self.output_size), float(self.adex0.V_reset.item()), device=device)
        w0 = torch.zeros((batch_size, self.output_size), device=device)

        V_rec = [
            torch.full((batch_size, self.output_size), float(n.V_reset.item()), device=device)
            for n in self.recurrent_neurons
        ]
        w_rec = [torch.zeros((batch_size, self.output_size), device=device) for _ in self.recurrent_neurons]

        spk_list = []
        for t in range(seq_len):
            x_t = hidden_states[:, t, :]
            current = self.input_drive(self.projection(x_t))
            spk0, V0, w0 = self.adex0(current, V0, w0)
            spk_out = self.dropout(spk0)

            spk_rec_input = spk_out
            for i in range(self.num_recurrent_layers):
                rec_current = self.recurrent_drives[i](self.recurrent_projections[i](spk_rec_input))
                spk_rec, V_rec[i], w_rec[i] = self.recurrent_neurons[i](rec_current, V_rec[i], w_rec[i])
                spk_rec_input = self.dropout(spk_rec)

            spk_list.append(spk_rec_input.unsqueeze(1))

        return torch.cat(spk_list, dim=1)


class HyperdimensionalMemoryModule(nn.Module):
    """
    Encodes spike trains into a memory bias via a fixed random projection into a
    high-dimensional space, followed by a small MLP.

    Pooling is *causal*: position t sees only spikes from positions <= t. Averaging over
    the whole sequence instead (the previous behaviour) mixed spikes from future tokens
    into every earlier position, which leaks the answer into a next-token training
    objective and makes reported training loss optimistic.
    """

    def __init__(self, input_dim: int, hdm_dim: int, output_dim: int):
        super().__init__()
        if input_dim <= 0 or hdm_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hdm_dim, and output_dim must be > 0")
        self.register_buffer("proj_matrix", torch.randn(input_dim, hdm_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hdm_dim, max(1, hdm_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hdm_dim // 2), output_dim),
        )

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_train: [batch, seq_len, input_dim]

        Returns:
            [batch, seq_len, output_dim] — a per-position memory bias built from the
            running (causal) mean of the spike train.
        """
        seq_len = spike_train.size(1)
        counts = torch.arange(
            1, seq_len + 1, device=spike_train.device, dtype=spike_train.dtype
        ).view(1, seq_len, 1)
        pooled_spikes = spike_train.cumsum(dim=1) / counts
        hdm_vector = pooled_spikes @ self.proj_matrix
        memory_bias = self.mlp(hdm_vector)
        return memory_bias


class DLPFCTransformer(nn.Module):
    """
    STAC V1 core model: pretrained GPT-2 backbone + spiking DLPFC layer + HEMM memory.
    """

    def __init__(
        self,
        *,
        model_name: str,
        dlpfc_output_size: int,
        num_recurrent_layers: int,
        adex_params: AdExParams,
        dropout_prob: float,
        hdm_dim: int,
    ):
        super().__init__()
        self.model_name = model_name
        self.gpt2 = GPT2Model.from_pretrained(model_name)

        gpt2_hidden_size = int(self.gpt2.config.hidden_size)
        self.dlpfc = DLPFCLayer(
            gpt2_hidden_size,
            int(dlpfc_output_size),
            num_recurrent_layers=int(num_recurrent_layers),
            adex_params=adex_params,
            dropout_prob=float(dropout_prob),
        )
        self.memory_module = HyperdimensionalMemoryModule(int(dlpfc_output_size), int(hdm_dim), int(dlpfc_output_size))
        self.layer_norm = nn.LayerNorm(int(dlpfc_output_size))
        self.dropout = nn.Dropout(p=float(dropout_prob))
        self.lm_head = nn.Linear(int(dlpfc_output_size), int(self.gpt2.config.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gpt_out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = gpt_out.last_hidden_state

        spk_trains = self.dlpfc(last_hidden)
        # memory_bias is per-position and causal, so it is added position-wise.
        memory_bias = self.memory_module(spk_trains)
        combined = spk_trains + memory_bias
        combined = self.dropout(self.layer_norm(combined))
        logits = self.lm_head(combined)
        return logits, spk_trains


