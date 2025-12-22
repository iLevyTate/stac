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
    Backward: Gaussian bump derivative approximation
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).to(dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = ctx.saved_tensors
        spike_pseudo_grad = torch.exp(-(input_tensor**2) / 2.0) / math.sqrt(2 * math.pi)
        return grad_output * spike_pseudo_grad


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
        exp_term = torch.exp((V - self.V_th) / self.delta_T).clamp(max=exp_clamp)
        dV = (dt / self.tau_m) * (-(V - self.V_rest) + self.delta_T * exp_term - w + input_current)
        V_new = V + dV

        dw = (dt / self.tau_w) * (self.a * (V - self.V_rest) - w)
        w_new = w + dw

        spike = surrogate_spike(V_new - self.V_th)
        V_final = torch.where(spike > 0.5, self.V_reset, V_new)
        w_final = w_new + self.b * spike
        return spike, V_final, w_final


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

        self.projection = nn.Linear(input_size, self.output_size)
        self.adex0 = DLPFCAdExNeuron(adex_params)

        self.recurrent_projections = nn.ModuleList(
            [nn.Linear(self.output_size, self.output_size) for _ in range(self.num_recurrent_layers)]
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
            current = self.projection(x_t)
            spk0, V0, w0 = self.adex0(current, V0, w0)
            spk_out = self.dropout(spk0)

            spk_rec_input = spk_out
            for i in range(self.num_recurrent_layers):
                rec_current = self.recurrent_projections[i](spk_rec_input)
                spk_rec, V_rec[i], w_rec[i] = self.recurrent_neurons[i](rec_current, V_rec[i], w_rec[i])
                spk_rec_input = self.dropout(spk_rec)

            spk_list.append(spk_rec_input.unsqueeze(1))

        return torch.cat(spk_list, dim=1)


class HyperdimensionalMemoryModule(nn.Module):
    """
    Encodes spike trains into a memory bias vector via a fixed random projection
    into a high-dimensional space, followed by a small MLP.
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
        pooled_spikes = torch.mean(spike_train, dim=1)
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
        memory_bias = self.memory_module(spk_trains)
        combined = spk_trains + memory_bias.unsqueeze(1)
        combined = self.dropout(self.layer_norm(combined))
        logits = self.lm_head(combined)
        return logits, spk_trains


