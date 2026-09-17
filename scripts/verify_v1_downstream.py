"""
Second reproduction for docs/corrigendum-2026-07.md: how far the V1 defect propagated.

`verify_v1_corrigendum.py` shows the spiking layer emitted nothing. This script shows what
that did to the rest of the model, because the answer turned out to be worse than the
corrigendum first stated.

The shipped `DLPFCLayer` returns only spike tensors; there is no residual path carrying the
GPT-2 hidden state past it. So when the spike train is uniformly zero:

  * the HEMM pools zeros, projects zeros, and returns its MLP's bias path — one constant
    vector, the same for every position and every input;
  * `combined = spk_trains + memory_bias` is therefore that constant, so the head sees the
    same input no matter what was tokenized;
  * and the only gradient path back to the backbone runs through the spiking layer, whose
    surrogate is exactly zero, so the backbone's gradient is exactly zero even though it sits
    in the optimizer.

The layer and neuron code below is the shipped notebook's, imported from the first script so
the two reproductions cannot drift apart. A small random tensor stands in for the GPT-2
hidden states: the question is structural, and what matters is that *nothing* upstream of the
neurons receives gradient.

    python scripts/verify_v1_downstream.py

Exits non-zero if the behaviour fails to reproduce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_v1_corrigendum import DLPFCAdExNeuron, V_REST  # noqa: E402

BATCH, SEQ, HIDDEN, WIDTH, HDM, VOCAB = 2, 24, 768, 128, 256, 1000


def build():
    torch.manual_seed(0)
    return {
        "projection": nn.Linear(HIDDEN, WIDTH),  # stands for everything upstream
        "neuron": DLPFCAdExNeuron(),
        "proj_matrix": torch.randn(WIDTH, HDM),  # HEMM's fixed random projection
        "mlp": nn.Sequential(nn.Linear(HDM, HDM // 2), nn.ReLU(), nn.Linear(HDM // 2, WIDTH)),
        "layer_norm": nn.LayerNorm(WIDTH),
        "lm_head": nn.Linear(WIDTH, VOCAB),
    }


def forward(m, x):
    V = torch.full((BATCH, WIDTH), V_REST)
    w = torch.zeros(BATCH, WIDTH)
    spikes = []
    for t in range(SEQ):
        spk, V, w = m["neuron"](m["projection"](x[:, t, :]), V, w)
        spikes.append(spk)
    spk_trains = torch.stack(spikes, dim=1)

    # Shipped HEMM: mean over the whole sequence, then broadcast back over positions.
    pooled = torch.mean(spk_trains, dim=1)
    memory_bias = m["mlp"](pooled @ m["proj_matrix"]).unsqueeze(1)

    combined = spk_trains + memory_bias
    logits = m["lm_head"](m["layer_norm"](combined))
    return logits, spk_trains, combined


def main() -> int:
    m = build()
    x1 = torch.randn(BATCH, SEQ, HIDDEN)
    x2 = torch.randn(BATCH, SEQ, HIDDEN) * 7.0 + 3.0  # a deliberately unlike input

    logits1, spikes1, combined1 = forward(m, x1)
    logits2, _, _ = forward(m, x2)

    logit_gap = (logits1 - logits2).abs().max().item()
    spread = (combined1 - combined1[0, 0]).abs().max().item()
    distinct = len(set(logits1.argmax(-1).flatten().tolist()))

    loss = nn.functional.cross_entropy(
        logits1.reshape(-1, VOCAB), torch.randint(0, VOCAB, (BATCH * SEQ,))
    )
    loss.backward()

    def gnorm(p):
        return 0.0 if p.grad is None else p.grad.norm().item()

    upstream = gnorm(m["projection"].weight)
    adex = gnorm(m["neuron"].tau_m)
    hemm_w = gnorm(m["mlp"][0].weight)
    hemm_b = gnorm(m["mlp"][0].bias)
    head = gnorm(m["lm_head"].weight)

    print("STAC V1 pre-fix reproduction, part 2: downstream effect")
    print("-" * 62)
    print(f"spikes emitted             {int(spikes1.sum())} of {spikes1.numel():,}")
    print(f"logit gap between two      {logit_gap:.3e}   (identical output)")
    print( "  completely unlike inputs")
    print(f"spread of the head's input {spread:.3e}   (one constant vector)")
    print( "  over all positions/rows")
    print(f"distinct predicted tokens  {distinct}")
    print("-" * 62)
    print(f"grad norm, upstream proj   {upstream:.3e}   (the backbone's path)")
    print(f"grad norm, AdEx tau_m      {adex:.3e}")
    print(f"grad norm, HEMM mlp weight {hemm_w:.3e}")
    print(f"grad norm, HEMM mlp bias   {hemm_b:.3e}   (trains: fed by nothing)")
    print(f"grad norm, lm_head weight  {head:.3e}   (trains: fed a constant)")
    print("-" * 62)

    ok = (
        int(spikes1.sum()) == 0
        and logit_gap == 0.0
        and spread == 0.0
        and distinct == 1
        and upstream == 0.0
        and adex == 0.0
        and hemm_w == 0.0
        and hemm_b > 0.0
        and head > 0.0
    )
    print(
        "REPRODUCED: as released, V1 was a constant predictor and the backbone got no gradient"
        if ok
        else "NOT REPRODUCED"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
