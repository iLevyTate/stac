#!/usr/bin/env python3
"""
What would it take for STAC's SNN conversion to actually beat the dense ANN on energy?

`spike_metrics.py` answers "what is the projected energy of this model?", and reports that
the current architecture is ~7.6x *worse* than the ANN it converts. This script answers the
follow-up: is that fixable within this design, and if so, how?

The model
---------
For a linear layer whose *input* is a binary spike train at rate rho, every MAC becomes a
spike-driven accumulate, so SynOps = rho * MACs over the covered portion. Writing f for the
fraction of total MACs that are spike-driven and r = E_AC / E_MAC:

    E_ANN = M * E_MAC
    E_SNN = T * (rho * f * M * E_AC  +  (1 - f) * M * E_MAC)

    ratio = E_SNN / E_ANN = T * (rho * f * r + 1 - f)

Requiring ratio < 1 rearranges into the two useful forms:

    required coverage    f*    = (T - 1) / (T * (1 - rho * r))
    affordable timesteps T_max = 1 / (1 - f * (1 - rho * r))

This model is validated against spike_metrics.py in tests/test_energy_analysis.py: it
reproduces the measured E_SNN/E_ANN to within 0.5%.

The headline consequence: rho enters only through the product rho*r, and r ~= 0.196, so
sparsity moves the result by a few percent. **Coverage sets the timestep budget.** That
runs against the usual SNN intuition that sparsity is the lever.

Usage:
    python scripts/energy_analysis.py --model distilgpt2 --seq_len 512
    python scripts/energy_analysis.py --arch smollm2-1.7b --seq_len 2048
    python scripts/energy_analysis.py --scaling          # sweep every known architecture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn as nn

from spike_metrics import ENERGY_PER_AC_PJ, ENERGY_PER_MAC_PJ

R = ENERGY_PER_AC_PJ / ENERGY_PER_MAC_PJ  # accumulate cost relative to a MAC
DEFAULT_SPIKE_RATE = 0.094  # measured, tiny-gpt2 spiking mode; see docs/baselines/


# --------------------------------------------------------------------------------------
# Architectures, so a model can be analysed without downloading weights
# --------------------------------------------------------------------------------------
# mlp_macs_per_token is the whole block: GPT-2 is c_fc + c_proj = 8*d^2; Llama-style is
# gate + up + down = 3*d*intermediate.

ARCHS = {
    "distilgpt2":    dict(d=768,  layers=6,  vocab=50257, mlp=8 * 768 ** 2,      kv_ratio=1.0),
    "gpt2":          dict(d=768,  layers=12, vocab=50257, mlp=8 * 768 ** 2,      kv_ratio=1.0),
    "smollm2-135m":  dict(d=576,  layers=30, vocab=49152, mlp=3 * 576 * 1536,    kv_ratio=3 / 9),
    "smollm2-360m":  dict(d=960,  layers=32, vocab=49152, mlp=3 * 960 * 2560,    kv_ratio=5 / 15),
    "smollm2-1.7b":  dict(d=2048, layers=24, vocab=49152, mlp=3 * 2048 * 8192,   kv_ratio=1.0),
}


def macs_from_arch(spec: dict, seq_len: int, batch: int = 1) -> dict:
    d, layers, vocab = spec["d"], spec["layers"], spec["vocab"]
    n, kv = seq_len, spec["kv_ratio"]
    return {
        "lm_head":       float(d * vocab * n * batch),
        "mlp":           float(layers * spec["mlp"] * n * batch),
        "attn_qkv_proj": float(layers * (d * d + 2 * d * d * kv) * n * batch),
        "attn_out_proj": float(layers * d * d * n * batch),
        "attn_QK^T":     float(layers * d * n * n * batch),
        "attn_AV":       float(layers * d * n * n * batch),
    }


def classify(name: str, cls: str) -> str:
    if "lm_head" in name:
        return "lm_head"
    if "mlp" in name or "feed_forward" in name:
        return "mlp"
    if "c_attn" in name or any(k in name for k in ("q_proj", "k_proj", "v_proj")):
        return "attn_qkv_proj"
    if "c_proj" in name or "o_proj" in name:
        return "attn_out_proj"
    return f"other[{cls}]"


def macs_from_model(model, seq_len: int, batch: int = 1) -> dict:
    """Measured MAC breakdown from real loaded weights."""
    cats: dict[str, float] = {}
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if isinstance(mod, nn.Linear):
            macs = float(mod.in_features) * mod.out_features * seq_len * batch
        elif cls == "Conv1D" and hasattr(mod, "weight"):
            in_f, out_f = mod.weight.shape  # Conv1D stores [in, out]
            macs = float(in_f) * out_f * seq_len * batch
        else:
            continue
        key = classify(name, cls)
        cats[key] = cats.get(key, 0.0) + macs

    cfg = model.config
    layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", 0)
    hidden = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", 0)
    qk = float(layers) * hidden * seq_len * seq_len * batch
    cats["attn_QK^T"] = qk
    cats["attn_AV"] = qk
    return cats


# --------------------------------------------------------------------------------------
# Crossover algebra
# --------------------------------------------------------------------------------------

def required_coverage(timesteps: int, spike_rate: float) -> float:
    return (timesteps - 1) / (timesteps * (1.0 - spike_rate * R))


def max_timesteps(coverage: float, spike_rate: float) -> float:
    denom = 1.0 - coverage * (1.0 - spike_rate * R)
    return float("inf") if denom <= 0 else 1.0 / denom


def energy_ratio(coverage: float, timesteps: int, spike_rate: float) -> float:
    """E_SNN / E_ANN. Below 1.0 is an advantage."""
    return timesteps * (spike_rate * coverage * R + 1.0 - coverage)


SCENARIOS = [
    ("current: QK^T only",       ["attn_QK^T"]),
    ("+ AV matmul",              ["attn_QK^T", "attn_AV"]),
    ("+ MLP layers",             ["attn_QK^T", "attn_AV", "mlp"]),
    ("+ attention projections",  ["attn_QK^T", "attn_AV", "mlp", "attn_qkv_proj", "attn_out_proj"]),
    ("+ lm_head (everything)",   None),
]


def coverage_of(cats: dict, comps, total: float) -> float:
    if comps is None:
        return 1.0
    return sum(cats.get(c, 0.0) for c in comps) / total


def report(label: str, cats: dict, rho: float, timesteps, show_breakdown=True) -> list:
    total = sum(cats.values())
    if show_breakdown:
        print(f"\nMAC breakdown — {label} ({total:,.0f} total)")
        print("-" * 60)
        for k, v in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {k:24s} {v:16,.0f}  {100 * v / total:6.2f}%")

        print(f"\nCoverage required for parity (spike rate {rho})")
        print("-" * 60)
        for t in timesteps:
            print(f"  T={t:<4} needs f* >= {100 * required_coverage(t, rho):6.2f}%")

    print(f"\nWhat each coverage scenario buys — {label}")
    print("-" * 78)
    print(f"  {'scenario':<28} {'coverage':>9} {'T_max':>7}   {'ratio @ T=8':>13}")
    out = []
    for name, comps in SCENARIOS:
        f = coverage_of(cats, comps, total)
        tmax = max_timesteps(f, rho)
        ratio8 = energy_ratio(f, 8, rho)
        tmax_s = "inf" if tmax == float("inf") else f"{tmax:.2f}"
        verdict = f"{1/ratio8:.2f}x better" if ratio8 < 1 else f"{ratio8:.1f}x worse"
        print(f"  {name:<28} {100 * f:>8.2f}% {tmax_s:>7}   {verdict:>13}")
        out.append(dict(scenario=name, coverage=f, t_max=tmax, ratio_at_T8=ratio8))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="load real weights (e.g. distilgpt2)")
    ap.add_argument("--arch", choices=sorted(ARCHS), help="analyse from architecture spec, no download")
    ap.add_argument("--scaling", action="store_true", help="sweep every known architecture")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--spike_rate", type=float, default=DEFAULT_SPIKE_RATE)
    ap.add_argument("--timesteps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rho = args.spike_rate
    print("STAC energy-crossover analysis")
    print(f"seq_len={args.seq_len}  spike_rate={rho}  "
          f"E_MAC={ENERGY_PER_MAC_PJ} pJ  E_AC={ENERGY_PER_AC_PJ} pJ  r={R:.4f}")

    results = {}

    if args.scaling:
        print("\n" + "=" * 78)
        print("Scaling: lm_head is d*V (linear in width) while the body is L*d^2 (quadratic),")
        print("so lm_head's share — the block that is hardest to spike — shrinks as models grow.")
        print("=" * 78)
        print(f"\n  {'architecture':<16} {'lm_head':>9} {'body cov.':>10} "
              f"{'T_max body':>11} {'T_max MLP-only':>15}")
        print("  " + "-" * 66)
        for name in ["distilgpt2", "gpt2", "smollm2-135m", "smollm2-360m", "smollm2-1.7b"]:
            cats = macs_from_arch(ARCHS[name], args.seq_len)
            total = sum(cats.values())
            head_frac = cats["lm_head"] / total
            body = 1.0 - head_frac
            mlp_only = coverage_of(cats, SCENARIOS[2][1], total)
            print(f"  {name:<16} {100*head_frac:>8.2f}% {100*body:>9.2f}% "
                  f"{max_timesteps(body, rho):>11.2f} {max_timesteps(mlp_only, rho):>15.2f}")
            results[name] = dict(lm_head_frac=head_frac, body_coverage=body,
                                 t_max_body=max_timesteps(body, rho),
                                 t_max_mlp_only=max_timesteps(mlp_only, rho))
        print("\n  'body' = everything except lm_head. 'MLP-only' = the roadmap's graduated-spiking step.")

    if args.arch:
        cats = macs_from_arch(ARCHS[args.arch], args.seq_len)
        results[args.arch] = report(args.arch, cats, rho, args.timesteps)

    if args.model:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model)
        cats = macs_from_model(model, args.seq_len)
        results[args.model] = report(args.model, cats, rho, args.timesteps)

    if not (args.scaling or args.arch or args.model):
        ap.error("pass --model, --arch, or --scaling")

    if args.json:
        args.json.write_text(json.dumps(
            dict(seq_len=args.seq_len, spike_rate=rho, r=R, results=results), indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
