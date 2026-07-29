#!/usr/bin/env python3
"""
Does the model still work at the spike coverage the energy analysis requires?

`docs/energy-crossover.md` establishes that an energy advantage needs ~90%+ of MACs to be
spike-driven, versus the ~5% the current conversion reaches. It explicitly does not say
whether generation survives at that coverage. This measures it.

For each coverage level the script:

  1. converts a pretrained model,
  2. extends spike coverage to the named components (`spike_coverage.py`),
  3. calibrates per-layer thresholds on held-out text,
  4. measures word-level perplexity on WikiText-2, and
  5. measures achieved coverage and the projected energy ratio (`spike_metrics.py`).

The output is the coverage/quality/energy frontier: which coverage levels are affordable,
which are survivable, and whether those two sets overlap at all.

Perplexity is computed with a sliding window over concatenated text, the standard protocol,
so the numbers are comparable to published GPT-2 figures for the same corpus.

Usage:
    python scripts/coverage_quality_sweep.py --model distilgpt2 --timesteps 8
    python scripts/coverage_quality_sweep.py --model distilgpt2 --max_tokens 20000 --json sweep.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from spike_coverage import apply_spike_coverage, calibrate_thresholds  # noqa: E402
from spike_metrics import SpikeCounter  # noqa: E402
from smollm2_converter import (TemporalSpikeProcessor, calibrate_spike_attention,  # noqa: E402
                                simplified_conversion)

_ALL = ["mlp", "attn_qkv_proj", "attn_out_proj", "lm_head"]

# (label, components, spiking_attention). components=None means the unconverted ANN.
#
# The two families are kept separate so attention-spiking damage cannot be attributed to
# coverage. Historical note: SpikeAttention originally used a leaky hard-reset neuron at a
# fixed threshold of 0.1, which cannot rate-code (it 1-bit-quantises at an arbitrary cut);
# it now uses the same calibrated signed soft-reset IF encoding as SpikeLinear, with
# thresholds set by calibrate_spike_attention().
LEVELS = [
    ("ann-baseline",           None,                                        False),
    # existing spiking attention, no added coverage
    ("spiking-attn only",      [],                                          True),
    # calibrated coverage, attention left dense
    ("mlp",                    ["mlp"],                                     False),
    ("mlp+proj",               ["mlp", "attn_qkv_proj", "attn_out_proj"],   False),
    ("mlp+proj+head",          _ALL,                                        False),
    # both together
    ("all + spiking-attn",     _ALL,                                        True),
]


def load_wikitext(tokenizer, max_tokens: int) -> torch.Tensor:
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in data["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids[:max_tokens]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, window: int, stride: int, device: str) -> float:
    """
    Sliding-window word-level perplexity.

    Only the newly-revealed tokens of each window are scored, so no token is counted twice
    and every prediction has the longest context the window allows.

    Every window is evaluated with `use_cache=False` and an explicit cache reset.
    `TemporalSpikeProcessor.forward` defaults to `use_cache=True` and keeps `self.kv_cache`
    on the module, so consecutive independent windows would otherwise be prefixed with
    stale keys from the previous window. That silently drove perplexity to exactly the
    vocabulary size — a uniform distribution — which reads as "the model is destroyed"
    rather than "the harness is wrong".
    """
    nlls, counted = [], 0
    prev_end = 0
    for begin in range(0, ids.size(0), stride):
        end = min(begin + window, ids.size(0))
        target_len = end - prev_end
        if target_len <= 0:
            continue
        chunk = ids[begin:end].unsqueeze(0).to(device)
        targets = chunk.clone()
        targets[:, :-target_len] = -100  # score only the fresh tail
        if (targets != -100).sum() <= 1:
            continue

        if hasattr(model, "reset_cache"):
            model.reset_cache()
        out = model(chunk, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = targets[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n = int((shift_labels != -100).sum())
        nlls.append(loss.item())
        counted += n
        prev_end = end
        if end == ids.size(0):
            break
    return float(torch.exp(torch.tensor(sum(nlls) / max(counted, 1))))


def build(model_name: str, components, spiking: bool, timesteps: int, signed: bool, device: str):
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_name)
    base.eval()
    if components is None:  # unconverted ANN reference
        return base.to(device), {"wrapped": {}, "total": 0}

    converted = simplified_conversion(
        base, timesteps, skip_gelu_replacement=True, real_spiking=spiking
    )
    info = {"wrapped": {}, "total": 0}
    if components:
        inner = converted.snn_model if isinstance(converted, TemporalSpikeProcessor) else converted
        info = apply_spike_coverage(inner, components, signed=signed)
    converted.eval()
    return converted.to(device), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--calib_batches", type=int, default=4)
    ap.add_argument("--unsigned", action="store_true",
                    help="single LIF population; discards negative activations")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_wikitext(tok, args.max_tokens)
    calib = [ids[i * args.window:(i + 1) * args.window].unsqueeze(0).to(args.device)
             for i in range(args.calib_batches)]

    print(f"coverage/quality sweep — model={args.model} T={args.timesteps} "
          f"tokens={ids.numel():,} signed={not args.unsigned} device={args.device}")
    print("=" * 92)
    print(f"{'level':<20} {'coverage':>9} {'spike rate':>11} {'perplexity':>11} "
          f"{'vs ANN':>8} {'energy':>11}")
    print("-" * 92)

    rows, baseline_ppl = [], None
    for label, components, spiking in LEVELS:
        started = time.time()
        model, info = build(args.model, components, spiking, args.timesteps,
                            not args.unsigned, args.device)

        if info["total"]:
            calibrate_thresholds(model, calib)
        if spiking:
            calibrate_spike_attention(model, calib)

        ppl = perplexity(model, ids, args.window, args.stride, args.device)
        if baseline_ppl is None:
            baseline_ppl = ppl

        coverage = spike_rate = ratio = float("nan")
        if components is not None:
            probe = ids[: args.window].unsqueeze(0).to(args.device)
            with SpikeCounter(model) as counter:
                model(probe)
            rep = counter.report(seq_len=args.window, timesteps=args.timesteps)
            coverage = rep.macs_spike_replaced / rep.macs if rep.macs else 0.0
            spike_rate = rep.spike_mean
            ratio = rep.energy_snn_pj / rep.energy_ann_pj if rep.energy_ann_pj else float("nan")

        energy = "—" if ratio != ratio else (
            f"{1/ratio:.2f}x better" if ratio < 1 else f"{ratio:.1f}x worse")
        cov_s = "—" if coverage != coverage else f"{100*coverage:.1f}%"
        rate_s = "—" if spike_rate != spike_rate else f"{spike_rate:.3f}"
        print(f"{label:<20} {cov_s:>9} {rate_s:>11} {ppl:>11.2f} "
              f"{ppl/baseline_ppl:>7.2f}x {energy:>11}   ({time.time()-started:.0f}s)")

        rows.append(dict(level=label, coverage=coverage, spike_rate=spike_rate,
                         perplexity=ppl, ppl_ratio=ppl / baseline_ppl,
                         energy_ratio=ratio, wrapped=info["wrapped"]))
        del model

    print("-" * 92)
    print("coverage = fraction of MACs that are spike-driven; energy = projected E_SNN/E_ANN.")

    if args.json:
        args.json.write_text(json.dumps(dict(
            model=args.model, timesteps=args.timesteps, tokens=int(ids.numel()),
            signed=not args.unsigned, window=args.window, stride=args.stride,
            rows=rows), indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
