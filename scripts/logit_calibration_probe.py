#!/usr/bin/env python3
"""
Is the conversion damage information loss, or logit-scale miscalibration?

`docs/coverage-quality.md` reports ~41x worse perplexity after conversion. Perplexity is
exquisitely sensitive to logit *scale* -- a network whose logits have the right shape but
the wrong magnitude reads as catastrophically bad while its actual predictions are intact,
because argmax is invariant to temperature and cross-entropy is not.

That distinction decides what the result means:

* if top-1 agreement with the ANN is high and one fitted scalar recovers most of the
  perplexity, the damage is calibration and the pipeline is close to working
  (`TemporalSpikeProcessor` already carries a `logit_scale` parameter for exactly this,
  initialised to 1.0 and never fitted);
* if agreement is low, the information really is destroyed and no rescaling will help.

This measures both on paired ANN/SNN logits over identical tokens.

Note: `simplified_conversion` mutates the model in place, so the ANN reference must be a
separately loaded copy. Reusing the pre-conversion handle silently benchmarks the spiking
model against itself.

Usage:
    python scripts/logit_calibration_probe.py --model distilgpt2 --timesteps 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from spike_coverage import apply_spike_coverage, calibrate_thresholds  # noqa: E402
from smollm2_converter import (TemporalSpikeProcessor, calibrate_spike_attention,  # noqa: E402
                                simplified_conversion)

ALL_COMPONENTS = ["mlp", "attn_qkv_proj", "attn_out_proj", "lm_head"]
SCALES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200]


def load_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in data["text"] if t.strip())
    return tokenizer(text, return_tensors="pt").input_ids[0][:max_tokens]


@torch.no_grad()
def paired_logits(ann, snn, ids, window, stride, timesteps, device):
    """Collect (ann_logits, snn_logits, targets) over identical evaluation windows."""
    from spikingjelly.activation_based import functional

    ann_out, snn_out, targets = [], [], []
    prev_end = 0
    for begin in range(0, ids.size(0), stride):
        end = min(begin + window, ids.size(0))
        target_len = end - prev_end
        if target_len <= 0:
            continue
        chunk = ids[begin:end].unsqueeze(0).to(device)
        tgt = chunk.clone()
        tgt[:, :-target_len] = -100
        if (tgt != -100).sum() <= 1:
            continue

        a = ann(chunk, use_cache=False).logits

        inner = snn.snn_model if isinstance(snn, TemporalSpikeProcessor) else snn
        functional.reset_net(inner)
        acc = None
        for _ in range(timesteps):
            out = inner(chunk, use_cache=False)
            lg = out.logits if hasattr(out, "logits") else out[0]
            acc = lg if acc is None else acc + lg
        s = acc / timesteps

        vocab = a.size(-1)
        ann_out.append(a[:, :-1, :].reshape(-1, vocab))
        snn_out.append(s[:, :-1, :].reshape(-1, vocab))
        targets.append(tgt[:, 1:].reshape(-1))
        prev_end = end
        if end == ids.size(0):
            break

    keep = torch.cat(targets) != -100
    return (torch.cat(ann_out)[keep].float(),
            torch.cat(snn_out)[keep].float(),
            torch.cat(targets)[keep])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--tau", type=float, default=None, help="leak constant; omit for no leak")
    ap.add_argument("--spiking_attn", action="store_true")
    ap.add_argument("--reference", choices=["ann", "converted"], default="ann",
                    help="'ann' compares against the untouched model, so the number includes "
                         "every cost of conversion. 'converted' compares against the same "
                         "model after simplified_conversion but WITHOUT spike coverage, "
                         "isolating the damage spiking alone does. Use 'converted' for "
                         "Llama-family models, where SpikeAttention drops RoPE and that "
                         "loss would otherwise be attributed to spiking.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_tokens(tok, args.max_tokens)

    # Independent loads throughout: simplified_conversion mutates in place.
    if args.reference == "ann":
        ann = AutoModelForCausalLM.from_pretrained(args.model).eval().to(args.device)
    else:
        # Same architectural transforms, no spike coverage. Isolates the cost of spiking
        # from the cost of conversion (activation swap, RoPE loss, normalisation changes).
        ref_base = AutoModelForCausalLM.from_pretrained(args.model).eval()
        ref = simplified_conversion(ref_base, args.timesteps, skip_gelu_replacement=True,
                                    real_spiking=args.spiking_attn)
        ann = (ref.snn_model if isinstance(ref, TemporalSpikeProcessor) else ref)
        ann.eval().to(args.device)

    to_convert = AutoModelForCausalLM.from_pretrained(args.model).eval()

    snn = simplified_conversion(to_convert, args.timesteps, skip_gelu_replacement=True,
                                real_spiking=args.spiking_attn)
    inner = snn.snn_model if isinstance(snn, TemporalSpikeProcessor) else snn
    apply_spike_coverage(inner, ALL_COMPONENTS, signed=True, tau=args.tau)
    snn.eval().to(args.device)
    calib_batch = [ids[:args.window].unsqueeze(0).to(args.device)]
    calibrate_thresholds(snn, calib_batch)
    if args.spiking_attn:
        calibrate_spike_attention(snn, calib_batch)

    a, s, tgt = paired_logits(ann, snn, ids, args.window, args.stride, args.timesteps, args.device)

    cosine = float(F.cosine_similarity(a, s, dim=-1).mean())
    agree = float((a.argmax(-1) == s.argmax(-1)).float().mean())
    ann_acc = float((a.argmax(-1) == tgt).float().mean())
    snn_acc = float((s.argmax(-1) == tgt).float().mean())

    ce = lambda z: float(F.cross_entropy(z, tgt))
    ppl = lambda z: float(torch.exp(torch.tensor(ce(z))))
    ann_ppl, snn_ppl = ppl(a), ppl(s)
    best_scale = min(SCALES, key=lambda k: ce(s * k))
    scaled_ppl = ppl(s * best_scale)

    ref_label = "ANN" if args.reference == "ann" else "REF"
    print(f"model={args.model}  T={args.timesteps}  tau={args.tau}  "
          f"spiking_attn={args.spiking_attn}  reference={args.reference}  "
          f"tokens={int(tgt.numel())}")
    if args.reference == "converted":
        print("  reference = converted-but-not-spiking (isolates spiking from conversion)")
    print("-" * 70)
    print(f"  {ref_label} logit std                {a.std():.4f}")
    print(f"  SNN logit std                {s.std():.4f}   (scale ratio {a.std()/s.std():.3f})")
    print(f"  cosine({ref_label}, SNN) per token   {cosine:.4f}")
    print(f"  top-1 agreement with {ref_label}     {agree:.4f}")
    print(f"  {ref_label} next-token accuracy      {ann_acc:.4f}")
    print(f"  SNN next-token accuracy      {snn_acc:.4f}")
    print("-" * 70)
    print(f"  perplexity  {ref_label}              {ann_ppl:>10.2f}")
    print(f"  perplexity  SNN as-is        {snn_ppl:>10.2f}   ({snn_ppl/ann_ppl:.1f}x)")
    print(f"  perplexity  SNN best scale   {scaled_ppl:>10.2f}   ({scaled_ppl/ann_ppl:.1f}x)"
          f"  at scale={best_scale}")
    print("-" * 70)
    recovered = (snn_ppl - scaled_ppl) / max(snn_ppl - ann_ppl, 1e-9)
    print(f"  fraction of the gap closed by one scalar: {100*recovered:.1f}%")

    if args.json:
        args.json.write_text(json.dumps(dict(
            model=args.model, timesteps=args.timesteps, tau=args.tau,
            spiking_attn=args.spiking_attn, reference=args.reference, tokens=int(tgt.numel()),
            cosine=cosine, top1_agreement=agree, ann_accuracy=ann_acc, snn_accuracy=snn_acc,
            ann_ppl=ann_ppl, snn_ppl=snn_ppl, scaled_ppl=scaled_ppl,
            best_scale=best_scale, gap_closed=recovered), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
