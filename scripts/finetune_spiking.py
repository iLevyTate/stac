#!/usr/bin/env python3
"""
Spike-aware fine-tuning: the one path left after post-hoc conversion was shown to collapse.

`docs/coverage-quality.md` establishes that converting a *frozen* transformer to spikes
collapses it, and that no post-hoc knob (coverage, calibration, leak, timesteps) recovers
it. The remaining option is to let the weights move — train the converted network end to
end so it learns to be robust to the spike quantisation, rather than hoping a frozen ANN
survives it. This is the paper's "quantization-aware conversion" track.

The setup:

  1. Convert the model and extend spike coverage (spike_coverage.py), then calibrate.
  2. Train through the T-timestep spiking forward with backprop-through-time. Gradients
     reach the weights via SpikingJelly's surrogate gradient on each IF/LIF neuron, so no
     special machinery is needed beyond a normal optimiser — the spikes are differentiable
     in the backward pass by construction.
  3. Optionally distil from the original ANN (KL to teacher logits): standard for SNN
     conversion recovery, since the teacher gives a denser signal than hard labels.

Loss: L = (1 - alpha) * CE(student, targets) + alpha * T_kd^2 * KL(student || teacher).

WARNING — compute. Backprop-through-time holds T forward passes in the graph, so both time
and memory scale with T. This is a GPU workload. On CPU it is only useful for a tiny
proof-of-concept; the script prints per-step timing and, with --max_seconds, stops cleanly
before a wall-clock budget so a probe run cannot hang.

Usage (GPU, real run):
    python scripts/finetune_spiking.py --model distilgpt2 --timesteps 8 \
        --steps 2000 --distill --eval_every 200

Usage (CPU proof-of-concept):
    python scripts/finetune_spiking.py --model distilgpt2 --timesteps 2 \
        --seq_len 32 --steps 50 --max_seconds 600 --components mlp
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from spike_coverage import apply_spike_coverage, calibrate_thresholds  # noqa: E402
from smollm2_converter import (TemporalSpikeProcessor, calibrate_spike_attention,  # noqa: E402
                                simplified_conversion)

ALL_COMPONENTS = ["mlp", "attn_qkv_proj", "attn_out_proj", "lm_head"]


def load_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in data["text"] if t.strip())
    return tokenizer(text, return_tensors="pt").input_ids[0][:max_tokens]


def spiking_logits(model, chunk, timesteps):
    """Averaged logits over the T-timestep spiking forward, graph retained for backprop."""
    from spikingjelly.activation_based import functional

    inner = model.snn_model if isinstance(model, TemporalSpikeProcessor) else model
    functional.reset_net(inner)
    acc = None
    for _ in range(timesteps):
        out = inner(chunk, use_cache=False)
        lg = out.logits if hasattr(out, "logits") else out[0]
        acc = lg if acc is None else acc + lg
    return acc / timesteps


@torch.no_grad()
def perplexity(model, ids, window, stride, timesteps, device):
    from spikingjelly.activation_based import functional

    inner = model.snn_model if isinstance(model, TemporalSpikeProcessor) else model
    # Perplexity must not be measured with dropout active. Training call sites leave the
    # model in train() mode, so without this every reported number is dropout-inflated and
    # non-reproducible (a rerun gives different values, masking real improvements).
    was_training = inner.training
    inner.eval()
    stride = max(1, stride)  # seq_len==1 -> stride 0 -> range() ValueError
    nlls, counted, prev = [], 0, 0
    for begin in range(0, ids.size(0), stride):
        end = min(begin + window, ids.size(0))
        tl = end - prev
        if tl <= 0:
            continue
        chunk = ids[begin:end].unsqueeze(0).to(device)
        tgt = chunk.clone()
        tgt[:, :-tl] = -100
        if (tgt != -100).sum() <= 1:
            continue
        functional.reset_net(inner)
        acc = None
        for _ in range(timesteps):
            out = inner(chunk, use_cache=False)
            lg = out.logits if hasattr(out, "logits") else out[0]
            acc = lg if acc is None else acc + lg
        logits = (acc / timesteps).float()
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                               tgt[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        nlls.append(loss.item())
        counted += int((tgt[:, 1:] != -100).sum())
        prev = end
        if end == ids.size(0):
            break
    ppl = float(torch.exp(torch.tensor(sum(nlls) / max(counted, 1))))
    if was_training:
        inner.train()
    return ppl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--components", nargs="+", default=ALL_COMPONENTS,
                    help="which components to spike; fewer = cheaper. Damage localises to "
                         "the early blocks (docs/coverage-quality.md), so a targeted run is "
                         "a reasonable first experiment.")
    ap.add_argument("--distill", action="store_true", help="add KL-to-ANN-teacher loss")
    ap.add_argument("--alpha", type=float, default=0.5, help="distillation weight")
    ap.add_argument("--kd_temp", type=float, default=2.0)
    ap.add_argument("--train_tokens", type=int, default=100_000)
    ap.add_argument("--eval_tokens", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--max_seconds", type=float, default=None,
                    help="stop cleanly before this wall-clock budget (for bounded probes)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    train_ids = load_tokens(tok, args.train_tokens)
    eval_ids = train_ids[: args.eval_tokens]

    teacher = None
    if args.distill:
        teacher = AutoModelForCausalLM.from_pretrained(args.model).eval().to(args.device)

    conv = simplified_conversion(
        AutoModelForCausalLM.from_pretrained(args.model).eval(),
        args.timesteps, skip_gelu_replacement=True, real_spiking=False,
    )
    inner = conv.snn_model if isinstance(conv, TemporalSpikeProcessor) else conv
    info = apply_spike_coverage(inner, args.components, signed=True)
    conv.to(args.device)
    calib = [train_ids[: args.seq_len].unsqueeze(0).to(args.device)]
    calibrate_thresholds(conv, calib)
    calibrate_spike_attention(conv, calib)

    print(f"spike-aware fine-tuning — model={args.model} T={args.timesteps} "
          f"seq={args.seq_len} device={args.device}")
    print(f"components={args.components} wrapped={info['wrapped']} distill={args.distill}")

    base_ppl = perplexity(conv, eval_ids, args.seq_len, args.seq_len // 2,
                          args.timesteps, args.device)
    print(f"step 0: eval perplexity {base_ppl:.2f}")

    conv.train()
    params = [p for p in conv.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    n_windows = max(1, (train_ids.size(0) - 1) // args.seq_len)
    started = time.time()
    for step in range(1, args.steps + 1):
        w = (step - 1) % n_windows
        chunk = train_ids[w * args.seq_len:(w + 1) * args.seq_len].unsqueeze(0).to(args.device)
        if chunk.size(1) < 2:
            continue

        t0 = time.time()
        opt.zero_grad()
        logits = spiking_logits(conv, chunk, args.timesteps)
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                             chunk[:, 1:].reshape(-1))
        loss = ce
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(chunk).logits[:, :-1, :]
            # Reshape to (tokens, vocab) so batchmean divides by the token count, not by
            # batch=1. Otherwise KD is ~(seq_len-1)x out of scale versus the per-token CE and
            # the documented L = (1-alpha)*CE + alpha*T^2*KL trade-off does not hold.
            V = logits.size(-1)
            kd = F.kl_div(
                F.log_softmax(logits[:, :-1, :].reshape(-1, V) / args.kd_temp, dim=-1),
                F.softmax(t_logits.reshape(-1, V) / args.kd_temp, dim=-1),
                reduction="batchmean",
            ) * (args.kd_temp ** 2)
            loss = (1 - args.alpha) * ce + args.alpha * kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step == 1:
            print(f"step 1: loss {loss.item():.3f}  ({time.time()-t0:.1f}s/step)")
        if step % args.eval_every == 0:
            ppl = perplexity(conv, eval_ids, args.seq_len, args.seq_len // 2,
                             args.timesteps, args.device)
            conv.train()
            print(f"step {step}: loss {loss.item():.3f}  eval perplexity {ppl:.2f}  "
                  f"({(time.time()-started)/step:.1f}s/step avg)")

        if args.max_seconds and time.time() - started > args.max_seconds:
            print(f"stopping at step {step}: hit --max_seconds={args.max_seconds}. "
                  f"{(time.time()-started)/step:.1f}s/step on {args.device}.")
            break

    final_ppl = perplexity(conv, eval_ids, args.seq_len, args.seq_len // 2,
                           args.timesteps, args.device)
    print(f"final: eval perplexity {final_ppl:.2f}  (baseline was {base_ppl:.2f})")
    if args.save:
        torch.save(conv.state_dict(), args.save)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
