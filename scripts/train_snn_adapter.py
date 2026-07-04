#!/usr/bin/env python3
"""
Distill an ANN teacher into the converted SNN student using a tiny LoRA adapter.

This is intended to improve ANN↔SNN parity (logits + multi-turn) while keeping transformer
semantics as intact as possible.

Evidence policy: do not claim improvements unless parity tests are run and logged under local/test_evidence/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Allow running this file directly (python scripts/train_snn_adapter.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from smollm2_converter import replace_gelu_with_relu, simplified_conversion
from smollm2_converter import SpikeLayerNorm


@dataclass
class TrainReport:
    timestamp_utc: str
    model_name: str
    device: str
    timesteps: int
    steps: int
    lr: float
    batch_size: int
    loss_type: str
    avg_loss: float
    out_dir: str


def _default_prompts() -> List[str]:
    # Small, mixed set including multi-turn formatted text to stress state/caching behavior.
    return [
        "The capital of France is",
        "2+2=",
        "In one sentence, explain gravity:",
        "User: My name is Alice.\nAssistant: Hello Alice.\nUser: What is my name?\nAssistant:",
        "User: I have 3 dogs and 2 cats.\nAssistant: OK.\nUser: How many pets do I have?\nAssistant:",
        "User: The capital of Japan is Tokyo.\nAssistant: OK.\nUser: What is the capital of Japan?\nAssistant:",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill ANN teacher into SNN student with LoRA adapters")
    p.add_argument("--model_name", type=str, default="distilgpt2")
    p.add_argument("--timesteps", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--out_dir", type=str, default="local/adapters/distill")
    p.add_argument("--loss_type", choices=["mse", "kl", "ce_teacher", "kl_ce"], default="kl")
    p.add_argument("--last_token_only", action="store_true",
                   help="If set, distill only the last-token distribution (recommended for parity).")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Distillation temperature for KL (default: 2.0)")
    p.add_argument("--train_spike_layernorm", action="store_true",
                   help="Also train SpikeLayerNorm affine params (small) in addition to LoRA.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Teacher
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student (converted)
    base = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    base = replace_gelu_with_relu(base)
    student_wrapper = simplified_conversion(base, timesteps=args.timesteps).to(device)
    student_wrapper.train()

    # LoRA adapters
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise RuntimeError("peft is required for this training loop. Install requirements.txt.") from e

    # Target common projection modules. After conversion, SpikeAttention exposes
    # q_proj/k_proj/v_proj/o_proj (nn.Linear), and the GPT-2 MLP keeps its Conv1D
    # c_fc/c_proj — PEFT handles both nn.Linear and Conv1D targets. Names not present in
    # a given backbone are simply ignored by PEFT.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj"],
    )
    # Apply LoRA to the *inner* HF CausalLM model to satisfy PEFT expectations.
    inner = getattr(student_wrapper, "snn_model", None)
    if inner is None:
        raise RuntimeError("Converted student wrapper missing snn_model attribute")
    inner = get_peft_model(inner, lora_cfg)
    inner.train()
    student_wrapper.snn_model = inner
    student = student_wrapper

    if args.train_spike_layernorm:
        # Enable training for SpikeLayerNorm affine params (weight/bias), which can strongly affect logit parity.
        for m in student.modules():
            if isinstance(m, SpikeLayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    # Only optimize parameters that actually require grad (LoRA adapter params, the
    # wrapper logit_scale, and optionally SpikeLayerNorm affine params). Passing frozen
    # params to AdamW wastes memory/state and can mask configuration mistakes.
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found on the student model.")
    opt = AdamW(trainable_params, lr=args.lr)

    prompts = _default_prompts()
    # Simple cyclic batching
    losses = []

    for step in range(args.steps):
        batch = [prompts[(step * args.batch_size + i) % len(prompts)] for i in range(args.batch_size)]
        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)

        with torch.no_grad():
            t_out = teacher(**tok)
            t_logits = t_out.logits

        # Prevent autograd from retaining cached tensors across steps (would cause "backward through graph a second time").
        if hasattr(student, "reset_cache"):
            student.reset_cache()
        s_out = student(tok.input_ids, attention_mask=tok.attention_mask, use_cache=False)
        s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]

        # Compare logits on all positions
        if args.last_token_only:
            t_logits_use = t_logits[:, -1:, :]
            s_logits_use = s_logits[:, -1:, :]
        else:
            t_logits_use = t_logits
            s_logits_use = s_logits

        if args.loss_type == "mse":
            loss = F.mse_loss(s_logits_use, t_logits_use)
        elif args.loss_type == "ce_teacher":
            # Hard distillation to improve top-1 agreement: cross-entropy to teacher argmax labels.
            # Use last token only unless overridden by last_token_only flag.
            labels = torch.argmax(t_logits_use.detach(), dim=-1)
            # Flatten (B,S,V) -> (B*S,V)
            loss = F.cross_entropy(
                s_logits_use.view(-1, s_logits_use.size(-1)),
                labels.view(-1),
            )
        elif args.loss_type == "kl_ce":
            T = float(args.temperature)
            V = s_logits_use.size(-1)
            # Flatten [B, S, V] -> [B*S, V] so 'batchmean' averages over all token
            # positions. Without flattening, batchmean divides only by B and inflates
            # the KL by a factor of S.
            t_probs = F.softmax((t_logits_use.float().reshape(-1, V) / T), dim=-1)
            s_logp = F.log_softmax((s_logits_use.float().reshape(-1, V) / T), dim=-1)
            kl = F.kl_div(s_logp, t_probs, reduction="batchmean") * (T * T)
            labels = torch.argmax(t_logits_use.detach(), dim=-1)
            ce = F.cross_entropy(
                s_logits_use.view(-1, s_logits_use.size(-1)),
                labels.view(-1),
            )
            loss = 0.5 * kl + 0.5 * ce
        else:
            # KL between softmax distributions (teacher as target), with temperature.
            # Flatten [B, S, V] -> [B*S, V] so 'batchmean' averages over all token
            # positions (otherwise the KL is inflated by a factor of S).
            T = float(args.temperature)
            V = s_logits_use.size(-1)
            t_probs = F.softmax((t_logits_use.float().reshape(-1, V) / T), dim=-1)
            s_logp = F.log_softmax((s_logits_use.float().reshape(-1, V) / T), dim=-1)
            loss = F.kl_div(s_logp, t_probs, reduction="batchmean") * (T * T)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step()

        losses.append(float(loss.item()))
        if (step + 1) % 25 == 0:
            avg = sum(losses[-25:]) / 25
            print(f"step={step+1}/{args.steps} avg_loss_25={avg:.6f}")

    avg_loss = sum(losses) / max(1, len(losses))

    # Save adapter
    # Save adapter weights (PEFT model) + wrapper logit_scale (non-PEFT param) + optional SpikeLayerNorm sidecar
    inner.save_pretrained(str(out_dir))
    if hasattr(student, "logit_scale"):
        torch.save(student.logit_scale.detach().cpu(), out_dir / "logit_scale.pt")
    if args.train_spike_layernorm:
        ln_state = {}
        for name, m in student.named_modules():
            if isinstance(m, SpikeLayerNorm):
                ln_state[name] = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save(ln_state, out_dir / "spike_layernorm_state.pt")
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    report = TrainReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model_name=args.model_name,
        device=device,
        timesteps=int(args.timesteps),
        steps=int(args.steps),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        loss_type=args.loss_type,
        avg_loss=float(avg_loss),
        out_dir=str(out_dir),
    )
    (out_dir / "train_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Saved adapter to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


