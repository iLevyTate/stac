from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

from loihi_constraints import validate_loihi_export_readiness, write_report

from .model import AdExParams, DLPFCAdExNeuron, DLPFCTransformer


@dataclass(frozen=True)
class STACV1Config:
    # Backbone
    model_name: str = "gpt2"

    # Sequence/model sizes
    seq_length: int = 128
    dlpfc_output_size: int = 512
    num_recurrent_layers: int = 1
    dropout_prob: float = 0.2
    hdm_dim: int = 1024

    # Training
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    l1_lambda: float = 1e-5

    # Repro
    seed: int = 42

    # Output
    output_dir: str = "stac_v1_output"

    # Neuron params
    adex_params: AdExParams = field(default_factory=AdExParams)


def set_seed(seed_value: int) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def freeze_for_hybrid_finetune(
    model: DLPFCTransformer,
    *,
    freeze_backbone: bool = True,
    train_lm_head: bool = True,
) -> None:
    """
    Implements STAC V1's *hybrid fine-tuning* mode:
    - Uses a pretrained transformer backbone (GPT-2) as a frozen feature extractor
    - Trains neuromorphic components (DLPFC spiking layer + HEMM)
    - Optionally trains the LM head
    """
    for p in model.parameters():
        p.requires_grad = True

    # model.py pins the AdEx reference potentials as non-trainable by design. The blanket
    # enable above silently un-froze them, so hybrid fine-tuning was training V_th /
    # V_reset / V_rest against that stated design. Re-pin them.
    for module in model.modules():
        if isinstance(module, DLPFCAdExNeuron):
            for fixed in ("V_th", "V_reset", "V_rest"):
                param = getattr(module, fixed, None)
                if param is not None:
                    param.requires_grad = False

    if freeze_backbone:
        for p in model.gpt2.parameters():
            p.requires_grad = False

    if not train_lm_head:
        for p in model.lm_head.parameters():
            p.requires_grad = False


def build_model_and_tokenizer(cfg: STACV1Config) -> Tuple[DLPFCTransformer, GPT2Tokenizer]:
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DLPFCTransformer(
        model_name=cfg.model_name,
        dlpfc_output_size=cfg.dlpfc_output_size,
        num_recurrent_layers=cfg.num_recurrent_layers,
        adex_params=cfg.adex_params,
        dropout_prob=cfg.dropout_prob,
        hdm_dim=cfg.hdm_dim,
    )
    return model, tokenizer


def _tokenize_texts(
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    *,
    seq_length: int,
) -> List[Dict[str, torch.Tensor]]:
    items: List[Dict[str, torch.Tensor]] = []
    for t in texts:
        enc = tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=int(seq_length),
            return_tensors="pt",
        )
        items.append(
            {
                "input_ids": enc["input_ids"].squeeze(0).to(dtype=torch.long),
                "attention_mask": enc["attention_mask"].squeeze(0).to(dtype=torch.long),
            }
        )
    return items


def build_dataloader_from_texts(
    tokenizer: GPT2Tokenizer,
    texts: Iterable[str],
    *,
    seq_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    cleaned = [t for t in texts if isinstance(t, str) and t.strip()]
    tokenized = _tokenize_texts(tokenizer, cleaned, seq_length=seq_length)
    return DataLoader(tokenized, batch_size=int(batch_size), shuffle=bool(shuffle))


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _perplexity(xent_loss: float) -> float:
    # Pure scalar/numpy math on a Python float; no autograd involved, so no need for
    # a torch.no_grad() context here.
    try:
        return float(np.exp(xent_loss))
    except Exception:
        return float("inf")


def _param_counts(model: torch.nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


def _module_param_counts(model: DLPFCTransformer) -> Dict[str, Dict[str, int]]:
    return {
        "gpt2": _param_counts(model.gpt2),
        "dlpfc": _param_counts(model.dlpfc),
        "memory_module": _param_counts(model.memory_module),
        "lm_head": _param_counts(model.lm_head),
        "full_model": _param_counts(model),
    }


@torch.no_grad()
def _spike_stats(spk_trains: torch.Tensor) -> Dict[str, float]:
    """
    spk_trains is expected to be [batch, seq_len, neurons].
    Values are typically 0/1 spikes (post-surrogate) possibly with dropout.
    """
    # Keep this robust even if values aren't strictly binary.
    x = spk_trains.detach()
    mean = float(x.mean().item())
    # Fraction of exact zeros (sparsity proxy)
    frac_zero = float((x == 0).to(dtype=torch.float32).mean().item())
    # Per-neuron mean spike rate (aggregate) – return summary stats only
    per_neuron = x.mean(dim=(0, 1))
    return {
        "spike_mean": mean,
        "spike_frac_zero": frac_zero,
        "spike_neuron_mean_min": float(per_neuron.min().item()),
        "spike_neuron_mean_max": float(per_neuron.max().item()),
    }


def save_checkpoint(
    *,
    model: DLPFCTransformer,
    cfg: STACV1Config,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Minimal checkpoint intended for reproducible V1 runs:
    - model weights
    - config used
    - optional extra metadata (metrics, loihi report path, etc.)

    This intentionally does NOT attempt to resume optimizer/scheduler state.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
    }
    if extra:
        payload["extra"] = extra
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    return out_path


def load_checkpoint(
    *,
    model: DLPFCTransformer,
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    in_path = Path(path)
    payload = torch.load(in_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload


def train_steps(
    model: DLPFCTransformer,
    dataloader: DataLoader,
    *,
    cfg: STACV1Config,
    device: torch.device,
    max_steps: int,
    hybrid_finetune: bool = True,
    train_lm_head: bool = True,
    write_loihi_report: bool = True,
    checkpoint_out: Optional[str | Path] = None,
) -> Dict[str, float]:
    model.to(device)
    model.train()

    if hybrid_finetune:
        freeze_for_hybrid_finetune(model, freeze_backbone=True, train_lm_head=train_lm_head)

    module_counts = _module_param_counts(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check hybrid fine-tuning freeze settings.")

    optimizer = AdamW(trainable_params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    total_steps = max(1, int(max_steps))
    # Cap warmup so it cannot swallow the whole run. With the default warmup_steps=100 and
    # the documented short runs (--steps 3), the schedule never left warmup: every step
    # ran at a small fraction of the configured learning rate, so nothing trained.
    warmup_steps = min(int(cfg.warmup_steps), max(0, total_steps // 10))
    if warmup_steps != int(cfg.warmup_steps):
        logging.getLogger(__name__).info(
            "Clamping warmup_steps %d -> %d for a %d-step run",
            int(cfg.warmup_steps), warmup_steps, total_steps,
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    steps = 0
    loss_sum = 0.0
    l1_sum = 0.0
    spike_sum = 0.0
    spike_frac_zero_sum = 0.0
    spike_neuron_min_sum = 0.0
    spike_neuron_max_sum = 0.0

    # Repeat the dataloader until max_steps is reached. Iterating it once silently capped
    # a run at len(dataloader) steps, so `--steps N` quietly did fewer than N.
    def _step_batches():
        while True:
            empty = True
            for batch in dataloader:
                empty = False
                yield batch
            if empty:
                return

    for batch in _step_batches():
        if steps >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, spk_trains = model(input_ids, attention_mask=attention_mask)

        # Mask padded positions to -100 so the loss ignores them. Without this the model
        # is trained to predict the eos padding tokens, which dominates short sequences.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_xent = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_l1 = float(cfg.l1_lambda) * torch.mean(torch.abs(spk_trains))
        loss = loss_xent + loss_l1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_sum += float(loss_xent.item())
        l1_sum += float(loss_l1.item())

        s = _spike_stats(spk_trains)
        spike_sum += s["spike_mean"]
        spike_frac_zero_sum += s["spike_frac_zero"]
        spike_neuron_min_sum += s["spike_neuron_mean_min"]
        spike_neuron_max_sum += s["spike_neuron_mean_max"]
        steps += 1

    avg_loss = loss_sum / max(1, steps)
    avg_l1 = l1_sum / max(1, steps)
    avg_spike = spike_sum / max(1, steps)
    avg_spike_frac_zero = spike_frac_zero_sum / max(1, steps)
    avg_spike_neuron_min = spike_neuron_min_sum / max(1, steps)
    avg_spike_neuron_max = spike_neuron_max_sum / max(1, steps)

    out: Dict[str, float] = {
        "steps": float(steps),
        "train_loss": float(avg_loss),
        "train_perplexity": _perplexity(avg_loss),
        "train_l1": float(avg_l1),
        "spike_mean": float(avg_spike),
        "spike_frac_zero": float(avg_spike_frac_zero),
        "spike_neuron_mean_min": float(avg_spike_neuron_min),
        "spike_neuron_mean_max": float(avg_spike_neuron_max),
    }

    summary: Dict[str, Any] = {
        "metrics": out,
        "hybrid_finetune": bool(hybrid_finetune),
        "train_lm_head": bool(train_lm_head),
        "param_counts": module_counts,
        "config": asdict(cfg),
    }

    if write_loihi_report:
        export_ready, report = validate_loihi_export_readiness(model, intended_weight_bits=8, require_spiking_neurons=True)
        report["stac_v1"] = {
            "hybrid_finetune": bool(hybrid_finetune),
            "train_lm_head": bool(train_lm_head),
            "config": asdict(cfg),
            "export_ready": bool(export_ready),
        }
        report_path = write_report(report, Path(cfg.output_dir) / "loihi_constraints_reports")
        out["loihi_export_ready"] = 1.0 if export_ready else 0.0
        out["loihi_report_written"] = 1.0
        summary["loihi"] = {
            "export_ready": bool(export_ready),
            "hard_block_count": int(report.get("hard_block_count", 0)),
            "warning_count": int(report.get("warning_count", 0)),
            "hard_blocks": [
                f.get("id")
                for f in (report.get("findings", []) or [])
                if isinstance(f, dict) and f.get("severity") == "HARD_BLOCK"
            ],
            "report_path": str(report_path),
        }
    else:
        summary["loihi"] = {"skipped": True}

    _atomic_write_json(Path(cfg.output_dir) / "last_run_summary.json", summary)

    if checkpoint_out is not None:
        save_checkpoint(model=model, cfg=cfg, path=checkpoint_out, extra={"last_run_summary": summary})

    return out


