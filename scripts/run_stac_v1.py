from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stac_v1.pipeline import (
    STACV1Config,
    build_dataloader_from_texts,
    build_model_and_tokenizer,
    load_checkpoint,
    set_seed,
    train_steps,
)


def _default_texts() -> list[str]:
    # Minimal local texts so STAC V1 can run without extra dependencies (e.g., datasets).
    return [
        "STAC V1 demonstrates hybrid fine-tuning: a pretrained transformer backbone plus neuromorphic spiking layers.",
        "Neuromorphic edge deployment prioritizes sparse activity and quantization, but dense attention remains a mapping challenge.",
        "Hyperdimensional memory (HEMM) provides a context bias over spike trains.",
        "Surrogate gradients enable training spiking dynamics end-to-end.",
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Run STAC V1 hybrid fine-tuning smoke pipeline (CPU/GPU).")
    p.add_argument("--model_name", default="gpt2", help="HF model name (use sshleifer/tiny-gpt2 for fast smoke runs).")
    p.add_argument("--seq_length", type=int, default=64)
    p.add_argument("--dlpfc_output_size", type=int, default=64)
    p.add_argument("--hdm_dim", type=int, default=128)
    p.add_argument("--num_recurrent_layers", type=int, default=1)
    p.add_argument("--dropout_prob", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=3, help="Number of optimization steps to run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="stac_v1_output")

    p.add_argument("--texts_file", default=None, help="Optional UTF-8 text file (one sample per line).")
    p.add_argument(
        "--text",
        action="append",
        default=[],
        help="Inline text sample. Can be provided multiple times. If set, overrides --texts_file.",
    )

    p.add_argument("--checkpoint_in", default=None, help="Optional checkpoint path to load model weights from.")
    p.add_argument("--checkpoint_out", default=None, help="Optional checkpoint path to write after the run.")

    p.add_argument("--no_hybrid_finetune", action="store_true", help="If set, trains all params (not hybrid).")
    p.add_argument("--no_train_lm_head", action="store_true", help="If set, does not train the LM head.")
    p.add_argument("--no_loihi_report", action="store_true", help="If set, skips writing the Loihi constraints report.")
    p.add_argument("--no_shuffle", action="store_true", help="If set, does not shuffle the small training set.")

    args = p.parse_args()

    cfg = STACV1Config(
        model_name=args.model_name,
        seq_length=args.seq_length,
        dlpfc_output_size=args.dlpfc_output_size,
        num_recurrent_layers=args.num_recurrent_layers,
        dropout_prob=args.dropout_prob,
        hdm_dim=args.hdm_dim,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model_and_tokenizer(cfg)

    if args.checkpoint_in:
        load_checkpoint(model=model, path=args.checkpoint_in, map_location="cpu", strict=True)
        print(f"Loaded checkpoint: {args.checkpoint_in}")

    if args.text:
        texts = args.text
    elif args.texts_file:
        texts = [ln.strip() for ln in Path(args.texts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        texts = _default_texts()

    loader = build_dataloader_from_texts(
        tokenizer,
        texts,
        seq_length=cfg.seq_length,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
    )

    metrics = train_steps(
        model,
        loader,
        cfg=cfg,
        device=device,
        max_steps=args.steps,
        hybrid_finetune=not args.no_hybrid_finetune,
        train_lm_head=not args.no_train_lm_head,
        write_loihi_report=not args.no_loihi_report,
        checkpoint_out=args.checkpoint_out,
    )

    print("STAC V1 run complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Output dir: {cfg.output_dir}")
    print("See last_run_summary.json for a machine-readable summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


