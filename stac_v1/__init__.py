"""
STAC V1 (research prototype) - runnable, importable implementation.

This package intentionally stays lightweight and dependency-minimal so it can run
in this repo without requiring Jupyter/Colab-only tooling.
"""

from .model import (
    # AdExParams is a required argument of DLPFCLayer/DLPFCAdExNeuron/DLPFCTransformer,
    # so leaving it out of the package exports made those classes unusable via
    # `from stac_v1 import ...` alone.
    AdExParams,
    CurrentDrive,
    DLPFCTransformer,
    DLPFCLayer,
    DLPFCAdExNeuron,
    HyperdimensionalMemoryModule,
    SurrogateSpikeFunction,
    surrogate_spike,
)
from .pipeline import (
    STACV1Config,
    build_model_and_tokenizer,
    build_dataloader_from_texts,
    load_checkpoint,
    save_checkpoint,
    train_steps,
    set_seed,
    freeze_for_hybrid_finetune,
)

__all__ = [
    "AdExParams",
    "CurrentDrive",
    "DLPFCTransformer",
    "DLPFCLayer",
    "DLPFCAdExNeuron",
    "HyperdimensionalMemoryModule",
    "SurrogateSpikeFunction",
    "surrogate_spike",
    "STACV1Config",
    "build_model_and_tokenizer",
    "build_dataloader_from_texts",
    "load_checkpoint",
    "save_checkpoint",
    "train_steps",
    "set_seed",
    "freeze_for_hybrid_finetune",
]


