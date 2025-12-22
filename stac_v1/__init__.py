"""
STAC V1 (research prototype) - runnable, importable implementation.

This package intentionally stays lightweight and dependency-minimal so it can run
in this repo without requiring Jupyter/Colab-only tooling.
"""

from .model import (
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
    train_steps,
    set_seed,
    freeze_for_hybrid_finetune,
)


