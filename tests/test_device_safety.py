"""
Device- and dtype-safety regression tests that run on CPU.

The audit found a bug that only manifests on CUDA: `TemporalSpikeProcessor.device` was
snapshotted in `__init__`, so a later `.to('cuda')` left every tensor derived from it on
the CPU. It was fixed blind, because no GPU was available — and that is the problem this
file addresses. A tensor created inside a forward pass without an explicit `device=` lands
on the default device, which is correct on CPU and wrong everywhere else, so CPU testing
alone can never surface it.

Instead of needing a GPU, these tests observe *how* tensors are allocated: every
allocation made by this repository's own code during a forward pass must state its device
explicitly (or use a `*_like` form that inherits one). That invariant is checkable on any
machine and rules out the entire bug class.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow running this file directly by putting the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from tests.test_spiking import _convert, _model_path  # shared offline-model helpers

# Allocation functions that place a tensor on a device of their own choosing. The
# `*_like` variants are deliberately absent: they inherit device and dtype from their
# argument, which is always safe.
_ALLOCATORS = ("zeros", "ones", "arange", "empty", "full", "tensor", "eye", "linspace")

# Files whose allocations we hold to this standard: this repository's own modules.
_WATCHED = ("smollm2_converter.py", "stac_v1/model.py", "stac_v1/pipeline.py", "spike_metrics.py")


def _from_watched_code(stack) -> str | None:
    """Return the watched file responsible for a call, if any."""
    for frame in reversed(stack[:-1]):
        filename = frame.filename.replace("\\", "/")
        for watched in _WATCHED:
            if filename.endswith(watched):
                return f"{watched}:{frame.lineno}"
    return None


class _AllocationRecorder:
    """Wrap torch allocators and record those made from this repo's code."""

    def __init__(self):
        self.calls = []
        self._originals = {}

    def __enter__(self):
        for name in _ALLOCATORS:
            original = getattr(torch, name)
            self._originals[name] = original
            setattr(torch, name, self._wrap(name, original))
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            setattr(torch, name, original)

    def _wrap(self, name, original):
        def wrapper(*args, **kwargs):
            origin = _from_watched_code(traceback.extract_stack())
            if origin is not None:
                self.calls.append(
                    {"fn": name, "origin": origin, "device": kwargs.get("device", "<unset>")}
                )
            return original(*args, **kwargs)

        return wrapper


def _forward_and_record(model, ids, **kwargs):
    with _AllocationRecorder() as recorder:
        with torch.no_grad():
            model(ids, **kwargs)
    return recorder.calls


@pytest.mark.parametrize("real_spiking", [False, True])
def test_forward_pass_allocations_specify_a_device(real_spiking):
    """
    Every tensor this repo allocates during a forward pass must name its device.

    An allocation without `device=` silently lands on the default device. On CPU that is
    invisible; on CUDA it produces "Expected all tensors to be on the same device"
    somewhere far from the cause — or, worse, works by accident.
    """
    model = _convert("tiny-gpt2", timesteps=2, real_spiking=real_spiking)
    ids = torch.randint(0, 200, (1, 12))

    calls = _forward_and_record(model, ids, use_cache=True)
    assert calls, "recorded no allocations — the recorder is not observing the forward pass"

    unset = [c for c in calls if c["device"] == "<unset>"]
    assert not unset, (
        "tensors allocated without an explicit device (these break on CUDA):\n"
        + "\n".join(f"  torch.{c['fn']}(...) at {c['origin']}" for c in unset)
    )


def test_kv_cache_path_allocations_specify_a_device():
    """Same invariant on the cached multi-turn path, which allocates masks and padding."""
    model = _convert("tiny-gpt2", timesteps=2, real_spiking=False)
    ids = torch.randint(0, 200, (1, 8))

    all_calls = []
    with torch.no_grad():
        for _ in range(3):
            all_calls += _forward_and_record(model, ids, use_cache=True)
            ids = torch.cat([ids, torch.randint(0, 200, (1, 1))], dim=1)

    unset = [c for c in all_calls if c["device"] == "<unset>"]
    assert not unset, (
        "tensors allocated without an explicit device on the KV-cache path:\n"
        + "\n".join(f"  torch.{c['fn']}(...) at {c['origin']}" for c in unset)
    )


def test_no_forward_pass_allocation_omits_a_device():
    """
    Static counterpart to the runtime checks above, covering branches tests do not reach.

    Runtime recording only sees code that actually executes: the mask-padding branch in
    TemporalSpikeProcessor.forward, for instance, needs a short attention mask to trigger,
    so a missing `device=` there survives a passing test run. Parsing the source instead
    covers every branch.

    The rule applies to `forward` methods only. Allocations in `__init__` build parameters
    and buffers, which `nn.Module.to()` relocates correctly.
    """
    import ast

    violations = []
    for watched in _WATCHED:
        path = _REPO_ROOT / watched
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "forward"):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not (isinstance(func, ast.Attribute) and func.attr in _ALLOCATORS):
                    continue
                if not (isinstance(func.value, ast.Name) and func.value.id == "torch"):
                    continue
                if any(kw.arg == "device" for kw in call.keywords):
                    continue
                violations.append(f"{watched}:{call.lineno} torch.{func.attr}(...) in {node.name}()")

    assert not violations, (
        "allocations inside forward() without an explicit device (these break on CUDA):\n"
        + "\n".join(f"  {v}" for v in violations)
        + "\nUse device=<input>.device, or a *_like form that inherits it."
    )


def test_processor_device_tracks_the_model():
    """
    `TemporalSpikeProcessor.device` must follow the model, not a snapshot from __init__.

    This is the exact CUDA-only bug the audit fixed blind: `.to(...)` changed where the
    parameters lived while `self.device` kept pointing at the construction-time device.
    """
    model = _convert("tiny-gpt2", timesteps=2, real_spiking=False)
    assert model.device == next(model.snn_model.parameters()).device

    moved = model.to("cpu")
    assert moved.device == next(moved.snn_model.parameters()).device, (
        "device is a stale snapshot rather than a live property"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conversion_preserves_model_dtype(dtype):
    """
    Converting must not silently promote precision.

    Replacement modules build parameters at torch's float32 default and `Tensor.copy_`
    keeps the destination dtype, so a fp16 model used to come back as float32 — doubling
    the memory of exactly the large models this project targets.
    """
    from transformers import AutoModelForCausalLM
    from smollm2_converter import simplified_conversion

    try:
        base = AutoModelForCausalLM.from_pretrained(_model_path("tiny-gpt2"), torch_dtype=dtype)
    except Exception as e:
        pytest.skip(f"Could not load the test model: {e}")

    converted = simplified_conversion(base, 2, skip_gelu_replacement=True)
    leaks = [
        name
        for name, tensor in list(converted.named_parameters()) + list(converted.named_buffers())
        if tensor.is_floating_point() and tensor.dtype != dtype
    ]
    assert not leaks, f"conversion promoted these out of {dtype}: {leaks}"

    ids = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = converted(ids, use_cache=False)
    assert out.logits.dtype == dtype
