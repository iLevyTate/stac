"""
Shared helpers for locating and converting offline test models.

Loaded by explicit path (see `load_offline_helpers` in the consuming test files) rather
than imported as `tests.something`. Cross-importing between test modules depends on
pytest's import mode, its rootdir detection and the ambient sys.path:
`from tests.test_spiking import ...` resolved locally but failed in CI with
"No module named 'tests.test_spiking'", because `tests/` has no __init__.py and is only a
namespace package when the repository root happens to be on sys.path. Path loading has no
such dependency.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM

from smollm2_converter import simplified_conversion

_REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_MODELS = _REPO_ROOT / "local" / "test-models"


def _make_test_models():
    """Import scripts/make_test_models.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_stac_make_test_models", _REPO_ROOT / "scripts" / "make_test_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_path(name: str) -> str:
    """
    Locate a test model, generating the offline fixtures if needed.

    `tiny-gpt2` honours STAC_TEST_MODEL so the suite can be pointed at a real checkpoint;
    the Llama variants are architecture-specific and always come from the generator.
    """
    if name == "tiny-gpt2":
        configured = os.environ.get("STAC_TEST_MODEL")
        if configured:
            return configured

    path = GENERATED_MODELS / name
    if not (path / "config.json").exists():
        try:
            path = _make_test_models().ensure_test_model(name, out_root=GENERATED_MODELS)
        except Exception as e:
            pytest.skip(f"Could not generate test model {name!r}: {e}")
    return str(path)


def convert(name: str, *, timesteps: int, real_spiking: bool):
    """Load a test model and run it through simplified_conversion."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path(name))
    except Exception as e:
        pytest.skip(f"Could not load {name!r}: {e}")
    model.eval()
    converted = simplified_conversion(
        model, timesteps, skip_gelu_replacement=True, real_spiking=real_spiking
    )
    converted.eval()
    return converted
