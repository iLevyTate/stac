"""Pytest configuration for the STAC test suite.

Ensures the repository root is importable so tests can `import smollm2_converter`,
`import loihi_constraints`, `from stac_v1 ...` etc. when run via `pytest` from any
working directory. The test modules are also runnable as standalone scripts and each
carries an equivalent sys.path shim of its own.

Also resolves the model the suite runs against. The tests default to Hugging Face hub
ids, so on an air-gapped runner (or behind a proxy that blocks huggingface.co) nothing
could be executed at all. When the hub is unreachable and STAC_TEST_MODEL is unset, a
tiny model is generated locally instead — see scripts/make_test_models.py.
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Where generated offline models live. Gitignored (`local/`).
_GENERATED_MODEL_ROOT = Path(_REPO_ROOT) / "local" / "test-models"


def _hub_reachable(timeout: float = 3.0) -> bool:
    """Cheap probe for Hugging Face hub availability."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    import socket
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co/api/models/gpt2", timeout=timeout)
        return True
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


def pytest_configure(config):
    """Point STAC_TEST_MODEL at a locally generated model when the hub is unreachable."""
    if os.environ.get("STAC_TEST_MODEL"):
        return
    if _hub_reachable():
        return

    try:
        from scripts.make_test_models import ensure_test_model
    except ImportError:
        # scripts/ is not a package; load it by path.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_stac_make_test_models", Path(_REPO_ROOT) / "scripts" / "make_test_models.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ensure_test_model = module.ensure_test_model

    try:
        path = ensure_test_model("tiny-gpt2", out_root=_GENERATED_MODEL_ROOT)
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"conftest: could not generate an offline test model ({e}); tests may skip.")
        return

    os.environ["STAC_TEST_MODEL"] = str(path)
    print(f"conftest: Hugging Face hub unreachable; using generated model at {path}")
