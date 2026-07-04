"""Pytest configuration for the STAC test suite.

Ensures the repository root is importable so tests can `import smollm2_converter`,
`import loihi_constraints`, `from stac_v1 ...` etc. when run via `pytest` from any
working directory. The test modules are also runnable as standalone scripts and each
carries an equivalent sys.path shim of its own.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
