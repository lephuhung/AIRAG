"""Task 3 — PostgreSQL checkpoint readiness guard.

Determines whether `No module named 'langgraph.checkpoint.postgres'` is
declaration drift or a merely-uninstalled declared requirement — without
installing anything, churning pins, or inventing fallbacks.

- Declaration tests always run: they parse the pinned requirement files and
  assert the exact checkpoint-stack pins the production import path needs.
- Import tests attempt the EXACT production symbols
  (`langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`,
  `persistence.checkpoint.create_v2_checkpointer`). When the declared extra
  is absent from the current interpreter they SKIP (not fail) with the exact
  installation command — a missing install must never masquerade as a code
  defect, and a missing declaration must never masquerade as an install gap.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[4]

#: The exact declared pins the production checkpoint path needs.
REQUIRED_POSTGRES_PIN = "langgraph-checkpoint-postgres==2.0.25"
REQUIRED_PSYCOPG_PIN = "psycopg[binary]==3.2.3"

INSTALL_HINT = (
    "declared requirement 'langgraph-checkpoint-postgres==2.0.25' is not "
    "installed in this interpreter; the backend container image is "
    "dependency-complete — install with "
    "'pip install -r backend/requirements.txt' (or run the persistence "
    "suites inside the backend container / bench venv)"
)


def _requirement_lines(*filenames: str) -> list[str]:
    lines: list[str] = []
    for name in filenames:
        path = BACKEND_ROOT / name
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def test_checkpoint_postgres_pin_is_declared():
    """No drift: the exact checkpoint-stack pins are declared (Task 3)."""
    lines = _requirement_lines("requirements.txt")
    assert REQUIRED_POSTGRES_PIN in lines
    assert REQUIRED_PSYCOPG_PIN in lines


def test_benchmark_requirements_agree_on_checkpoint_postgres_pin():
    """Benchmark env pins the same postgres extra (no pin divergence)."""
    lines = _requirement_lines("requirements-v2-benchmark.txt")
    assert REQUIRED_POSTGRES_PIN in lines


def test_production_postgres_saver_surface_is_available():
    """Import the exact production symbols, or skip with the install gap."""
    try:
        module = importlib.import_module("langgraph.checkpoint.postgres.aio")
    except ModuleNotFoundError as exc:
        if "langgraph.checkpoint.postgres" in str(exc):
            pytest.skip(INSTALL_HINT)
        raise
    saver = getattr(module, "AsyncPostgresSaver", None)
    assert saver is not None
    # The exact surface persistence/checkpoint.py uses (no more, no less).
    assert callable(getattr(saver, "from_conn_string", None))
    assert callable(getattr(saver, "setup", None))


def test_production_checkpoint_factory_imports_or_skips_truthfully():
    """persistence.checkpoint imports whole, or fails only on the extra."""
    try:
        import app.services.agents.v2.persistence.checkpoint as checkpoint
    except ModuleNotFoundError as exc:
        if "langgraph.checkpoint.postgres" in str(exc):
            pytest.skip(INSTALL_HINT)
        raise
    assert callable(checkpoint.create_v2_checkpointer)
    assert callable(checkpoint.check_v2_checkpointer)
    assert callable(checkpoint.setup_v2_checkpointer)
    assert checkpoint.CHECKPOINT_TABLES
