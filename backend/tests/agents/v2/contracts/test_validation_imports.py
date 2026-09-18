"""Discovery plan Task 2 — cycle-free import order (subprocess, both directions)."""
from __future__ import annotations

import subprocess
import sys

from app.services.agents.v2.contracts import validation, validation_support


def _run_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )


def test_exception_identities_shared() -> None:
    assert validation.ContractValidationError is validation_support.ContractValidationError
    assert (
        validation.IncompatibleCheckpointError
        is validation_support.IncompatibleCheckpointError
    )


def test_import_validation_then_discovery_contracts() -> None:
    result = _run_snippet(
        "import app.services.agents.v2.contracts.validation;"
        "import app.services.agents.v2.discovery_bootstrap.contracts;"
        "import app.services.agents.supervisor_v2"
    )
    assert result.returncode == 0, result.stderr


def test_import_discovery_contracts_then_validation() -> None:
    result = _run_snippet(
        "import app.services.agents.v2.discovery_bootstrap.contracts;"
        "import app.services.agents.v2.contracts.validation;"
        "import app.services.agents.supervisor_v2"
    )
    assert result.returncode == 0, result.stderr


def test_legacy_discovery_module_exports_unchanged() -> None:
    result = _run_snippet(
        "from app.services.agents.v2.discovery import ("
        "DiscoveryDenied, DiscoveryDisabled, mint_candidate, request_addition)"
    )
    assert result.returncode == 0, result.stderr
    import app.services.agents.v2.discovery as legacy

    assert legacy.DiscoveryDenied is not None
    assert callable(legacy.mint_candidate)


def test_discovery_contracts_module_is_data_only() -> None:
    import ast
    import inspect

    import app.services.agents.v2.discovery_bootstrap.contracts as dc

    tree = ast.parse(inspect.getsource(dc))
    forbidden = (
        "contracts.validation",
        "contracts.state",
        "contracts.planning",
        "discovery.validation",
        "validation",
        "state",
        "planning",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not any(
                node.module.endswith(name) for name in forbidden
            ), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(
                    alias.name.endswith(name) for name in forbidden
                ), alias.name
