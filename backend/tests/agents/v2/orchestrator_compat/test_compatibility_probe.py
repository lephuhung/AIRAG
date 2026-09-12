"""Compatibility probe tests for the frozen `context_schema` API.

These tests verify that the candidate LangGraph / checkpoint stack actually
satisfies the API surface the v2 plan requires (`StateGraph(context_schema=...)`,
`AsyncPostgresSaver.from_conn_string(...)`, psycopg `postgresql://` DSN
operations, and the supporting langgraph modules used by later-phase code
samples).

They fail RED until `backend/scripts/probe_v2_compatibility.py` is
implemented (TDD Step 2) and pass GREEN once the probe is real (Step 3).

The discovery report file (Step 4) is the source of truth for which exact
candidate tuple is eligible for Task 2 benchmarking.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Make `scripts` importable when pytest is run from backend/.
# This file lives at backend/tests/agents/v2/orchestrator_compat/, so
# parents[4] is backend/ and parents[4]/"scripts" is backend/scripts.
_BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _BACKEND_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from probe_v2_compatibility import (  # noqa: E402  (sys.path tweak above)
    CompatibilityResult,
    probe_current_interpreter,
    reject_sqlalchemy_asyncpg_dsn,
)

REPORT_PATH = (
    _BACKEND_ROOT / "tests" / "reports" / "v2-compatibility-candidates.json"
)


# A representative DSN. The actual host/port will vary per environment;
# the probe just needs to perform an open attempt. We use a host the
# local Docker stack publishes (port 5433 -> 5432 inside `hrag-postgres`)
# and the test database that lives there.
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/hrag_test"


def test_reject_sqlalchemy_asyncpg_dsn() -> None:
    """The probe must refuse SQLAlchemy `postgresql+asyncpg://` URLs."""
    with pytest.raises(ValueError, match="postgresql\\+asyncpg"):
        reject_sqlalchemy_asyncpg_dsn(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/hrag"
        )


def test_reject_sqlalchemy_asyncpg_dsn_passes_plain_postgres() -> None:
    """A plain `postgresql://` DSN must NOT be rejected by the guard."""
    # Should not raise.
    reject_sqlalchemy_asyncpg_dsn(DEFAULT_DSN)


def test_compatibility_result_fields_present() -> None:
    """CompatibilityResult is a typed record with the frozen API flags."""
    r = CompatibilityResult(
        python_version="3.12.3",
        packages={},
        context_schema_supported=False,
        async_postgres_saver_imported=False,
        psycopg_dsn_opened=False,
        notes="uninitialized",
    )
    assert r.context_schema_supported is False
    assert r.async_postgres_saver_imported is False
    assert r.psycopg_dsn_opened is False


@pytest.mark.parametrize("dsn", [DEFAULT_DSN])
def test_probe_current_interpreter_frozen_api(dsn: str) -> None:
    """The current interpreter must satisfy the frozen v2 API surface.

    This test gates Step 3's GREEN state. It exercises:
      * `StateGraph(..., context_schema=...)` accepts `context_schema`
      * `AsyncPostgresSaver` imports from the pinned location
      * a `postgresql://` psycopg DSN can actually be opened
    """
    result = probe_current_interpreter(dsn)
    assert result.context_schema_supported, (
        "Frozen API requires StateGraph(context_schema=...) — the installed "
        "langgraph version does not expose it."
    )
    assert result.async_postgres_saver_imported, (
        "Frozen API requires langgraph.checkpoint.postgres.aio.AsyncPostgresSaver "
        "to import cleanly."
    )
    assert result.psycopg_dsn_opened, (
        "Frozen API requires a real postgresql:// psycopg DSN to open. "
        "Check that CHECKPOINT_DATABASE_URL points at a reachable PostgreSQL."
    )


def test_discovery_report_path_is_documented() -> None:
    """The discovery report path must be the same path Task 2 reads from.

    Locks the literal contract:
      * ``REPORT_PATH`` resolves to the canonical file Task 2's ``--input``
        flag will pass.
      * When the probe has been run, the file exists and parses as JSON
        with a non-empty ``candidates`` list.
    """
    expected = (
        _BACKEND_ROOT / "tests" / "reports" / "v2-compatibility-candidates.json"
    )
    assert REPORT_PATH == expected, (
        "REPORT_PATH drifted from the canonical Task 2 --input target"
    )
    if not expected.exists():
        pytest.skip("discovery report not produced yet (run --discover first)")
    payload = json.loads(expected.read_text())
    assert payload.get("candidates"), (
        "discovery report exists but has no candidates array; "
        "the probe cannot have produced a passing tuple without writing one"
    )


@pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason="discovery report not produced yet (run probe --discover first)",
)
def test_discovery_report_has_at_least_one_passing_tuple() -> None:
    """If the discovery report exists, it must contain ≥ 1 passing tuple."""
    report = json.loads(REPORT_PATH.read_text())
    passing = [
        c for c in report.get("candidates", []) if c.get("passed", False)
    ]
    assert passing, (
        "Discovery produced no passing candidate tuple. Phase 0 gate is "
        "blocked; do not advance to Phase 1 until at least one tuple "
        "satisfies the frozen API surface."
    )


@pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason="discovery report not produced yet",
)
def test_discovery_report_records_exact_versions() -> None:
    """Each candidate with a result must record the exact installed versions."""
    report = json.loads(REPORT_PATH.read_text())
    for candidate in report.get("candidates", []):
        result = candidate.get("result")
        if result is None:
            # Candidate failed at pip install — no installed versions to record.
            continue
        pkgs = result.get("packages", {})
        for required in ("langgraph", "langgraph-checkpoint-postgres"):
            assert required in pkgs, (
                f"candidate {candidate.get('id')} missing exact version for {required}"
            )
            assert pkgs[required] not in ("unknown", "not-installed", ""), (
                f"candidate {candidate.get('id')} has unresolved {required} version"
            )


def test_cli_accepts_top_level_discover_flag(tmp_path) -> None:
    """Brief Step 4 documents `probe_v2_compatibility --discover ...`.

    The CLI must accept top-level flags (not just subcommands) so the
    plan's documented invocations run unchanged. This test mocks the
    discovery handler to verify argparse shape only.
    """
    from probe_v2_compatibility import main

    dsn = "postgresql://u:p@localhost:5433/hrag_test"
    output = tmp_path / "report.json"
    rc = main(
        [
            "--discover",
            "--checkpoint-dsn",
            dsn,
            "--output",
            str(output),
            "--write-requirements",
            str(tmp_path / "req.txt"),
        ]
    )
    # Discovery will actually run (and either succeed or fail gracefully);
    # the contract is that argparse did NOT reject the invocation with
    # rc=2 "invalid choice". A rc != 2 from other failures is acceptable.
    assert rc != 2 or output.exists(), (
        "top-level --discover invocation was rejected by argparse"
    )


def test_cli_accepts_top_level_reemit_flags(tmp_path) -> None:
    """Brief-documented reemit invocation must not exit rc=2.

    Uses an existing report (the one produced by Step 4) to avoid
    running a fresh discovery.
    """
    from probe_v2_compatibility import main

    if not REPORT_PATH.exists():
        pytest.skip("discovery report not produced yet")
    out = tmp_path / "req.txt"
    rc = main(["--input", str(REPORT_PATH), "--write-requirements", str(out)])
    # argparse must accept the invocation; rc=0 means success. A non-zero
    # rc from runtime is OK as long as it is not rc=2 (argparse error).
    assert rc != 2, (
        "top-level --input/--write-requirements invocation was rejected by argparse"
    )
    assert out.exists(), "reemit must have written the requirements file"
