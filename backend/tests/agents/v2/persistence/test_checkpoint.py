"""Phase 1D Task 10 — async PostgreSQL v2 checkpointing.

Covers the four deliverables of this task:

1. Production ``backend/requirements.txt`` carries the Phase-0 exact pins for
   the psycopg3-backed checkpoint stack.
2. ``Settings`` exposes a psycopg-compatible ``CHECKPOINT_DATABASE_URL`` and
   rejects the asyncpg driver, absent credentials, an empty value, and direct
   reuse of ``settings.DATABASE_URL``.
3. ``persistence/checkpoint.py`` validates the DSN, opens an
   ``AsyncPostgresSaver`` lifecycle, and sets up / checks the saver tables.
4. Against a **disposable** database (``hrag_test_checkpoint``, created if
   absent) a checkpoint round-trips for a configurable thread ID and an
   incompatible ``contract_version`` is rejected by the spec §25 gate.

The disposable DB is deliberately distinct from ``hrag_test_v2`` /
``hrag_test_v2_task3`` (the persistence suite's DB) so these tests never
clobber it. ``--check`` is asserted to be read-only: on a database with no
saver tables it reports them missing and creates nothing.
"""

from __future__ import annotations

import ast
import inspect
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from psycopg import sql
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.validation import (
    IncompatibleCheckpointError,
    validate_checkpoint_payload,
)
from app.services.agents.v2.persistence import checkpoint as checkpoint_module
from app.services.agents.v2.persistence.checkpoint import (
    CHECKPOINT_TABLES,
    CheckpointCheck,
    check_v2_checkpointer,
    create_v2_checkpointer,
    main as checkpoint_main,
    setup_v2_checkpointer,
    validate_psycopg_dsn,
)

BACKEND_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = BACKEND_ROOT.parent

ASYNC_PG_DSN = "postgresql+asyncpg://postgres:postgres@localhost:5433/hrag_checkpoints"
VALID_PSYCOPG_DSN = "postgresql://postgres:postgres@localhost:5433/hrag_test_checkpoint"

# Disposable database dedicated to checkpointing tests. Override with
# CHECKPOINT_TEST_DATABASE_URL if the controller provisions a different name.
CHECKPOINT_TEST_DSN = os.environ.get(
    "CHECKPOINT_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_checkpoint",
)

_METADATA = {"source": "input", "step": 0, "parents": {}}
_NEW_VERSIONS = {"contract_version": "2.0", "marker": "1"}


# ---------------------------------------------------------------------------
# Fixtures + sync helpers (test-only database lifecycle)
# ---------------------------------------------------------------------------


def _maintenance_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    return urlunparse(parsed._replace(path="/postgres"))


def _ensure_database(dsn: str) -> None:
    """Create the disposable checkpoint database if it does not exist."""
    dbname = urlparse(dsn).path.lstrip("/")
    assert dbname, f"checkpoint test DSN must name a database: {dsn!r}"
    with psycopg.connect(_maintenance_dsn(dsn), autocommit=True, connect_timeout=5) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


def _present_saver_tables(dsn: str) -> frozenset[str]:
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (list(CHECKPOINT_TABLES),),
        ).fetchall()
    return frozenset(row[0] for row in rows)


def _drop_saver_tables(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS "
            "checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations "
            "CASCADE"
        )


@pytest.fixture(scope="session")
def checkpoint_dsn() -> str:
    _ensure_database(CHECKPOINT_TEST_DSN)
    return CHECKPOINT_TEST_DSN


@pytest.fixture
def configured_checkpoint_dsn(checkpoint_dsn: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point ``settings.CHECKPOINT_DATABASE_URL`` at the disposable DB.

    The production default would create/upgrade a ``hrag_checkpoints`` database;
    the tests must exercise the configured setting without touching it.
    """
    monkeypatch.setattr(settings, "CHECKPOINT_DATABASE_URL", checkpoint_dsn)
    return settings.CHECKPOINT_DATABASE_URL


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _checkpoint(*, contract_version: str, marker: str) -> dict:
    return {
        "v": 1,
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"contract_version": contract_version, "marker": marker},
        "channel_versions": {"contract_version": "1", "marker": "1"},
        "versions_seen": {},
    }


# ---------------------------------------------------------------------------
# 1. Phase-0 exact pins
# ---------------------------------------------------------------------------


def test_production_requirements_pin_phase0_checkpoint_dependencies() -> None:
    requirements = (BACKEND_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "psycopg[binary]==3.2.3" in requirements
    assert "langgraph-checkpoint-postgres==2.0.25" in requirements


def test_env_example_documents_psycopg_checkpoint_dsn() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "CHECKPOINT_DATABASE_URL=postgresql://" in env_example


# ---------------------------------------------------------------------------
# 2. Settings validation
# ---------------------------------------------------------------------------


def test_settings_default_checkpoint_dsn_is_a_distinct_psycopg_dsn() -> None:
    current = Settings(_env_file=None)
    assert current.CHECKPOINT_DATABASE_URL.startswith("postgresql://")
    assert "+" not in current.CHECKPOINT_DATABASE_URL.split("://", 1)[0]
    assert current.CHECKPOINT_DATABASE_URL != current.DATABASE_URL


def test_settings_accepts_explicit_psycopg_dsn() -> None:
    current = Settings(_env_file=None, CHECKPOINT_DATABASE_URL=VALID_PSYCOPG_DSN)
    assert current.CHECKPOINT_DATABASE_URL == VALID_PSYCOPG_DSN


def test_settings_rejects_asyncpg_scheme() -> None:
    with pytest.raises(ValidationError, match="postgresql://"):
        Settings(_env_file=None, CHECKPOINT_DATABASE_URL=ASYNC_PG_DSN)


def test_settings_rejects_absent_credentials() -> None:
    with pytest.raises(ValidationError, match="username and password"):
        Settings(
            _env_file=None,
            CHECKPOINT_DATABASE_URL="postgresql://localhost:5433/hrag_checkpoints",
        )
    with pytest.raises(ValidationError, match="username and password"):
        Settings(
            _env_file=None,
            CHECKPOINT_DATABASE_URL="postgresql://postgres@localhost:5433/hrag_checkpoints",
        )
    with pytest.raises(ValidationError, match="username and password"):
        Settings(
            _env_file=None,
            CHECKPOINT_DATABASE_URL="postgresql://:secret@localhost:5433/hrag_checkpoints",
        )


def test_settings_rejects_empty_checkpoint_dsn() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHECKPOINT_DATABASE_URL="")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHECKPOINT_DATABASE_URL="   ")


def test_settings_rejects_direct_reuse_of_database_url() -> None:
    with pytest.raises(ValidationError, match="must not reuse DATABASE_URL"):
        Settings(_env_file=None, CHECKPOINT_DATABASE_URL=settings.DATABASE_URL)


def test_settings_rejects_driver_stripped_reuse_of_database_url() -> None:
    app_dsn = "postgresql+asyncpg://postgres:postgres@localhost:5433/hrag"
    equivalent = app_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    with pytest.raises(ValidationError, match="must not reuse DATABASE_URL"):
        Settings(
            _env_file=None,
            DATABASE_URL=app_dsn,
            CHECKPOINT_DATABASE_URL=equivalent,
        )


# ---------------------------------------------------------------------------
# 3. validate_psycopg_dsn + lifecycle guards
# ---------------------------------------------------------------------------


def test_validate_psycopg_dsn_accepts_psycopg_dsn() -> None:
    assert validate_psycopg_dsn(VALID_PSYCOPG_DSN) is None


def test_validate_psycopg_dsn_rejects_asyncpg_scheme() -> None:
    with pytest.raises(ValueError, match="postgresql://"):
        validate_psycopg_dsn(ASYNC_PG_DSN)


def test_validate_psycopg_dsn_rejects_absent_credentials() -> None:
    with pytest.raises(ValueError, match="username and password"):
        validate_psycopg_dsn("postgresql://localhost:5433/hrag_test_checkpoint")


def test_validate_psycopg_dsn_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_psycopg_dsn("")


def test_validate_psycopg_dsn_rejects_database_url_reuse() -> None:
    with pytest.raises(ValueError, match="must not reuse DATABASE_URL"):
        validate_psycopg_dsn(settings.DATABASE_URL)


@pytest.mark.asyncio
async def test_create_v2_checkpointer_rejects_bad_dsn_before_connecting() -> None:
    # The asyncpg DSN would raise a psycopg connection error if it were passed
    # through; the ValueError proves validation runs before any connection.
    with pytest.raises(ValueError, match="postgresql://"):
        async with create_v2_checkpointer(ASYNC_PG_DSN):
            pytest.fail("context manager must not yield for an invalid DSN")


def test_module_has_no_import_time_instantiation_or_migration() -> None:
    tree = ast.parse(inspect.getsource(checkpoint_module))
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert top_level_calls == [], (
        "checkpoint.py must not instantiate or migrate the checkpointer at import"
    )


# ---------------------------------------------------------------------------
# 4. Live disposable-PostgreSQL lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_is_idempotent_and_creates_saver_tables(
    configured_checkpoint_dsn: str,
) -> None:
    await setup_v2_checkpointer(configured_checkpoint_dsn)
    await setup_v2_checkpointer(configured_checkpoint_dsn)

    result = await check_v2_checkpointer(configured_checkpoint_dsn)
    assert isinstance(result, CheckpointCheck)
    assert result.is_ready
    assert result.present == frozenset(CHECKPOINT_TABLES)
    assert result.missing == frozenset()


@pytest.mark.asyncio
async def test_check_reports_missing_tables_without_creating_them(
    configured_checkpoint_dsn: str,
) -> None:
    _drop_saver_tables(configured_checkpoint_dsn)

    result = await check_v2_checkpointer(configured_checkpoint_dsn)
    assert not result.is_ready
    assert result.missing == frozenset(CHECKPOINT_TABLES)
    # --check is read-only: it must not have created the tables it reports.
    assert _present_saver_tables(configured_checkpoint_dsn) == frozenset()


@pytest.mark.asyncio
async def test_setup_write_and_read_checkpoint_for_configurable_thread_id(
    configured_checkpoint_dsn: str,
) -> None:
    await setup_v2_checkpointer(configured_checkpoint_dsn)

    thread_id = f"task10-{uuid.uuid4()}"
    marker = f"marker-{uuid.uuid4()}"
    unknown_thread = f"task10-unused-{uuid.uuid4()}"

    async with create_v2_checkpointer(configured_checkpoint_dsn) as saver:
        await saver.aput(
            _config(thread_id),
            _checkpoint(contract_version=CONTRACT_VERSION, marker=marker),
            dict(_METADATA),
            dict(_NEW_VERSIONS),
        )
        loaded = await saver.aget_tuple(_config(thread_id))
        missing = await saver.aget_tuple(_config(unknown_thread))

    assert loaded is not None
    assert loaded.config["configurable"]["thread_id"] == thread_id
    assert loaded.checkpoint["channel_values"]["marker"] == marker
    assert loaded.checkpoint["channel_values"]["contract_version"] == CONTRACT_VERSION
    # A distinct thread ID keeps its own checkpoint space.
    assert missing is None


@pytest.mark.asyncio
async def test_incompatible_state_version_is_rejected(
    configured_checkpoint_dsn: str,
) -> None:
    await setup_v2_checkpointer(configured_checkpoint_dsn)

    thread_id = f"task10-version-{uuid.uuid4()}"
    async with create_v2_checkpointer(configured_checkpoint_dsn) as saver:
        await saver.aput(
            _config(thread_id),
            _checkpoint(contract_version="1.0", marker="legacy"),
            dict(_METADATA),
            dict(_NEW_VERSIONS),
        )
        loaded = await saver.aget_tuple(_config(thread_id))

    assert loaded is not None
    stored = loaded.checkpoint["channel_values"]
    assert stored["contract_version"] == "1.0"
    with pytest.raises(IncompatibleCheckpointError, match=r"1\.0"):
        validate_checkpoint_payload(stored)


# ---------------------------------------------------------------------------
# 5. CLI modes
# ---------------------------------------------------------------------------


def test_cli_setup_then_check_on_fresh_database(
    configured_checkpoint_dsn: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _drop_saver_tables(configured_checkpoint_dsn)

    assert checkpoint_main(["--check", "--dsn", configured_checkpoint_dsn]) == 1
    assert "ready=False" in capsys.readouterr().out

    assert checkpoint_main(["--setup", "--dsn", configured_checkpoint_dsn]) == 0
    assert checkpoint_main(["--check", "--dsn", configured_checkpoint_dsn]) == 0
    assert "ready=True" in capsys.readouterr().out


def test_cli_rejects_invalid_dsn(capsys: pytest.CaptureFixture[str]) -> None:
    assert checkpoint_main(["--setup", "--dsn", ASYNC_PG_DSN]) == 2
    assert "postgresql://" in capsys.readouterr().err
