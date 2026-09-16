"""Task 7A — version-2 -> version-3 rollout-control upgrade path.

A database at v2 schema version 2 (the T3 evidence-only-lease shape:
12 ``V2_SCHEMA_V1_TABLES`` tables, nullable lease ``revision_id``) must
upgrade in place to version 3 via the rollout DDL:

- ``agent_rollout_control`` created with the exact contract columns and
  exactly one seeded disabled row (``id = 1``).
- ``agent_rollout_metrics`` created append-only with the exact contract
  columns (inserts only, no update path).
- The upgrade is idempotent, holds the advisory lock, imports no ORM
  metadata, rejects newer versions and unsupported version gaps.

Setup regresses a migrated database back to the version-2 shape (drop
the two rollout tables, set the version row to 2) so the test exercises
the 2 -> 3 step from any starting state, independent of suite order.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.services.agents.v2.persistence.migrate import (
    V2_ROLLOUT_TABLES,
    V2_SCHEMA_V1_TABLES,
    V2_SCHEMA_V3_TABLES,
    V2_SCHEMA_VERSION,
    apply_v2_schema,
    check_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)

EXPECTED_CONTROL_COLUMNS = frozenset(
    {
        "id",
        "enabled",
        "shadow_percent",
        "canary_percent",
        "canary_workspaces",
        "kill_switch",
        "updated_at",
        "updated_by",
        "version",
    }
)

EXPECTED_METRICS_COLUMNS = frozenset(
    {
        "id",
        "arm",
        "request_id_hash",
        "workspace_id_hash",
        "started_at",
        "finished_at",
        "duration_ms",
        "terminal_status",
        "citation_count",
        "cancelled",
        "security_checkpoint_secret",
        "security_ungrounded_factual_success",
        "security_acl_leak",
        "security_duplicate_production_write",
        "created_at",
    }
)


@pytest.fixture
def db() -> Engine:
    """A SQLAlchemy ``Engine`` pointed at the v2 test database."""
    engine = make_engine(V2_DSN)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def version2_db(db: Engine) -> Engine:
    """Regress the database to the version-2 shape, then hand it over.

    Ensures the full v4 schema first (so the V1 tables exist regardless
    of suite order), then drops the two rollout tables and sets the
    version row back to 2. Restores the v4 schema on teardown so later
    tests observe a migrated database.
    """
    apply_v2_schema(db)
    with db.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS agent_rollout_metrics"))
        conn.execute(text("DROP TABLE IF EXISTS agent_rollout_control"))
        conn.execute(text("UPDATE v2_schema_version SET version = 2"))
    try:
        yield db
    finally:
        apply_v2_schema(db)


def _recorded_version(conn) -> int | None:
    return conn.execute(
        text("SELECT version FROM v2_schema_version ORDER BY version DESC LIMIT 1")
    ).scalar()


def _columns(conn, table: str) -> dict[str, str]:
    rows = conn.execute(
        text(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def test_rollout_tables_constant_is_the_v3_delta() -> None:
    assert isinstance(V2_ROLLOUT_TABLES, frozenset)
    assert V2_ROLLOUT_TABLES == frozenset(
        {"agent_rollout_control", "agent_rollout_metrics"}
    )
    assert V2_SCHEMA_V3_TABLES == V2_SCHEMA_V1_TABLES | V2_ROLLOUT_TABLES
    assert len(V2_SCHEMA_V1_TABLES) == 12
    assert len(V2_SCHEMA_V3_TABLES) == 14
    assert V2_SCHEMA_VERSION == 5


def test_version2_fixture_shape(version2_db: Engine) -> None:
    """Guard: the fixture really is the version-2 shape before upgrading."""
    with version2_db.connect() as conn:
        assert _recorded_version(conn) == 2
        tables = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            ).fetchall()
        }
    assert V2_SCHEMA_V1_TABLES <= tables
    assert not (V2_ROLLOUT_TABLES & tables), (
        f"fixture must not contain rollout tables: "
        f"{sorted(V2_ROLLOUT_TABLES & tables)}"
    )


def test_version2_upgrades_to_version4(version2_db: Engine) -> None:
    apply_v2_schema(version2_db)
    with version2_db.connect() as conn:
        assert _recorded_version(conn) == V2_SCHEMA_VERSION
        control = _columns(conn, "agent_rollout_control")
        metrics = _columns(conn, "agent_rollout_metrics")
    assert set(control) == set(EXPECTED_CONTROL_COLUMNS), (
        f"control columns mismatch: {sorted(set(control) ^ set(EXPECTED_CONTROL_COLUMNS))}"
    )
    assert set(metrics) == set(EXPECTED_METRICS_COLUMNS), (
        f"metrics columns mismatch: {sorted(set(metrics) ^ set(EXPECTED_METRICS_COLUMNS))}"
    )
    check = check_v2_schema(version2_db)
    assert check.applied is True
    assert check.version == V2_SCHEMA_VERSION
    assert check.is_clean is True, check


def test_rollout_control_seed_row_is_single_and_disabled(
    version2_db: Engine,
) -> None:
    apply_v2_schema(version2_db)
    with version2_db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, enabled, shadow_percent, canary_percent, "
                "canary_workspaces, kill_switch, updated_by, version "
                "FROM agent_rollout_control"
            )
        ).fetchall()
    assert len(rows) == 1, f"expected exactly one control row, got {len(rows)}"
    row = rows[0]
    assert row[0] == 1
    assert row[1] is False
    assert row[2] == 0
    assert row[3] == 0
    assert row[4] in ("[]", []) or list(row[4]) == []
    assert row[5] is False
    assert row[7] == 1


def test_rollout_upgrade_is_idempotent(version2_db: Engine) -> None:
    apply_v2_schema(version2_db)
    with version2_db.connect() as conn:
        n_before = conn.execute(
            text("SELECT count(*) FROM agent_rollout_control")
        ).scalar()
    apply_v2_schema(version2_db)
    with version2_db.connect() as conn:
        assert _recorded_version(conn) == V2_SCHEMA_VERSION
        n_after = conn.execute(
            text("SELECT count(*) FROM agent_rollout_control")
        ).scalar()
        n_versions = conn.execute(
            text("SELECT count(*) FROM v2_schema_version")
        ).scalar()
    assert n_before == n_after == 1
    assert n_versions == 1
    assert check_v2_schema(version2_db).is_clean is True


def test_rollout_metrics_rejects_update_and_delete(
    version2_db: Engine,
) -> None:
    """DB-level append-only: UPDATE and DELETE fail, INSERT succeeds."""
    from sqlalchemy.exc import DBAPIError

    apply_v2_schema(version2_db)
    with version2_db.begin() as conn:
        metric_id = conn.execute(
            text(
                "INSERT INTO agent_rollout_metrics "
                "(arm, request_id_hash, started_at, terminal_status) "
                "VALUES ('v2', 'req-immutable', now(), 'success') "
                "RETURNING id"
            ),
        ).scalar()
    assert metric_id is not None
    with pytest.raises(DBAPIError):
        with version2_db.begin() as conn:
            conn.execute(
                text(
                    "UPDATE agent_rollout_metrics "
                    "SET terminal_status = 'failed' WHERE id = :i"
                ),
                {"i": metric_id},
            )
    with pytest.raises(DBAPIError):
        with version2_db.begin() as conn:
            conn.execute(
                text("DELETE FROM agent_rollout_metrics WHERE id = :i"),
                {"i": metric_id},
            )
    with version2_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agent_rollout_metrics "
                "(arm, request_id_hash, started_at, terminal_status) "
                "VALUES ('v2', 'req-immutable-2', now(), 'success')"
            ),
        )
        n = conn.execute(
            text("SELECT count(*) FROM agent_rollout_metrics")
        ).scalar()
        status = conn.execute(
            text(
                "SELECT terminal_status FROM agent_rollout_metrics "
                "WHERE id = :i"
            ),
            {"i": metric_id},
        ).scalar()
    assert status == "success"
    assert n >= 2


def test_rollout_metrics_append_only_survives_reapply(
    version2_db: Engine,
) -> None:
    """Re-running apply must not drop the UPDATE/DELETE enforcement."""
    from sqlalchemy.exc import DBAPIError

    apply_v2_schema(version2_db)
    apply_v2_schema(version2_db)
    with version2_db.begin() as conn:
        metric_id = conn.execute(
            text(
                "INSERT INTO agent_rollout_metrics "
                "(arm, request_id_hash, started_at, terminal_status) "
                "VALUES ('shadow', 'req-reapply', now(), 'success') "
                "RETURNING id"
            ),
        ).scalar()
    with pytest.raises(DBAPIError):
        with version2_db.begin() as conn:
            conn.execute(
                text("DELETE FROM agent_rollout_metrics WHERE id = :i"),
                {"i": metric_id},
            )
    assert check_v2_schema(version2_db).is_clean is True


def test_rollout_metrics_is_append_only(version2_db: Engine) -> None:
    """Two inserts with identical business keys both persist: no dedup
    arbiter, no update path — the table is insert-only by construction."""
    apply_v2_schema(version2_db)
    payload = {
        "arm": "shadow",
        "request_id_hash": "req-abc",
        "workspace_id_hash": "ws-abc",
        "terminal_status": "success",
    }
    with version2_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agent_rollout_metrics "
                "(arm, request_id_hash, workspace_id_hash, started_at, "
                " terminal_status) "
                "VALUES (:arm, :request_id_hash, :workspace_id_hash, now(), "
                " :terminal_status)"
            ),
            payload,
        )
        conn.execute(
            text(
                "INSERT INTO agent_rollout_metrics "
                "(arm, request_id_hash, workspace_id_hash, started_at, "
                " terminal_status) "
                "VALUES (:arm, :request_id_hash, :workspace_id_hash, now(), "
                " :terminal_status)"
            ),
            payload,
        )
    with version2_db.connect() as conn:
        n = conn.execute(
            text(
                "SELECT count(*) FROM agent_rollout_metrics "
                "WHERE request_id_hash = 'req-abc'"
            )
        ).scalar()
    assert n == 2, f"append-only metrics must keep both rows, got {n}"


def test_rollout_metrics_arm_check_rejects_unknown_arm(
    version2_db: Engine,
) -> None:
    from sqlalchemy.exc import DBAPIError

    apply_v2_schema(version2_db)
    with pytest.raises(DBAPIError):
        with version2_db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO agent_rollout_metrics "
                    "(arm, request_id_hash, started_at, terminal_status) "
                    "VALUES ('v3', 'req-bad', now(), 'success')"
                )
            )


def test_check_does_not_report_rollout_tables_as_extra(
    version2_db: Engine,
) -> None:
    apply_v2_schema(version2_db)
    check = check_v2_schema(version2_db)
    assert check.extra_tables == frozenset(), check.extra_tables
    assert check.missing_tables == frozenset(), check.missing_tables


def test_check_reports_genuinely_missing_rollout_table(
    version2_db: Engine,
) -> None:
    """A v4 database missing one rollout table must name it in missing."""
    apply_v2_schema(version2_db)
    with version2_db.begin() as conn:
        conn.execute(text("DROP TABLE agent_rollout_metrics"))
    try:
        check = check_v2_schema(version2_db)
        assert check.applied is True
        assert check.version == V2_SCHEMA_VERSION
        assert "agent_rollout_metrics" in check.missing_tables, check
        assert check.is_clean is False
    finally:
        # Restore via the public upgrade path (apply at version 3 is a
        # no-op by design, so regress the version row first).
        with version2_db.begin() as conn:
            conn.execute(text("UPDATE v2_schema_version SET version = 2"))
        apply_v2_schema(version2_db)
    assert check_v2_schema(version2_db).is_clean is True


def test_stepwise_version1_upgrades_to_version4(version2_db: Engine) -> None:
    """The 1 -> 2 lease repair, the 2 -> 3 rollout DDL, and the 3 -> 4
    stage DDL compose: a version-1 database lands directly at 4 in a
    single apply."""
    with version2_db.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE revision_retention_leases "
                "ALTER COLUMN revision_id SET NOT NULL"
            )
        )
        conn.execute(text("UPDATE v2_schema_version SET version = 1"))
    apply_v2_schema(version2_db)
    with version2_db.connect() as conn:
        assert _recorded_version(conn) == V2_SCHEMA_VERSION
        nullable = conn.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = 'revision_retention_leases' "
                "AND column_name = 'revision_id'"
            )
        ).scalar()
        assert nullable == "YES"
        control_n = conn.execute(
            text("SELECT count(*) FROM agent_rollout_control")
        ).scalar()
        assert control_n == 1
    assert check_v2_schema(version2_db).is_clean is True


def test_apply_rejects_newer_versions(db: Engine) -> None:
    """A recorded version above V2_SCHEMA_VERSION raises; no DDL churn."""
    apply_v2_schema(db)
    with db.begin() as conn:
        conn.execute(text("UPDATE v2_schema_version SET version = 99"))
    try:
        with pytest.raises(RuntimeError, match="unsupported v2 schema version"):
            apply_v2_schema(db)
        with db.connect() as conn:
            assert _recorded_version(conn) == 99
    finally:
        with db.begin() as conn:
            conn.execute(text("UPDATE v2_schema_version SET version = 4"))


def test_apply_rejects_unsupported_version_gap(db: Engine) -> None:
    apply_v2_schema(db)
    with db.begin() as conn:
        conn.execute(text("UPDATE v2_schema_version SET version = 0"))
    try:
        with pytest.raises(RuntimeError, match="unsupported v2 schema version"):
            apply_v2_schema(db)
    finally:
        with db.begin() as conn:
            conn.execute(text("UPDATE v2_schema_version SET version = 4"))
    assert check_v2_schema(db).is_clean is True


def test_migrate_module_contract_for_rollout() -> None:
    """Source-level: advisory lock precedes DDL; no ORM imports; no
    create_all; no legacy-table UPDATE/DELETE for the rollout step."""
    src_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "services"
        / "agents"
        / "v2"
        / "persistence"
        / "migrate.py"
    )
    src = src_path.read_text(encoding="utf-8")
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)

    assert "pg_advisory_xact_lock" in code
    assert code.find("pg_advisory_xact_lock") < code.find("CREATE TABLE")
    assert "agent_rollout_control" in code
    assert "agent_rollout_metrics" in code

    code_no_strings = re.sub(r"'[^'\n]*'", "''", code)
    code_no_strings = re.sub(r'"[^"\n]*"', '""', code_no_strings)
    assert "from app.models" not in code_no_strings
    assert "import app.models" not in code_no_strings
    assert "create_all" not in code_no_strings
