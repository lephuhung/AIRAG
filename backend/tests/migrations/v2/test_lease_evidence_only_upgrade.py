"""T3 round 2 (N1) — version-1 -> version-2 lease upgrade path.

A database that applied the pre-C1 v2 schema has
``revision_retention_leases.revision_id NOT NULL`` with a version-1 row.
``apply_v2_schema`` must upgrade it in place (nullable column, version row 2),
and ``check_v2_schema`` must report a NOT NULL lease column as a shape error
(``migrate check`` exits non-zero) instead of clean.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.services.agents.v2.persistence.migrate import (
    apply_v2_schema,
    check_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)


@pytest.fixture
def db() -> Engine:
    """A SQLAlchemy ``Engine`` pointed at the v2 test database."""
    engine = make_engine(V2_DSN)
    try:
        yield engine
    finally:
        engine.dispose()


def _lease_nullable(conn) -> str | None:
    return conn.execute(
        text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = 'revision_retention_leases' "
            "AND column_name = 'revision_id'"
        )
    ).scalar()


def _recorded_version(conn) -> int | None:
    return conn.execute(
        text("SELECT version FROM v2_schema_version ORDER BY version DESC LIMIT 1")
    ).scalar()


def test_pre_c1_lease_table_upgrades_to_nullable_and_version_2(db: Engine) -> None:
    """Simulate the pre-C1 shape, run apply, assert the upgrade sticks."""
    from app.services.agents.v2.persistence.migrate import V2_SCHEMA_VERSION

    with db.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE revision_retention_leases "
                "ALTER COLUMN revision_id SET NOT NULL"
            )
        )
        conn.execute(text("UPDATE v2_schema_version SET version = 1"))
        assert _lease_nullable(conn) == "NO"
        assert _recorded_version(conn) == 1

    apply_v2_schema(db)

    with db.connect() as conn:
        assert _lease_nullable(conn) == "YES"
        assert _recorded_version(conn) == V2_SCHEMA_VERSION == 2
    assert check_v2_schema(db).is_clean is True


def test_check_detects_not_null_lease_column_as_shape_error(db: Engine) -> None:
    """A NOT NULL lease column fails the shape gate (migrate check non-zero)."""
    from app.services.agents.v2.persistence.migrate import main

    with db.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE revision_retention_leases "
                "ALTER COLUMN revision_id SET NOT NULL"
            )
        )
    try:
        check = check_v2_schema(db)
        assert check.applied is True
        assert any(
            "revision_retention_leases.revision_id" in error
            for error in check.shape_errors
        ), check.shape_errors
        assert check.is_clean is False
        assert main(["check", "--dsn", V2_DSN]) == 1
    finally:
        with db.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE revision_retention_leases "
                    "ALTER COLUMN revision_id DROP NOT NULL"
                )
            )
    assert check_v2_schema(db).is_clean is True
    assert main(["check", "--dsn", V2_DSN]) == 0
