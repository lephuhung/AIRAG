"""Control test for the v2 schema migration.

Asserts the *exact* schema delta introduced by Release 1A. Only the 12
``V2_SCHEMA_V1_TABLES`` may be added; no legacy tables may be touched;
no hidden table may sneak in that Release 1B later maps.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.services.agents.v2.persistence.migrate import (
    V2_SCHEMA_V1_TABLES,
    apply_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)


def _table_names(conn) -> frozenset[str]:
    rows = conn.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
    ).fetchall()
    return frozenset(row[0] for row in rows)


@pytest.fixture
def db() -> Engine:
    """A SQLAlchemy ``Engine`` pointed at the v2 test database.

    The migration runner takes an ``Engine``; tests that need raw
    connection access call ``engine.connect()`` inside the test.
    """
    engine = make_engine(V2_DSN)
    try:
        yield engine
    finally:
        engine.dispose()


def _psycopg_connect() -> psycopg.Connection:
    """Legacy psycopg connection used only for setup/teardown DDL.

    The migration runner is sync-SQLAlchemy-only; setup helpers may use
    psycopg for brevity, but the migration itself is always driven via
    the ``Engine`` API.
    """
    return psycopg.connect(V2_DSN, autocommit=False)


def test_v2_schema_v1_tables_constant_is_frozen_set() -> None:
    assert isinstance(V2_SCHEMA_V1_TABLES, frozenset)
    assert len(V2_SCHEMA_V1_TABLES) == 12


def test_schema_delta_is_exactly_v2_tables(db: Engine) -> None:
    """Capture before, apply migration, capture after, assert exact delta."""
    # Drop any prior v2 state so the test starts from a clean baseline.
    # Setup uses a raw psycopg connection because it predates the migration
    # API; the migration itself runs through the public Engine contract.
    with _psycopg_connect() as setup:
        with setup.cursor() as cur:
            for tbl in V2_SCHEMA_V1_TABLES:
                cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
            cur.execute(
                "ALTER TABLE documents DROP COLUMN IF EXISTS current_revision_id"
            )
            cur.execute(
                "ALTER TABLE documents DROP COLUMN IF EXISTS source_deleted_at"
            )
            cur.execute(
                "ALTER TABLE document_images DROP COLUMN IF EXISTS revision_id"
            )
            cur.execute(
                "ALTER TABLE document_tables DROP COLUMN IF EXISTS revision_id"
            )
        setup.commit()

    with db.connect() as conn:
        before = _table_names(conn)

    apply_v2_schema(db)

    with db.connect() as conn:
        after = _table_names(conn)

    added = after - before
    removed = before - after

    assert added == V2_SCHEMA_V1_TABLES, (
        f"schema delta mismatch: added={sorted(added)}, expected={sorted(V2_SCHEMA_V1_TABLES)}"
    )
    assert not removed, (
        f"migration removed legacy tables (forbidden): {sorted(removed)}"
    )


def test_schema_version_row_present(db: Engine) -> None:
    with db.connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM v2_schema_version WHERE version = 1")
        ).scalar()
    assert n == 1