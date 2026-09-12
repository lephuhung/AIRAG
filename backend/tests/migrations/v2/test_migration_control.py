"""Control test for the v2 schema migration.

Asserts the *exact* schema delta introduced by Release 1A. Only the 12
``V2_SCHEMA_V1_TABLES`` may be added; no legacy tables may be touched;
no hidden table may sneak in that Release 1B later maps.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from app.services.agents.v2.persistence.migrate import (
    V2_SCHEMA_V1_TABLES,
    apply_v2_schema,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)


def _table_names(conn: psycopg.Connection) -> frozenset[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        return frozenset(row[0] for row in cur.fetchall())


@pytest.fixture
def db() -> psycopg.Connection:
    conn = psycopg.connect(V2_DSN, autocommit=False)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def test_v2_schema_v1_tables_constant_is_frozen_set() -> None:
    assert isinstance(V2_SCHEMA_V1_TABLES, frozenset)
    assert len(V2_SCHEMA_V1_TABLES) == 12


def test_schema_delta_is_exactly_v2_tables(db: psycopg.Connection) -> None:
    """Capture before, apply migration, capture after, assert exact delta."""
    # Drop any prior v2 state so the test starts from a clean baseline.
    with db.cursor() as cur:
        for tbl in V2_SCHEMA_V1_TABLES:
            cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
        cur.execute("ALTER TABLE documents DROP COLUMN IF EXISTS current_revision_id")
        cur.execute("ALTER TABLE documents DROP COLUMN IF EXISTS source_deleted_at")
        cur.execute("ALTER TABLE document_images DROP COLUMN IF EXISTS revision_id")
        cur.execute("ALTER TABLE document_tables DROP COLUMN IF EXISTS revision_id")
    db.commit()

    before = _table_names(db)
    apply_v2_schema(db)
    db.commit()
    after = _table_names(db)

    added = after - before
    removed = before - after

    assert added == V2_SCHEMA_V1_TABLES, (
        f"schema delta mismatch: added={sorted(added)}, expected={sorted(V2_SCHEMA_V1_TABLES)}"
    )
    assert not removed, (
        f"migration removed legacy tables (forbidden): {sorted(removed)}"
    )


def test_schema_version_row_present(db: psycopg.Connection) -> None:
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM v2_schema_version WHERE version = 1")
        assert cur.fetchone()[0] == 1
