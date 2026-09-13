"""P1 Task 1 RED — revision-stage migration shape and upgrade path.

Asserts the exact ``document_revision_stages`` schema installed by the
v2 migration (version 4):

- ``(revision_id, stage)`` primary/unique key.
- ``revision_id UUID REFERENCES document_revisions(revision_id)
  ON DELETE RESTRICT``.
- ``stage`` / ``state`` CHECK literals (stages: parse/embed/caption/kg;
  states: pending/running/completed/skipped/failed).
- ``attempt_count`` non-negative, ``updated_at`` NOT NULL,
  ``failure_class`` nullable.
- Idempotent re-apply and stepwise 3 -> 4 upgrade.
- ``check_v2_schema`` fails closed when the table is absent.

These tests run against ``hrag_test_v2`` (the session bootstrap in
``conftest.py`` rebuilds the current schema once per session).
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from app.services.agents.v2.persistence.migrate import (
    V2_SCHEMA_V3_TABLES,
    V2_SCHEMA_V4_TABLES,
    V2_SCHEMA_VERSION,
    V2_STAGE_TABLES,
    apply_v2_schema,
    check_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)

EXPECTED_STAGE_COLUMNS = frozenset(
    {
        "revision_id",
        "stage",
        "state",
        "attempt_count",
        "updated_at",
        "failure_class",
    }
)


@pytest.fixture
def db() -> Engine:
    engine = make_engine(V2_DSN)
    try:
        yield engine
    finally:
        engine.dispose()


def test_stage_table_constants_are_frozen() -> None:
    assert V2_STAGE_TABLES == frozenset({"document_revision_stages"})
    assert V2_SCHEMA_V4_TABLES == V2_SCHEMA_V3_TABLES | V2_STAGE_TABLES
    assert V2_SCHEMA_VERSION == 4


def test_stage_table_shape_is_exact(db: Engine) -> None:
    with db.connect() as conn:
        cols = {
            row[0]: row
            for row in conn.execute(
                text(
                    "SELECT column_name, is_nullable, data_type, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' "
                    "AND table_name = 'document_revision_stages'"
                )
            ).fetchall()
        }
    assert frozenset(cols) == EXPECTED_STAGE_COLUMNS, sorted(cols)
    assert cols["revision_id"][1] == "NO"
    assert cols["stage"][1] == "NO"
    assert cols["state"][1] == "NO"
    assert cols["attempt_count"][1] == "NO"
    assert cols["updated_at"][1] == "NO"
    assert cols["failure_class"][1] == "YES"


def test_stage_primary_key_is_revision_id_and_stage(db: Engine) -> None:
    with db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT string_agg(a.attname, ',' ORDER BY "
                "array_position(c.conkey, a.attnum)) "
                "FROM pg_constraint c "
                "JOIN pg_attribute a ON a.attrelid = c.conrelid "
                "AND a.attnum = ANY(c.conkey) "
                "WHERE c.conrelid = 'public.document_revision_stages'::regclass "
                "AND c.contype = 'p'"
            )
        ).scalar()
    assert row == "revision_id,stage", row


def test_stage_fk_is_restrict_to_revisions(db: Engine) -> None:
    with db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'public.document_revision_stages'::regclass "
                "AND contype = 'f'"
            )
        ).fetchall()
    defs = [str(r[0]) for r in rows]
    assert any(
        "REFERENCES document_revisions(revision_id)" in d
        and "ON DELETE RESTRICT" in d
        for d in defs
    ), defs


def test_stage_check_constraints_cover_exact_literals(db: Engine) -> None:
    with db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'public.document_revision_stages'::regclass "
                "AND contype = 'c'"
            )
        ).fetchall()
    defs = " ".join(str(r[0]) for r in rows)
    for literal in ("parse", "embed", "caption", "kg"):
        assert literal in defs, defs
    for literal in ("pending", "running", "completed", "skipped", "failed"):
        assert literal in defs, defs
    assert "attempt_count" in defs, defs


def test_stage_checks_reject_bad_literals_and_counts(db: Engine) -> None:
    import uuid as _uuid

    kb_id = str(_uuid.uuid4())
    doc_id = str(_uuid.uuid4())
    revision_id = str(_uuid.uuid4())
    with db.begin() as conn:
        conn.execute(
            text("INSERT INTO knowledge_bases (id, name) VALUES (:kid, 'kb')"),
            {"kid": kb_id},
        )
        conn.execute(
            text(
                "INSERT INTO documents (id, workspace_id) "
                "VALUES (:did, :kid)"
            ),
            {"did": doc_id, "kid": kb_id},
        )
        conn.execute(
            text(
                "INSERT INTO document_revisions "
                "(revision_id, document_id, generation, status) "
                "VALUES (:rid, :did, 1, 'draft')"
            ),
            {"rid": revision_id, "did": doc_id},
        )
    try:
        with db.begin() as conn:
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO document_revision_stages "
                        "(revision_id, stage, state) "
                        "VALUES (:rid, 'frobnicate', 'pending')"
                    ),
                    {"rid": str(revision_id)},
                )
        with db.begin() as conn:
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO document_revision_stages "
                        "(revision_id, stage, state) "
                        "VALUES (:rid, 'parse', 'vibing')"
                    ),
                    {"rid": str(revision_id)},
                )
        with db.begin() as conn:
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO document_revision_stages "
                        "(revision_id, stage, state, attempt_count) "
                        "VALUES (:rid, 'parse', 'pending', -1)"
                    ),
                    {"rid": str(revision_id)},
                )
    finally:
        with db.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM document_revision_stages WHERE revision_id = :rid"
                ),
                {"rid": revision_id},
            )
            conn.execute(
                text("DELETE FROM document_revisions WHERE revision_id = :rid"),
                {"rid": revision_id},
            )
            conn.execute(
                text("DELETE FROM documents WHERE id = :did"),
                {"did": doc_id},
            )
            conn.execute(
                text("DELETE FROM knowledge_bases WHERE id = :kid"),
                {"kid": kb_id},
            )


def test_stage_fk_rejects_unknown_revision(db: Engine) -> None:
    with db.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO document_revision_stages "
                    "(revision_id, stage, state) "
                    "VALUES ('00000000-0000-0000-0000-000000000000', "
                    "'parse', 'pending')"
                )
            )


def test_migration_reapply_is_idempotent(db: Engine) -> None:
    apply_v2_schema(db)
    apply_v2_schema(db)
    check = check_v2_schema(db)
    assert check.applied is True, check
    assert check.version == V2_SCHEMA_VERSION, check
    assert check.is_clean is True, check


def test_v3_to_v4_upgrade_creates_stage_table(db: Engine) -> None:
    with db.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS document_revision_stages"))
        conn.execute(text("UPDATE v2_schema_version SET version = 3"))
    try:
        apply_v2_schema(db)
        check = check_v2_schema(db)
        assert check.version == 4, check
        assert check.is_clean is True, check
        with db.connect() as conn:
            present = conn.execute(
                text(
                    "SELECT to_regclass('public.document_revision_stages')"
                    " IS NOT NULL"
                )
            ).scalar()
        assert present is True
    finally:
        apply_v2_schema(db)


def test_check_fails_closed_without_stage_table(db: Engine) -> None:
    with db.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS document_revision_stages"))
    try:
        check = check_v2_schema(db)
        assert check.applied is True, check
        assert check.is_clean is False, check
        assert "document_revision_stages" in check.missing_tables, check
    finally:
        apply_v2_schema(db)
