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
        # Restore reliably even if the body failed midway (M1 pattern):
        # a bare re-apply is a no-op at version 4, so force the
        # stepwise 3 -> 4 upgrade only when the table is still absent.
        with db.begin() as conn:
            present = conn.execute(
                text(
                    "SELECT to_regclass('public.document_revision_stages')"
                    " IS NOT NULL"
                )
            ).scalar()
            if not present:
                conn.execute(
                    text(
                        "UPDATE v2_schema_version SET version = 3 "
                        "WHERE version = 4"
                    )
                )
        apply_v2_schema(db)


def _stage_constraints(db: Engine) -> list[tuple[str, str, str]]:
    """``(conname, contype, definition)`` for the live stage table."""
    with db.connect() as conn:
        return [
            (str(r[0]), str(r[1]), str(r[2]))
            for r in conn.execute(
                text(
                    "SELECT c.conname, c.contype, "
                    "pg_get_constraintdef(c.oid) "
                    "FROM pg_constraint c "
                    "WHERE c.conrelid = "
                    "'public.document_revision_stages'::regclass"
                )
            ).fetchall()
        ]


def _restore_stage_constraints(db: Engine) -> None:
    """Re-add any missing stage PK / RESTRICT FK / literal CHECKs.

    M2 teardown: ``apply_v2_schema`` only creates a missing *table*, so a
    test that drops a single constraint must restore that constraint
    itself (idempotent: only missing pieces are re-added)."""
    existing = _stage_constraints(db)
    kinds = {contype for _, contype, _ in existing}
    names = {name for name, _, _ in existing}
    stmts: list[str] = []
    if "p" not in kinds:
        stmts.append(
            "ALTER TABLE document_revision_stages "
            "ADD PRIMARY KEY (revision_id, stage)"
        )
    if "f" not in kinds:
        stmts.append(
            "ALTER TABLE document_revision_stages "
            "ADD FOREIGN KEY (revision_id) "
            "REFERENCES document_revisions(revision_id) ON DELETE RESTRICT"
        )
    if "document_revision_stages_stage_check" not in names:
        stmts.append(
            "ALTER TABLE document_revision_stages "
            "ADD CONSTRAINT document_revision_stages_stage_check "
            "CHECK (stage IN ('parse', 'embed', 'caption', 'kg'))"
        )
    if "document_revision_stages_state_check" not in names:
        stmts.append(
            "ALTER TABLE document_revision_stages "
            "ADD CONSTRAINT document_revision_stages_state_check "
            "CHECK (state IN "
            "('pending', 'running', 'completed', 'skipped', 'failed'))"
        )
    if "document_revision_stages_attempt_count_check" not in names:
        stmts.append(
            "ALTER TABLE document_revision_stages "
            "ADD CONSTRAINT document_revision_stages_attempt_count_check "
            "CHECK (attempt_count >= 0)"
        )
    if stmts:
        with db.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))


def _stage_fk_name(db: Engine) -> str:
    for name, contype, _ in _stage_constraints(db):
        if contype == "f":
            return name
    raise AssertionError("stage table has no FOREIGN KEY constraint")


def test_check_reports_no_stage_shape_errors_on_healthy_schema(
    db: Engine,
) -> None:
    check = check_v2_schema(db)
    assert check.is_clean is True, check
    assert not [
        error for error in check.shape_errors
        if "document_revision_stages" in error
    ], check.shape_errors


def test_check_fails_closed_when_stage_pk_missing(db: Engine) -> None:
    with db.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE document_revision_stages "
                "DROP CONSTRAINT document_revision_stages_pkey"
            )
        )
    try:
        check = check_v2_schema(db)
        assert check.is_clean is False, check
        assert any(
            "document_revision_stages" in error
            and "PRIMARY KEY" in error
            for error in check.shape_errors
        ), check.shape_errors
    finally:
        _restore_stage_constraints(db)
        assert check_v2_schema(db).is_clean is True


def test_check_fails_closed_when_stage_fk_missing(db: Engine) -> None:
    fk_name = _stage_fk_name(db)
    with db.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE document_revision_stages DROP CONSTRAINT {fk_name}"
            )
        )
    try:
        check = check_v2_schema(db)
        assert check.is_clean is False, check
        assert any(
            "document_revision_stages" in error
            and "FOREIGN KEY" in error
            for error in check.shape_errors
        ), check.shape_errors
    finally:
        _restore_stage_constraints(db)
        assert check_v2_schema(db).is_clean is True


def test_check_fails_closed_when_stage_fk_not_restrict(db: Engine) -> None:
    fk_name = _stage_fk_name(db)
    with db.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE document_revision_stages DROP CONSTRAINT {fk_name}"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE document_revision_stages "
                f"ADD CONSTRAINT {fk_name} FOREIGN KEY (revision_id) "
                "REFERENCES document_revisions(revision_id) ON DELETE CASCADE"
            )
        )
    try:
        check = check_v2_schema(db)
        assert check.is_clean is False, check
        assert any(
            "document_revision_stages" in error and "RESTRICT" in error
            for error in check.shape_errors
        ), check.shape_errors
    finally:
        with db.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE document_revision_stages "
                    f"DROP CONSTRAINT IF EXISTS {fk_name}"
                )
            )
        _restore_stage_constraints(db)
        assert check_v2_schema(db).is_clean is True


def test_check_fails_closed_when_stage_state_check_missing(
    db: Engine,
) -> None:
    with db.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE document_revision_stages "
                "DROP CONSTRAINT document_revision_stages_state_check"
            )
        )
    try:
        check = check_v2_schema(db)
        assert check.is_clean is False, check
        assert any(
            "document_revision_stages" in error and "CHECK" in error
            for error in check.shape_errors
        ), check.shape_errors
    finally:
        _restore_stage_constraints(db)
        assert check_v2_schema(db).is_clean is True


def test_check_fails_closed_without_stage_table(db: Engine) -> None:
    with db.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS document_revision_stages"))
    try:
        check = check_v2_schema(db)
        assert check.applied is True, check
        assert check.is_clean is False, check
        assert "document_revision_stages" in check.missing_tables, check
    finally:
        # ``apply_v2_schema`` is a no-op when the version row already reads
        # 4, so force the stepwise 3 -> 4 upgrade to reliably recreate the
        # dropped table (M1: a bare re-apply never restores it).
        with db.begin() as conn:
            conn.execute(text("UPDATE v2_schema_version SET version = 3"))
        apply_v2_schema(db)
        restored = check_v2_schema(db)
        assert restored.is_clean is True, restored
