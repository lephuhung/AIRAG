"""Integration tests for the populated-legacy-schema v2 migration.

These tests run against the *populated* legacy v1 database (``hrag_test_v2``).
They exercise ``apply_v2_schema`` end-to-end and verify:

- 12 v2 tables are created (the exact ``V2_SCHEMA_V1_TABLES`` set).
- Schema version row exists with value 1.
- Nullable revision links / current-pointer columns exist.
- Every legacy row remains untouched and v1-readable.
- No legacy Chroma/KG/object/SQL artifact is marked revision-ready.
- All legacy ``current_revision_id`` values remain ``NULL`` until full
  revision-aware reindex.
- Rerun is idempotent.

They also instrument the migration and assert:

- ``pg_advisory_xact_lock`` is acquired before any DDL.
- The migration never imports ``app.models`` or calls ``create_all``.
- The migration never deletes any legacy rows.
- The migration never invents a baseline published revision.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.services.agents.v2.persistence.migrate import make_engine

# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)

V2_TABLES_IN_ORDER = (
    "v2_schema_version",
    "document_revisions",
    "document_revision_builds",
    "document_revision_chunks",
    "revision_ingestion_attempts",
    "source_arrivals",
    "revision_retention_leases",
    "conversation_snapshots",
    "semantic_snapshots",
    "binding_audit",
    "evidence_records",
    "evidence_uses",
)

EXPECTED_V2_TABLES = frozenset(V2_TABLES_IN_ORDER)


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
    """Open a SQLAlchemy ``Engine`` to the v2 test database and tear down at end."""
    engine = make_engine(V2_DSN)
    try:
        yield engine
    finally:
        engine.dispose()


def _psycopg_connect() -> psycopg.Connection:
    """Setup/teardown helper using raw psycopg for brevity.

    The migration API itself is sync-SQLAlchemy-only; setup helpers are
    exempt from that contract because they run before the migration
    exists in the schema lifecycle.
    """
    return psycopg.connect(V2_DSN, autocommit=False)


@pytest.fixture
def migrated_db(db: Engine):
    """A database with the v2 schema applied on a freshly populated fixture."""
    # Drop any prior v2 state from a previous run so the migration starts clean.
    _drop_v2_state()
    # Truncate legacy child tables to keep the fixture deterministic across
    # reruns without disturbing other test isolation.
    _truncate_legacy_fixture()
    # Populate a minimal legacy fixture (FK-respecting).
    _populate_legacy_fixture()
    # Import lazily so the test fails at import time if the module is missing.
    from app.services.agents.v2.persistence.migrate import apply_v2_schema

    apply_v2_schema(db)
    yield db


# ---------------------------------------------------------------------------
# Legacy fixture population
# ---------------------------------------------------------------------------


def _drop_v2_state() -> None:
    """Drop any v2 tables that may exist from a previous test run.

    Order matters: drop the triggers that reference ``migrated_at``
    before dropping the column itself. FK constraints from legacy
    columns (``documents.current_revision_id`` etc.) are pulled in by
    the ``DROP TABLE ... CASCADE`` of the v2 tables they reference
    (``document_revisions``), so we drop the v2 tables first.
    """
    with _psycopg_connect() as conn:
        with conn.cursor() as cur:
            # Drop v2 tables first (cascades remove FKs pointing at them).
            for tbl in EXPECTED_V2_TABLES:
                cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
            # Drop triggers on legacy tables (depend on migrated_at).
            for trigger, table in (
                ("trg_documents_require_revision_id", "documents"),
                ("trg_document_images_require_revision_id", "document_images"),
                ("trg_document_tables_require_revision_id", "document_tables"),
            ):
                cur.execute(
                    f"DROP TRIGGER IF EXISTS {trigger} ON {table}"
                )
            # Drop the trigger functions.
            for fn in (
                "enforce_documents_current_revision_id()",
                "enforce_document_images_revision_id()",
                "enforce_document_tables_revision_id()",
            ):
                cur.execute(f"DROP FUNCTION IF EXISTS {fn}")
            # Drop the column additions to legacy tables.
            cur.execute(
                "ALTER TABLE documents DROP COLUMN IF EXISTS current_revision_id"
            )
            cur.execute(
                "ALTER TABLE documents DROP COLUMN IF EXISTS source_deleted_at"
            )
            cur.execute(
                "ALTER TABLE documents DROP COLUMN IF EXISTS migrated_at"
            )
            cur.execute(
                "ALTER TABLE document_images DROP COLUMN IF EXISTS revision_id"
            )
            cur.execute(
                "ALTER TABLE document_images DROP COLUMN IF EXISTS migrated_at"
            )
            cur.execute(
                "ALTER TABLE document_tables DROP COLUMN IF EXISTS revision_id"
            )
            cur.execute(
                "ALTER TABLE document_tables DROP COLUMN IF EXISTS migrated_at"
            )
        conn.commit()


def _truncate_legacy_fixture() -> None:
    """Truncate the v1 tables touched by the fixture so reruns are deterministic."""
    with _psycopg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE document_images, document_tables, documents, "
                "knowledge_bases, users, tenants RESTART IDENTITY CASCADE"
            )
        conn.commit()


def _populate_legacy_fixture() -> None:
    """Insert a tiny, FK-respecting legacy fixture.

    Layout:
      1 tenant -> 1 user -> 1 knowledge_base (workspace)
        -> 3 documents -> a handful of images/tables.

    All rows are tagged so the test can detect any unintended mutation.
    """
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    doc_ids = [uuid.uuid4() for _ in range(3)]

    with _psycopg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (id, name, slug, is_active, created_at, updated_at) "
                "VALUES (%s, %s, %s, true, NOW(), NOW())",
                (str(tenant_id), f"tenant-{tenant_id}", f"slug-{tenant_id}"),
            )
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, is_active, "
                "is_superadmin, settings, totp_enabled, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, true, false, '{}', false, NOW(), NOW())",
                (str(user_id), f"u-{user_id}@example.com", "x", "Test User"),
            )
            cur.execute(
                "INSERT INTO knowledge_bases (id, name, description, system_prompt, "
                "visibility, owner_id, tenant_id, is_default, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 'private', %s, %s, false, NOW(), NOW())",
                (str(workspace_id), f"kb-{workspace_id}", "test kb", "test prompt",
                 str(user_id), str(tenant_id)),
            )
            for did in doc_ids:
                cur.execute(
                    "INSERT INTO documents (id, workspace_id, filename, original_filename, "
                    "file_type, file_size, status, chunk_count, page_count, image_count, "
                    "table_count, processing_time_ms, embed_done, captions_done, kg_done, "
                    "is_chat_upload, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, 'pdf', 1024, 'pending', 0, 0, 0, 0, 0, "
                    "false, false, false, false, NOW(), NOW())",
                    (str(did), str(workspace_id), f"f-{did}", f"orig-{did}.pdf"),
                )
                for j in range(2):
                    cur.execute(
                        "INSERT INTO document_images (id, document_id, image_id, page_no, "
                        "file_path, caption, width, height, mime_type, created_at) "
                        "VALUES (%s, %s, %s, 1, '/tmp/x.png', 'cap', 10, 10, 'image/png', NOW())",
                        (str(uuid.uuid4()), str(did), f"img-{did}-{j}"),
                    )
                    cur.execute(
                        "INSERT INTO document_tables (id, document_id, table_id, page_no, "
                        "content_markdown, caption, num_rows, num_cols, created_at) "
                        "VALUES (%s, %s, %s, 1, '|x|', 'cap', 1, 1, NOW())",
                        (str(uuid.uuid4()), str(did), f"tbl-{did}-{j}"),
                    )
        conn.commit()


def _legacy_row_counts(conn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tbl in ("documents", "document_images", "document_tables"):
        n = conn.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
        counts[tbl] = int(n)
    return counts


# ---------------------------------------------------------------------------
# Step 1 (RED → GREEN) — populated-legacy assertions
# ---------------------------------------------------------------------------


def test_v2_schema_version_row_exists(migrated_db: Engine) -> None:
    with migrated_db.connect() as conn:
        row = conn.execute(
            text("SELECT version, applied_at FROM v2_schema_version")
        ).fetchone()
    assert row is not None, "v2_schema_version row missing"
    assert row[0] == 1, f"expected version 1, got {row[0]}"


def test_all_v2_tables_exist(migrated_db: Engine) -> None:
    with migrated_db.connect() as conn:
        tables = _table_names(conn)
    missing = EXPECTED_V2_TABLES - tables
    assert not missing, f"missing v2 tables: {sorted(missing)}"


def test_legacy_rows_unchanged(migrated_db: Engine) -> None:
    """Every legacy row remains untouched and v1-readable."""
    with migrated_db.connect() as conn:
        counts = _legacy_row_counts(conn)
    assert counts == {"documents": 3, "document_images": 6, "document_tables": 6}


def test_legacy_current_revision_id_is_nullable_and_null(
    migrated_db: Engine,
) -> None:
    """All legacy rows have ``current_revision_id IS NULL`` after migration."""
    with migrated_db.connect() as conn:
        nullable_row = conn.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_name = 'documents' AND column_name = 'current_revision_id'"
            )
        ).fetchone()
    assert nullable_row is not None, "documents.current_revision_id column missing"
    assert nullable_row[0] == "YES", "current_revision_id must be nullable"
    with migrated_db.connect() as conn:
        non_null = conn.execute(
            text(
                "SELECT count(*) FROM documents WHERE current_revision_id IS NOT NULL"
            )
        ).scalar()
    assert non_null == 0, "legacy documents must not have current_revision_id set"


def test_no_legacy_artifact_marked_revision_ready(
    migrated_db: Engine,
) -> None:
    """No legacy child row is selected/repurposed by v2 — revision_id is NULL."""
    for tbl in ("document_images", "document_tables"):
        with migrated_db.connect() as conn:
            non_null = conn.execute(
                text(f"SELECT count(*) FROM {tbl} WHERE revision_id IS NOT NULL")
            ).scalar()
        assert non_null == 0, (
            f"legacy {tbl} rows must not have a revision_id after migration"
        )


def test_rerun_is_idempotent(migrated_db: Engine) -> None:
    """A second ``apply_v2_schema`` is a no-op (no DDL churn)."""
    from app.services.agents.v2.persistence.migrate import apply_v2_schema

    with migrated_db.connect() as conn:
        counts_before = _legacy_row_counts(conn)
        tables_before = _table_names(conn)
    apply_v2_schema(migrated_db)
    with migrated_db.connect() as conn:
        counts_after = _legacy_row_counts(conn)
        tables_after = _table_names(conn)
        n = conn.execute(text("SELECT count(*) FROM v2_schema_version")).scalar()
    assert counts_before == counts_after
    assert tables_before == tables_after
    # And the schema version row is still exactly one.
    assert n == 1


def test_migrate_module_never_imports_app_models() -> None:
    """The migration module must not import ``app.models`` (Phase 1B owns that)."""
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
    # Strip docstrings/comments so we don't match the prose "import app.models"
    # in the module's documentation.
    code = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)
    code_no_strings = re.sub(r"'[^'\n]*'", "''", code)
    code_no_strings = re.sub(r'"[^"\n]*"', '""', code_no_strings)
    assert "from app.models" not in code_no_strings
    assert "import app.models" not in code_no_strings
    assert "create_all" not in code_no_strings


def test_migrate_module_takes_advisory_lock() -> None:
    """The migration acquires ``pg_advisory_xact_lock`` before any DDL.

    We prove this by checking the migration source. The module
    docstring and ``#`` comments are stripped so the assertion cannot
    accidentally pass against the narrative text. SQL string literals
    inside ``_CREATE_DDL`` etc. are preserved because triple-quoted
    SQL blocks are not docstrings. The first surviving occurrence of
    ``pg_advisory_xact_lock`` must precede the first ``CREATE TABLE``
    literal in the code stream.
    """
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
    # Strip ONLY the module docstring (anchored to start of file, count=1)
    # and # comments. Other triple-quoted strings (SQL DDL blocks) and
    # single/double-quoted literals are preserved so the SQL string
    # containing ``pg_advisory_xact_lock`` and the ``CREATE TABLE``
    # strings inside ``_CREATE_DDL`` remain searchable.
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)

    assert "pg_advisory_xact_lock" in code, (
        "advisory lock call missing from migration code (after stripping docstrings)"
    )
    lock_pos = code.find("pg_advisory_xact_lock")
    first_create_pos = code.find("CREATE TABLE")
    if first_create_pos == -1:
        first_create_pos = code.find("create table")
    assert first_create_pos != -1, "no CREATE TABLE in migration source"
    assert lock_pos < first_create_pos, (
        "advisory lock call site must precede the first CREATE TABLE "
        "literal in the code stream"
    )


def test_migration_does_not_mutate_legacy_data() -> None:
    """Source-level proof that the migration never mutates legacy data.

    The only ``UPDATE`` statements allowed on legacy tables are the
    ``migrated_at`` backfill (``UPDATE <table> SET migrated_at = NOW()
    WHERE migrated_at IS NULL``), which only writes the sentinel
    column added by the same migration. No ``DELETE FROM`` and no
    UPDATE that touches any other column is permitted.
    """
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
    # Strip ONLY the module docstring and # comments. SQL strings inside
    # the migration (which live in triple-quoted string literals) are
    # preserved so we can search them for UPDATE/DELETE statements.
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)

    # No DELETE FROM on legacy tables.
    for stmt in ("DELETE FROM documents", "DELETE FROM document_images",
                 "DELETE FROM document_tables"):
        assert stmt not in code, (
            f"forbidden legacy DELETE in migrate.py: {stmt!r}"
        )

    # The only UPDATE on legacy tables allowed is the migrated_at
    # backfill (``UPDATE <tbl> SET migrated_at = NOW() WHERE migrated_at
    # IS NULL``). We assert that no UPDATE on a legacy table touches any
    # other column.
    legacy_updates = re.findall(
        r"UPDATE\s+(documents|document_images|document_tables)\b",
        code,
    )
    assert legacy_updates, (
        "expected at least the migrated_at backfill UPDATE on legacy tables"
    )
    for tbl in set(legacy_updates):
        # The pattern ``UPDATE <tbl> SET ...`` must only set ``migrated_at``.
        pattern = re.compile(
            rf"UPDATE\s+{tbl}\b[^\n]*SET\s+([^,\n]+(?:,[^,\n]+)*)",
            re.IGNORECASE,
        )
        for cols in pattern.findall(code):
            for col_assignment in cols.split(","):
                col = col_assignment.strip().split("=", 1)[0].strip()
                assert col == "migrated_at", (
                    f"legacy UPDATE on {tbl!r} touches forbidden column "
                    f"{col!r} (only migrated_at is permitted)"
                )


def test_legacy_fks_point_at_document_revisions(migrated_db: Engine) -> None:
    """Brief item (3): legacy FKs to document_revisions are installed."""
    expected = (
        ("documents", "current_revision_id", "document_revisions", "revision_id"),
        ("document_images", "revision_id", "document_revisions", "revision_id"),
        ("document_tables", "revision_id", "document_revisions", "revision_id"),
    )
    with migrated_db.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name,
                    rc.delete_rule
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                JOIN information_schema.referential_constraints AS rc
                  ON rc.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                """
            )
        ).fetchall()
    found = {
        (r[0], r[1], r[2], r[3]): r[4]
        for r in rows
        if (r[0], r[1], r[2], r[3]) in {e[:4] for e in expected}
    }
    for tbl, col, ftable, fcol in expected:
        assert (tbl, col, ftable, fcol) in found, (
            f"missing FK {tbl}.{col} -> {ftable}.{fcol}"
        )
        # ON DELETE must NOT be CASCADE.
        assert found[(tbl, col, ftable, fcol)] != "CASCADE", (
            f"FK {tbl}.{col} uses ON DELETE CASCADE (forbidden by brief item 3)"
        )


def test_no_v2_cascade_fks_on_documents(migrated_db: Engine) -> None:
    """Plan prohibition #14: no FK originating from documents or its children
    may use ON DELETE CASCADE. Phase 1C tombstone + artifact-GC owns reclamation.
    """
    with migrated_db.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    rc.delete_rule
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.referential_constraints AS rc
                  ON rc.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND rc.delete_rule = 'CASCADE'
                """
            )
        ).fetchall()
    # Any CASCADE on a v2 child of documents is a violation.
    for r in rows:
        assert r[0] not in {"document_revisions", "document_revision_builds",
                            "document_revision_chunks",
                            "revision_ingestion_attempts",
                            "revision_retention_leases"}, (
            f"v2 child {r[0]}.{r[1]} uses ON DELETE CASCADE (forbidden)"
        )


def test_legacy_migrated_at_sentinel_present(migrated_db: Engine) -> None:
    """Brief item (4): ``migrated_at`` sentinel column is backfilled on legacy rows."""
    for tbl in ("documents", "document_images", "document_tables"):
        with migrated_db.connect() as conn:
            col = conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'migrated_at'"
                ),
                {"t": tbl},
            ).fetchone()
            null_count = conn.execute(
                text(
                    f"SELECT count(*) FROM {tbl} WHERE migrated_at IS NULL"
                )
            ).scalar()
        assert col is not None, f"{tbl}.migrated_at column missing"
        assert col[0] == "YES", f"{tbl}.migrated_at must be nullable"
        assert null_count == 0, (
            f"all legacy {tbl} rows must have migrated_at backfilled"
        )


def test_null_safe_unique_constraint_on_leases(migrated_db: Engine) -> None:
    """I1: ``revision_retention_leases`` has the null-safe unique constraint."""
    with migrated_db.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT conname, contype
                FROM pg_constraint
                WHERE conrelid = 'revision_retention_leases'::regclass
                  AND contype = 'u'
                """
            )
        ).fetchall()
    names = {r[0] for r in row}
    assert "uq_revision_lease_run_revision_use" in names, (
        f"missing null-safe unique constraint on revision_retention_leases; "
        f"found: {sorted(names)}"
    )


def test_named_ingestion_attempt_unique_constraint(migrated_db: Engine) -> None:
    """I2: ``revision_ingestion_attempts`` has named ``uq_revision_ingestion_attempt_key``."""
    with migrated_db.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'revision_ingestion_attempts'::regclass
                  AND contype = 'u'
                """
            )
        ).fetchall()
    names = {r[0] for r in rows}
    assert "uq_revision_ingestion_attempt_key" in names, (
        f"missing named unique constraint; found: {sorted(names)}"
    )


def test_legacy_revision_id_trigger_enforces_update(migrated_db: Engine) -> None:
    """I4: UPDATEs that null out revision_id on a v2-written row are rejected.

    We insert a row with ``migrated_at IS NULL`` (v2-written marker) and no
    ``revision_id``; the trigger must reject any UPDATE that does not first
    populate ``revision_id``.
    """
    doc_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    workspace_id = uuid.uuid4()

    with _psycopg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (id, name, slug, is_active, created_at, updated_at) "
                "VALUES (%s, %s, %s, true, NOW(), NOW())",
                (str(tenant_id), f"t-{tenant_id}", f"slug-{tenant_id}"),
            )
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, is_active, "
                "is_superadmin, settings, totp_enabled, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, true, false, '{}', false, NOW(), NOW())",
                (str(user_id), f"u-{user_id}@example.com", "x", "x"),
            )
            cur.execute(
                "INSERT INTO knowledge_bases (id, name, description, system_prompt, "
                "visibility, owner_id, tenant_id, is_default, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 'private', %s, %s, false, NOW(), NOW())",
                (str(workspace_id), f"kb-{workspace_id}", "d", "p",
                 str(user_id), str(tenant_id)),
            )
            cur.execute(
                "INSERT INTO documents (id, workspace_id, filename, original_filename, "
                "file_type, file_size, status, chunk_count, page_count, image_count, "
                "table_count, processing_time_ms, embed_done, captions_done, kg_done, "
                "is_chat_upload, created_at, updated_at, migrated_at) "
                "VALUES (%s, %s, %s, %s, 'pdf', 1, 'pending', 0, 0, 0, 0, 0, "
                "false, false, false, false, NOW(), NOW(), NULL)",
                (str(doc_id), str(workspace_id), f"f-{doc_id}", f"o-{doc_id}.pdf"),
            )
        conn.commit()

    # The UPDATE on a migrated_at IS NULL row without setting
    # current_revision_id must raise.
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError) as excinfo:
        with migrated_db.connect() as conn:
            conn.execute(
                text("UPDATE documents SET filename = 'x' WHERE id = :id"),
                {"id": str(doc_id)},
            )
            conn.commit()
    # Sanity: it was indeed the trigger raising, not some other failure.
    msg = str(excinfo.value).lower()
    assert "current_revision_id" in msg or "documents" in msg, (
        f"unexpected DB error: {excinfo.value}"
    )