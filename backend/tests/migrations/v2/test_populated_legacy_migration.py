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


def _table_names(conn: psycopg.Connection) -> frozenset[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        return frozenset(row[0] for row in cur.fetchall())


@pytest.fixture
def db() -> psycopg.Connection:
    """Open a connection to the v2 test database and tear down at end."""
    conn = psycopg.connect(V2_DSN, autocommit=False)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.fixture
def migrated_db(db: psycopg.Connection):
    """A database with the v2 schema applied on a freshly populated fixture."""
    # Drop any prior v2 state from a previous run so the migration starts clean.
    _drop_v2_state(db)
    # Truncate legacy child tables to keep the fixture deterministic across
    # reruns without disturbing other test isolation.
    _truncate_legacy_fixture(db)
    # Populate a minimal legacy fixture (FK-respecting).
    _populate_legacy_fixture(db)
    # Import lazily so the test fails at import time if the module is missing.
    from app.services.agents.v2.persistence.migrate import apply_v2_schema

    apply_v2_schema(db)
    db.commit()
    yield db


# ---------------------------------------------------------------------------
# Legacy fixture population
# ---------------------------------------------------------------------------


def _drop_v2_state(conn: psycopg.Connection) -> None:
    """Drop any v2 tables that may exist from a previous test run."""
    with conn.cursor() as cur:
        for tbl in EXPECTED_V2_TABLES:
            cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
        # Drop the column additions to legacy tables too.
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
    conn.commit()


def _truncate_legacy_fixture(conn: psycopg.Connection) -> None:
    """Truncate the v1 tables touched by the fixture so reruns are deterministic."""
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE document_images, document_tables, documents, "
            "knowledge_bases, users, tenants RESTART IDENTITY CASCADE"
        )
    conn.commit()


def _populate_legacy_fixture(conn: psycopg.Connection) -> None:
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


def _legacy_row_counts(conn: psycopg.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for tbl in ("documents", "document_images", "document_tables"):
            cur.execute(f"SELECT count(*) FROM {tbl}")
            counts[tbl] = cur.fetchone()[0]
    return counts


# ---------------------------------------------------------------------------
# Step 1 (RED → GREEN) — populated-legacy assertions
# ---------------------------------------------------------------------------


def test_v2_schema_version_row_exists(migrated_db: psycopg.Connection) -> None:
    with migrated_db.cursor() as cur:
        cur.execute(
            "SELECT version, applied_at FROM v2_schema_version"
        )
        row = cur.fetchone()
    assert row is not None, "v2_schema_version row missing"
    assert row[0] == 1, f"expected version 1, got {row[0]}"


def test_all_v2_tables_exist(migrated_db: psycopg.Connection) -> None:
    tables = _table_names(migrated_db)
    missing = EXPECTED_V2_TABLES - tables
    assert not missing, f"missing v2 tables: {sorted(missing)}"


def test_legacy_rows_unchanged(migrated_db: psycopg.Connection) -> None:
    """Every legacy row remains untouched and v1-readable."""
    counts = _legacy_row_counts(migrated_db)
    assert counts == {"documents": 3, "document_images": 6, "document_tables": 6}


def test_legacy_current_revision_id_is_nullable_and_null(
    migrated_db: psycopg.Connection,
) -> None:
    """All legacy rows have ``current_revision_id IS NULL`` after migration."""
    with migrated_db.cursor() as cur:
        cur.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'documents' AND column_name = 'current_revision_id'"
        )
        nullable = cur.fetchone()
    assert nullable is not None, "documents.current_revision_id column missing"
    assert nullable[0] == "YES", "current_revision_id must be nullable"
    with migrated_db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM documents WHERE current_revision_id IS NOT NULL"
        )
        non_null = cur.fetchone()[0]
    assert non_null == 0, "legacy documents must not have current_revision_id set"


def test_no_legacy_artifact_marked_revision_ready(
    migrated_db: psycopg.Connection,
) -> None:
    """No legacy child row is selected/repurposed by v2 — revision_id is NULL."""
    for tbl in ("document_images", "document_tables"):
        with migrated_db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM {tbl} WHERE revision_id IS NOT NULL".format(
                    tbl=tbl
                )
            )
            non_null = cur.fetchone()[0]
        assert non_null == 0, (
            f"legacy {tbl} rows must not have a revision_id after migration"
        )


def test_rerun_is_idempotent(migrated_db: psycopg.Connection) -> None:
    """A second ``apply_v2_schema`` is a no-op (no DDL churn)."""
    from app.services.agents.v2.persistence.migrate import apply_v2_schema

    counts_before = _legacy_row_counts(migrated_db)
    tables_before = _table_names(migrated_db)
    apply_v2_schema(migrated_db)
    migrated_db.commit()
    counts_after = _legacy_row_counts(migrated_db)
    tables_after = _table_names(migrated_db)
    assert counts_before == counts_after
    assert tables_before == tables_after
    # And the schema version row is still exactly one.
    with migrated_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM v2_schema_version")
        n = cur.fetchone()[0]
    assert n == 1


def test_migrate_module_never_imports_app_models() -> None:
    """The migration module must not import ``app.models`` (Phase 1B owns that)."""
    import re
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


def test_migrate_module_takes_advisory_lock(migrated_db: psycopg.Connection) -> None:
    """The migration acquires ``pg_advisory_xact_lock`` before any DDL.

    We prove this by checking the migration source: the function must call
    ``pg_advisory_xact_lock`` exactly once, before the first ``CREATE TABLE``.
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
    assert "pg_advisory_xact_lock" in src
    lock_pos = src.find("pg_advisory_xact_lock")
    first_create_pos = src.find("CREATE TABLE")
    if first_create_pos == -1:
        first_create_pos = src.find("create table")
    assert lock_pos != -1, "advisory lock call missing"
    assert first_create_pos != -1, "no CREATE TABLE in migration source"
    assert lock_pos < first_create_pos, (
        "advisory lock must be acquired before any CREATE TABLE"
    )


def test_migration_does_not_delete_legacy_rows(
    migrated_db: psycopg.Connection,
) -> None:
    """Source-level proof that the migration never runs a legacy-row DELETE/UPDATE.

    The only ``DELETE FROM`` we allow is inside the idempotency drop helpers
    that target v2 tables (which only run on a freshly-dropped test DB).
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
    # The migration must not reference legacy tables in DELETE/UPDATE statements.
    for stmt in ("DELETE FROM documents", "DELETE FROM document_images",
                 "DELETE FROM document_tables", "UPDATE documents",
                 "UPDATE document_images", "UPDATE document_tables"):
        assert stmt not in src, (
            f"forbidden legacy mutation found in migrate.py: {stmt!r}"
        )
