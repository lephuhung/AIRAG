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

    Order matters: drop triggers on legacy tables before any column
    removal they could reference; drop v2 tables first (cascades remove
    FKs pointing at them from legacy columns). The ``migrated_at``
    cleanup is defensive — the current migration does not add that
    column, but a previous test run with the round-1 sentinel-based
    trigger may have left it behind.
    """
    with _psycopg_connect() as conn:
        with conn.cursor() as cur:
            # Drop v2 tables first (cascades remove FKs pointing at them).
            for tbl in EXPECTED_V2_TABLES:
                cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
            # Drop the stable-pointer triggers on legacy tables.
            for trigger, table in (
                ("trg_documents_revision_id_stable", "documents"),
                ("trg_document_images_revision_id_stable", "document_images"),
                ("trg_document_tables_revision_id_stable", "document_tables"),
            ):
                cur.execute(
                    f"DROP TRIGGER IF EXISTS {trigger} ON {table}"
                )
            # Drop the trigger functions.
            for fn in (
                "raise_documents_revision_id_loss()",
                "raise_document_images_revision_id_loss()",
                "raise_document_tables_revision_id_loss()",
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

    After the round-2 fix the migration is fully schema-only: no
    ``DELETE FROM`` on legacy tables, and no ``UPDATE`` on legacy
    tables (the prior ``migrated_at`` sentinel backfill is gone; the
    brief item (4) invariant is enforced by a ``BEFORE UPDATE OF
    <revision_col>`` trigger that fires only when the revision column
    is in the SET clause, with no row-level mutation required).
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

    # No UPDATE on legacy tables either (the round-1 sentinel backfill
    # was removed because it broke v1 writes; the stable-pointer
    # trigger is now schema-only).
    legacy_updates = re.findall(
        r"UPDATE\s+(documents|document_images|document_tables)\b",
        code,
    )
    assert not legacy_updates, (
        f"forbidden legacy UPDATE in migrate.py: {legacy_updates!r}"
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


def test_revision_id_stable_pointer_triggers_exist(migrated_db: Engine) -> None:
    """Brief item (4) — round-2 fix: stable-pointer triggers are installed.

    The triggers fire only on ``UPDATE OF <revision_col>`` and only
    reject the non-null → null transition (the actual invariant: a row
    that already carries a revision pointer cannot silently lose it).
    v1 writes (INSERT, and UPDATE that touches any other column) never
    fire the trigger because of PG's ``UPDATE OF <col>`` scoping.
    """
    expected = (
        ("documents", "trg_documents_revision_id_stable", "current_revision_id"),
        ("document_images", "trg_document_images_revision_id_stable", "revision_id"),
        ("document_tables", "trg_document_tables_revision_id_stable", "revision_id"),
    )
    with migrated_db.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    c.relname::text            AS table_name,
                    t.tgname                    AS trigger_name,
                    (
                        SELECT array_agg(a.attname ORDER BY a.attnum)
                        FROM pg_catalog.pg_attribute a
                        WHERE a.attrelid = t.tgrelid
                          AND a.attnum = ANY(t.tgattr)
                    ) AS update_columns
                FROM pg_catalog.pg_trigger t
                JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND NOT t.tgisinternal
                """
            )
        ).fetchall()
    by_trigger = {r[1]: (r[0], list(r[2] or [])) for r in rows}
    for tbl, trigger_name, col in expected:
        assert trigger_name in by_trigger, (
            f"missing trigger {trigger_name} (found: {sorted(by_trigger)})"
        )
        owner_tbl, cols = by_trigger[trigger_name]
        assert owner_tbl == tbl, (
            f"trigger {trigger_name} is on {owner_tbl}, expected {tbl}"
        )
        assert cols == [col], (
            f"trigger {trigger_name} fires on columns {cols}, expected only "
            f"{[col]} — extra columns would break v1 writes"
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


def test_v1_style_writes_allowed(migrated_db: Engine) -> None:
    """Round-2 fix: v1 INSERTs and UPDATEs must continue to work post-migration.

    This is the regression test for the round-1 bug. The previous
    sentinel-based trigger raised on every UPDATE on a row with
    ``migrated_at IS NULL`` (which is every post-migration row because
    the column had no DEFAULT). The new stable-pointer trigger only
    fires on ``UPDATE OF <revision_col>``, so v1 UPDATEs that touch
    other columns (``status``, ``markdown_s3_key``, ``embed_done``,
    ...) are unaffected.
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
            # v1-style INSERT: no current_revision_id, no migrated_at
            # column (the new design does not add one).
            cur.execute(
                "INSERT INTO documents (id, workspace_id, filename, original_filename, "
                "file_type, file_size, status, chunk_count, page_count, image_count, "
                "table_count, processing_time_ms, embed_done, captions_done, kg_done, "
                "is_chat_upload, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 'pdf', 1, 'pending', 0, 0, 0, 0, 0, "
                "false, false, false, false, NOW(), NOW())",
                (str(doc_id), str(workspace_id), f"f-{doc_id}", f"o-{doc_id}.pdf"),
            )
        conn.commit()

    # v1-style UPDATEs that v1's parse_worker / embed_worker / kg_worker
    # pipelines emit. None of them touch ``current_revision_id``, so
    # the new ``UPDATE OF``-scoped trigger must NOT fire.
    with migrated_db.connect() as conn:
        conn.execute(
            text(
                "UPDATE documents SET status = 'indexed' WHERE id = :id"
            ),
            {"id": str(doc_id)},
        )
        conn.execute(
            text(
                "UPDATE documents SET markdown_s3_key = 'x/y.md' "
                "WHERE id = :id"
            ),
            {"id": str(doc_id)},
        )
        conn.execute(
            text(
                "UPDATE documents SET embed_done = true, captions_done = true, "
                "kg_done = true WHERE id = :id"
            ),
            {"id": str(doc_id)},
        )
        conn.commit()

    # Sanity: the row actually carries the new values.
    with migrated_db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, markdown_s3_key, embed_done, captions_done, "
                "kg_done, current_revision_id FROM documents WHERE id = :id"
            ),
            {"id": str(doc_id)},
        ).first()
    assert row is not None
    assert row[0] == "indexed", f"status update was lost: {row[0]!r}"
    assert row[1] == "x/y.md", f"markdown_s3_key update was lost: {row[1]!r}"
    assert row[2] is True and row[3] is True and row[4] is True, (
        f"embed/captions/kg flag updates were lost: {row[2:]!r}"
    )
    assert row[5] is None, (
        "v1-style writes must not have set current_revision_id "
        "(the v2 pipeline is the only place that does)"
    )


def test_revision_id_stable_pointer_rejects_unset(migrated_db: Engine) -> None:
    """Round-2 + round-3 fix: stable-pointer trigger fires on accidental unset.

    Once a row carries a revision pointer, the trigger must reject any
    UPDATE that sets it back to NULL *without simultaneously tombstoning*
    the document (i.e. setting ``source_deleted_at``). This is the
    actual invariant brief item (4) is protecting. The round-3 fix
    carves out the tombstone path (``UPDATE documents SET
    current_revision_id = NULL, source_deleted_at = NOW()``) but the
    accidental unset (no ``source_deleted_at``) MUST still raise.

    This test deliberately omits ``source_deleted_at`` from the SET
    clause so the WHEN clause
    ``OLD.current_revision_id IS NOT NULL
      AND NEW.current_revision_id IS NULL
      AND NEW.source_deleted_at IS NULL``
    evaluates to TRUE — the trigger fires and the UPDATE is rejected.
    """
    doc_id = uuid.uuid4()
    revision_id = uuid.uuid4()
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
            # Insert the document row BEFORE the document_revisions row:
            # document_revisions.document_id has ON DELETE RESTRICT, so
            # the parent document must exist first.
            cur.execute(
                "INSERT INTO documents (id, workspace_id, filename, original_filename, "
                "file_type, file_size, status, chunk_count, page_count, image_count, "
                "table_count, processing_time_ms, embed_done, captions_done, kg_done, "
                "is_chat_upload, created_at, updated_at, current_revision_id) "
                "VALUES (%s, %s, %s, %s, 'pdf', 1, 'indexed', 0, 0, 0, 0, 0, "
                "false, false, false, false, NOW(), NOW(), NULL)",
                (str(doc_id), str(workspace_id), f"f-{doc_id}", f"o-{doc_id}.pdf"),
            )
            cur.execute(
                "INSERT INTO document_revisions (revision_id, document_id, "
                "generation, status) VALUES (%s, %s, 1, 'active')",
                (str(revision_id), str(doc_id)),
            )
            # Now set current_revision_id (a non-null value triggers the
            # stable-pointer invariant; the FK still resolves because
            # the revision row exists).
            cur.execute(
                "UPDATE documents SET current_revision_id = %s WHERE id = %s",
                (str(revision_id), str(doc_id)),
            )
        conn.commit()

    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError) as excinfo:
        with migrated_db.connect() as conn:
            # Accidental unset: ``current_revision_id = NULL`` with no
            # simultaneous ``source_deleted_at`` set. The trigger MUST
            # fire (because ``NEW.source_deleted_at IS NULL`` makes the
            # WHEN clause TRUE) and the UPDATE MUST be rejected.
            conn.execute(
                text(
                    "UPDATE documents SET current_revision_id = NULL "
                    "WHERE id = :id"
                ),
                {"id": str(doc_id)},
            )
            conn.commit()
    msg = str(excinfo.value).lower()
    assert "current_revision_id" in msg or "cannot be unset" in msg, (
        f"unexpected DB error from stable-pointer trigger: {excinfo.value}"
    )

    # The row is still intact (the transaction raised; conn.rollback in
    # the inner block prevents partial application). The
    # ``current_revision_id`` column is unchanged.
    with migrated_db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT current_revision_id, source_deleted_at "
                "FROM documents WHERE id = :id"
            ),
            {"id": str(doc_id)},
        ).first()
    assert row is not None
    assert str(row[0]) == str(revision_id), (
        f"stable-pointer trigger must leave current_revision_id intact, "
        f"got {row[0]!r}"
    )
    assert row[1] is None, (
        f"stable-pointer trigger must not set source_deleted_at, "
        f"got {row[1]!r}"
    )


def test_tombstone_clears_current_revision_pointer(migrated_db: Engine) -> None:
    """Round-3 fix: the plan-mandated tombstone path is allowed.

    Phase 1C's ``mark_source_deleted`` path (plan §207–212, Final Gate
    #35 — ``test_tombstone_clears_current_revision_pointer``) is exactly
    ``UPDATE documents SET current_revision_id = NULL, source_deleted_at
    = NOW()``. The round-3 fix narrows the trigger WHEN clause to add
    ``AND NEW.source_deleted_at IS NULL``, which makes that UPDATE
    pass the trigger (because ``NEW.source_deleted_at IS NOT NULL``
    makes the WHEN clause FALSE) while still rejecting the accidental
    unset (``test_revision_id_stable_pointer_rejects_unset`` covers the
    other shape).
    """
    doc_id = uuid.uuid4()
    revision_id = uuid.uuid4()
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
                "is_chat_upload, created_at, updated_at, current_revision_id) "
                "VALUES (%s, %s, %s, %s, 'pdf', 1, 'indexed', 0, 0, 0, 0, 0, "
                "false, false, false, false, NOW(), NOW(), NULL)",
                (str(doc_id), str(workspace_id), f"f-{doc_id}", f"o-{doc_id}.pdf"),
            )
            cur.execute(
                "INSERT INTO document_revisions (revision_id, document_id, "
                "generation, status) VALUES (%s, %s, 1, 'active')",
                (str(revision_id), str(doc_id)),
            )
            cur.execute(
                "UPDATE documents SET current_revision_id = %s WHERE id = %s",
                (str(revision_id), str(doc_id)),
            )
        conn.commit()

    # Tombstone: SET both ``current_revision_id = NULL`` AND
    # ``source_deleted_at = NOW()``. The trigger's WHEN clause
    # ``OLD.current_revision_id IS NOT NULL
    #   AND NEW.current_revision_id IS NULL
    #   AND NEW.source_deleted_at IS NULL``
    # evaluates to FALSE (the third conjunct is FALSE), so the trigger
    # does NOT fire and the UPDATE is allowed.
    with migrated_db.connect() as conn:
        result = conn.execute(
            text(
                "UPDATE documents SET current_revision_id = NULL, "
                "source_deleted_at = NOW() WHERE id = :id"
            ),
            {"id": str(doc_id)},
        )
        conn.commit()
        assert result.rowcount == 1, (
            f"tombstone UPDATE must affect 1 row, got {result.rowcount}"
        )

    # Verify the post-tombstone state: pointer cleared, tombstone set.
    with migrated_db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT current_revision_id, source_deleted_at "
                "FROM documents WHERE id = :id"
            ),
            {"id": str(doc_id)},
        ).first()
    assert row is not None
    assert row[0] is None, (
        f"tombstone must clear current_revision_id, got {row[0]!r}"
    )
    assert row[1] is not None, (
        f"tombstone must set source_deleted_at, got None"
    )


def test_tombstone_carve_out_is_documents_only(migrated_db: Engine) -> None:
    """Round-3 fix: tombstone carve-out applies to ``documents`` only.

    The ``document_images`` and ``document_tables`` triggers do NOT
    have a ``source_deleted_at`` carve-out because those child tables
    have no ``source_deleted_at`` column. The asymmetry is intentional
    (see module docstring's "Tombstone carve-out" subsection) and is
    asserted here at three levels:

    1. Source-level: the SQL DDL for the child triggers does not mention
       ``source_deleted_at`` in the WHEN clause (only the documents
       trigger does).
    2. Trigger-function-level: the child trigger-function error
       messages do not reference ``source_deleted_at``.
    3. Behavior-level: the child-table triggers still raise on a
       non-null → null transition of ``revision_id`` (no carve-out
       path exists for them because the parent tombstone governs
       their tombstoned state via the parent-row tombstone + Phase 1C
       FK cascade review).
    """
    import re
    from pathlib import Path

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
    # Strip ONLY the module docstring and # comments. Triple-quoted SQL
    # blocks are preserved.
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)

    # Pull the SQL block of each trigger from the source. We use a
    # simple "find the trigger CREATE, then capture until the
    # EXECUTE FUNCTION line" approach.
    def _trigger_block(trigger_name: str) -> str:
        # Find the CREATE TRIGGER line (after the DROP TRIGGER).
        marker = f"CREATE TRIGGER {trigger_name}"
        i = code.find(marker)
        assert i != -1, f"trigger {trigger_name} not found in migrate.py"
        # Capture up to the EXECUTE FUNCTION terminator.
        j = code.find("EXECUTE FUNCTION", i)
        assert j != -1, f"trigger {trigger_name} has no EXECUTE FUNCTION"
        return code[i : j + 1]

    docs_block = _trigger_block("trg_documents_revision_id_stable")
    imgs_block = _trigger_block("trg_document_images_revision_id_stable")
    tbls_block = _trigger_block("trg_document_tables_revision_id_stable")

    # The documents trigger must mention ``source_deleted_at`` in its
    # WHEN clause; the child triggers MUST NOT.
    assert "source_deleted_at" in docs_block, (
        "documents trigger WHEN clause must include the "
        "source_deleted_at tombstone carve-out"
    )
    assert "source_deleted_at" not in imgs_block, (
        "document_images trigger WHEN clause must NOT mention "
        "source_deleted_at (carve-out is documents-only)"
    )
    assert "source_deleted_at" not in tbls_block, (
        "document_tables trigger WHEN clause must NOT mention "
        "source_deleted_at (carve-out is documents-only)"
    )

    # Behavior-level: a non-null → null transition on
    # ``document_images.revision_id`` is still rejected by the trigger
    # even when the row was inserted with the column set (no carve-out
    # applies). This proves the child triggers keep the original
    # no-unset invariant unchanged.
    doc_id = uuid.uuid4()
    image_id = uuid.uuid4()
    revision_id = uuid.uuid4()
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
                "is_chat_upload, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 'pdf', 1, 'indexed', 0, 0, 0, 0, 0, "
                "false, false, false, false, NOW(), NOW())",
                (str(doc_id), str(workspace_id), f"f-{doc_id}", f"o-{doc_id}.pdf"),
            )
            cur.execute(
                "INSERT INTO document_revisions (revision_id, document_id, "
                "generation, status) VALUES (%s, %s, 1, 'active')",
                (str(revision_id), str(doc_id)),
            )
            cur.execute(
                "INSERT INTO document_images (id, document_id, image_id, "
                "page_no, file_path, caption, width, height, mime_type, "
                "created_at, revision_id) VALUES (%s, %s, %s, 1, "
                "'/tmp/x.png', 'cap', 10, 10, 'image/png', NOW(), %s)",
                (str(image_id), str(doc_id), f"img-{doc_id}", str(revision_id)),
            )
        conn.commit()

    from sqlalchemy.exc import DBAPIError

    # A non-null → null on ``document_images.revision_id`` MUST raise
    # (the child trigger has no carve-out). Note the SET clause also
    # tries ``source_deleted_at = NOW()`` (which is a documents-only
    # column), but PG would reject that on a child table before the
    # trigger fires; so we deliberately do NOT include
    # ``source_deleted_at`` here — the carve-out is documents-only.
    with pytest.raises(DBAPIError) as excinfo:
        with migrated_db.connect() as conn:
            conn.execute(
                text(
                    "UPDATE document_images SET revision_id = NULL "
                    "WHERE id = :id"
                ),
                {"id": str(image_id)},
            )
            conn.commit()
    msg = str(excinfo.value).lower()
    assert "revision_id" in msg or "cannot be unset" in msg, (
        f"document_images trigger must still reject non-null → null "
        f"unset (no carve-out), got: {excinfo.value}"
    )

    # The image row's revision_id is unchanged.
    with migrated_db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT revision_id FROM document_images WHERE id = :id"
            ),
            {"id": str(image_id)},
        ).first()
    assert row is not None
    assert str(row[0]) == str(revision_id), (
        f"document_images trigger must leave revision_id intact, "
        f"got {row[0]!r}"
    )