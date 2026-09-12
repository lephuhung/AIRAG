"""Test fixtures for Phase 1C Task 3 — revision-owned build/state/publication.

The v2 schema migration (``app.services.agents.v2.persistence.migrate``) is
the **only** code path that creates v2 tables. This conftest bootstraps a
**fresh** test database (``hrag_test_v2_task3``) with:

1. The minimal legacy v1 schema (just enough columns to satisfy the
   v2 migration's ALTER TABLE / FK constraints + the stable-pointer
   trigger).
2. The v2 schema (12 tables + triggers + version row), applied via
   ``apply_v2_schema``.

The test DSN is overridable through ``V2_TASK3_DATABASE_URL``; the
default targets the controller-prepared ``hrag_test_v2_task3``
database. The test author never has to manage the database lifecycle
during a test session — every test that uses the ``db`` fixture gets
a SAVEPOINT-wrapped async session that auto-rolls back on teardown.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import AsyncIterator

import psycopg
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.engine import Engine

from app.services.agents.v2.persistence.migrate import apply_v2_schema, make_engine

DEFAULT_TASK3_DSN = "postgresql://postgres:postgres@localhost:5433/hrag_test_v2_task3"
DEFAULT_TASK3_ASYNC_DSN = (
    "postgresql+asyncpg://postgres:postgres@localhost:5433/hrag_test_v2_task3"
)


def _task3_dsn() -> str:
    return os.environ.get("V2_TASK3_DATABASE_URL", DEFAULT_TASK3_DSN)


def _task3_async_dsn_sync() -> tuple[str, str]:
    """Return (sync_dsn, async_dsn)."""
    sync_dsn = _task3_dsn()
    async_dsn = sync_dsn.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    return sync_dsn, async_dsn


# ---------------------------------------------------------------------------
# Legacy schema bootstrap (minimal — just enough for the v2 migration's
# ALTER TABLE / FK constraints to succeed).
# ---------------------------------------------------------------------------


_LEGACY_DDL: tuple[str, ...] = (
    # documentstatus enum (the v2 migration's UPDATE/INSERT may reference it)
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'documentstatus') THEN
            CREATE TYPE documentstatus AS ENUM (
                'pending', 'parsing', 'ocring', 'chunking', 'embedding',
                'building_kg', 'indexed', 'failed'
            );
        END IF;
    END $$;
    """,
    # tenants
    """
    CREATE TABLE IF NOT EXISTS tenants (
        id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        name         VARCHAR(255) NOT NULL,
        slug         VARCHAR(255),
        is_active    BOOLEAN      NOT NULL DEFAULT true,
        created_at   TIMESTAMP    NOT NULL DEFAULT NOW(),
        updated_at   TIMESTAMP    NOT NULL DEFAULT NOW()
    )
    """,
    # users
    """
    CREATE TABLE IF NOT EXISTS users (
        id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        email           VARCHAR(255) NOT NULL,
        password_hash   VARCHAR(255),
        full_name       VARCHAR(255),
        is_active       BOOLEAN      NOT NULL DEFAULT true,
        is_superadmin   BOOLEAN      NOT NULL DEFAULT false,
        settings        JSONB        NOT NULL DEFAULT '{}'::jsonb,
        totp_enabled    BOOLEAN      NOT NULL DEFAULT false,
        totp_secret     VARCHAR(64),
        avatar_url      VARCHAR(1024),
        created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
    )
    """,
    # knowledge_bases (parent of documents)
    """
    CREATE TABLE IF NOT EXISTS knowledge_bases (
        id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        name            VARCHAR(255) NOT NULL,
        description     TEXT,
        system_prompt   TEXT,
        visibility      VARCHAR(20)  NOT NULL DEFAULT 'public',
        owner_id        UUID         REFERENCES users(id),
        tenant_id       UUID         REFERENCES tenants(id),
        is_default      BOOLEAN      NOT NULL DEFAULT false,
        created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
    )
    """,
    # documents — full schema so v1-style code paths can write to it
    """
    CREATE TABLE IF NOT EXISTS documents (
        id                          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id                UUID         NOT NULL REFERENCES knowledge_bases(id),
        filename                    VARCHAR(255),
        original_filename           VARCHAR(255),
        file_type                   VARCHAR(50),
        file_size                   INTEGER,
        status                      documentstatus DEFAULT 'pending',
        chunk_count                 INTEGER      NOT NULL DEFAULT 0,
        error_message               VARCHAR(500),
        created_at                  TIMESTAMP    NOT NULL DEFAULT NOW(),
        updated_at                  TIMESTAMP    NOT NULL DEFAULT NOW(),
        markdown_s3_key             VARCHAR(500),
        upload_s3_key               VARCHAR(500),
        page_count                  INTEGER      NOT NULL DEFAULT 0,
        image_count                 INTEGER      NOT NULL DEFAULT 0,
        table_count                 INTEGER      NOT NULL DEFAULT 0,
        parser_version              VARCHAR(50),
        processing_time_ms          INTEGER      NOT NULL DEFAULT 0,
        embed_done                  BOOLEAN      NOT NULL DEFAULT false,
        captions_done               BOOLEAN      NOT NULL DEFAULT false,
        kg_done                     BOOLEAN      NOT NULL DEFAULT false,
        raw_chunks_json             TEXT,
        digital_signatures          JSON,
        document_type_id            UUID,
        document_number             VARCHAR(100),
        document_title              VARCHAR(500),
        location                    VARCHAR(255),
        issuing_agency              VARCHAR(255),
        parent_agency               VARCHAR(255),
        published_date              VARCHAR(100),
        signer_name                 VARCHAR(255),
        kg_root_entity_id           VARCHAR(500),
        uploaded_by                 UUID         REFERENCES users(id),
        is_chat_upload              BOOLEAN      NOT NULL DEFAULT false,
        content_hash                VARCHAR(64),
        validity_status             VARCHAR(30)  DEFAULT 'unknown',
        superseded_by_number        VARCHAR(100),
        superseded_by_document_id   UUID         REFERENCES documents(id),
        effective_date              VARCHAR(100),
        validity_events             JSON
    )
    """,
    # document_images
    """
    CREATE TABLE IF NOT EXISTS document_images (
        id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        image_id    VARCHAR(100),
        page_no     INTEGER      NOT NULL DEFAULT 0,
        file_path   VARCHAR(500),
        caption     TEXT,
        width       INTEGER      NOT NULL DEFAULT 0,
        height      INTEGER      NOT NULL DEFAULT 0,
        mime_type   VARCHAR(50)  NOT NULL DEFAULT 'image/png',
        created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
    )
    """,
    # document_tables
    """
    CREATE TABLE IF NOT EXISTS document_tables (
        id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id       UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        table_id          VARCHAR(100),
        page_no           INTEGER      NOT NULL DEFAULT 0,
        content_markdown  TEXT,
        caption           TEXT,
        num_rows          INTEGER      NOT NULL DEFAULT 0,
        num_cols          INTEGER      NOT NULL DEFAULT 0,
        created_at        TIMESTAMP    NOT NULL DEFAULT NOW()
    )
    """,
)


def _reset_legacy_schema(conn) -> None:
    """Tear down legacy + v2 state, rebuild minimal legacy, reapply v2 migration.

    The DDL that the v2 migration adds (current_revision_id /
    source_deleted_at columns on documents + the document_revisions →
    documents FK) is owned by ``apply_v2_schema`` so we drop v2 tables
    first (which cascades the FKs) before re-running the migration.

    Every DROP uses ``IF EXISTS`` so the reset is safe on a brand-new
    database where neither legacy nor v2 objects exist yet — this is
    the typical first-run case.

    The reset uses a SAVEPOINT per DDL so a failure (e.g., dropping a
    trigger whose table does not exist on a brand-new DB) does not
    abort the outer transaction. After the savepoint rolls back, the
    outer transaction is still usable.
    """
    def _sp_exec(stmt: str) -> None:
        """Execute inside a SAVEPOINT so a failure can be rolled back
        without aborting the outer transaction."""
        try:
            with conn.begin_nested() as sp:
                conn.execute(text(stmt))
        except Exception as exc:  # object does not exist / already dropped
            # Surface the failure instead of hiding a broken reset.
            print(f"warn: savepoint DDL failed: {exc}", file=sys.stderr)

    # Drop v2 tables (cascades remove FKs pointing at them from legacy tables).
    for tbl in (
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
    ):
        _sp_exec(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
    # Drop triggers + functions the migration installs.
    for trigger, table in (
        ("trg_documents_revision_id_stable", "documents"),
        ("trg_document_images_revision_id_stable", "document_images"),
        ("trg_document_tables_revision_id_stable", "document_tables"),
    ):
        _sp_exec(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
    for fn in (
        "raise_documents_revision_id_loss()",
        "raise_document_images_revision_id_loss()",
        "raise_document_tables_revision_id_loss()",
    ):
        _sp_exec(f"DROP FUNCTION IF EXISTS {fn}")
    # Drop v2 columns that the migration may have added to legacy tables.
    for stmt in (
        "ALTER TABLE documents DROP COLUMN IF EXISTS current_revision_id",
        "ALTER TABLE documents DROP COLUMN IF EXISTS source_deleted_at",
        "ALTER TABLE document_images DROP COLUMN IF EXISTS revision_id",
        "ALTER TABLE document_tables DROP COLUMN IF EXISTS revision_id",
    ):
        _sp_exec(stmt)
    # Drop legacy child tables and re-create them pristine (also clears
    # any dropped-column slot accumulation from prior runs).
    for stmt in (
        "DROP TABLE IF EXISTS document_tables CASCADE",
        "DROP TABLE IF EXISTS document_images CASCADE",
        "DROP TABLE IF EXISTS documents CASCADE",
        "DROP TABLE IF EXISTS knowledge_bases CASCADE",
        "DROP TABLE IF EXISTS users CASCADE",
        "DROP TABLE IF EXISTS tenants CASCADE",
        "DROP TYPE IF EXISTS documentstatus CASCADE",
    ):
        _sp_exec(stmt)


# ---------------------------------------------------------------------------
# Session-scoped fixture: bootstrap the v2 schema once per test process.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def v2_sync_engine() -> Engine:
    """A sync SQLAlchemy engine for v2 schema setup/teardown.

    Tests themselves use the ``async_db`` fixture (async session); this
    engine is reserved for DDL (bootstrap, reset between modules).
    """
    sync_dsn, _ = _task3_async_dsn_sync()
    engine = make_engine(sync_dsn)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_v2_schema(v2_sync_engine: Engine) -> None:
    """One-shot bootstrap: reset legacy + apply v2 migration.

    Runs once per pytest session (autouse). Subsequent tests use the
    ``async_db`` fixture and rely on SAVEPOINT isolation — no per-test
    DDL.
    """
    with v2_sync_engine.begin() as conn:
        _reset_legacy_schema(conn)
        for ddl in _LEGACY_DDL:
            conn.execute(text(ddl))
    # Run the migration in its own transaction (it owns its own begin()).
    apply_v2_schema(v2_sync_engine)


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_engine():
    """Per-test async engine (fresh pool, disposed on teardown)."""
    _, async_dsn = _task3_async_dsn_sync()
    engine = create_async_engine(async_dsn, future=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def async_db(async_engine, document_factory) -> AsyncIterator[AsyncSession]:
    """An AsyncSession wrapped in an outer transaction + SAVEPOINT.

    Every test starts with a clean slate (no rows from other tests),
    regardless of whether the test commits or rolls back. The
    SAVEPOINT pattern isolates each test from the bootstrap fixtures.

    ``document_factory`` is an explicit dependency **only to force fixture
    teardown ordering**: pytest finalizes fixtures LIFO by setup order, so
    depending on ``document_factory`` guarantees it is set up *before* this
    session. Teardown then rolls this session back *before*
    ``document_factory`` runs its raw autocommit ``DELETE FROM documents`` —
    otherwise that DELETE blocks forever on the row lock held by this still-open
    uncommitted transaction (an intra-process Postgres lock deadlock; the whole
    test suite hangs at the first test's teardown).
    """
    maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False,
        autocommit=False, autoflush=False,
    )
    async with maker() as session:
        await session.begin()
        # Outer transaction is now open; open a SAVEPOINT for the test.
        nested = await session.begin_nested()
        try:
            yield session
        finally:
            try:
                await nested.rollback()
            except Exception:
                pass
            try:
                await session.rollback()
            except Exception:
                pass


@pytest_asyncio.fixture
async def raw_connection():
    """A raw psycopg connection for setup helpers that need DDL/DML
    outside the SAVEPOINT (e.g., seeding a tenant/workspace/document
    row that the test's repository code never touches)."""
    sync_dsn, _ = _task3_async_dsn_sync()
    conn = psycopg.connect(sync_dsn, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Domain-row seed helpers — operate on a raw psycopg connection (autocommit)
# so the rows are visible to async transactions but isolated from the
# test's SAVEPOINT. Each helper cleans its own slate.
# ---------------------------------------------------------------------------


@pytest.fixture
def document_factory(raw_connection):
    """Return a callable that inserts a fresh legacy ``documents`` row
    (with parent tenants/users/knowledge_bases) outside the test's
    SAVEPOINT.

    Usage::

        doc_id = document_factory()  # one document + tenant + workspace + user
        doc_id = document_factory(is_chat_upload=True)
    """
    created: list[str] = []

    def _factory(
        *,
        is_chat_upload: bool = False,
        content_hash: str | None = None,
        workspace_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        if workspace_id is None:
            workspace_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        with raw_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (id, name, slug) "
                "VALUES (%s, %s, %s)",
                (
                    str(tenant_id),
                    f"tenant-{tenant_id}",
                    f"slug-{tenant_id}",
                ),
            )
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name) "
                "VALUES (%s, %s, %s, %s)",
                (
                    str(user_id),
                    f"u-{user_id}@example.com",
                    "x",
                    "Test User",
                ),
            )
            cur.execute(
                "INSERT INTO knowledge_bases (id, name, owner_id, tenant_id) "
                "VALUES (%s, %s, %s, %s)",
                (
                    str(workspace_id),
                    f"kb-{workspace_id}",
                    str(user_id),
                    str(tenant_id),
                ),
            )
            cur.execute(
                "INSERT INTO documents ("
                "id, workspace_id, filename, original_filename, file_type, "
                "file_size, content_hash, is_chat_upload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    str(doc_id),
                    str(workspace_id),
                    f"f-{doc_id}",
                    f"orig-{doc_id}.pdf",
                    "pdf",
                    1024,
                    content_hash,
                    is_chat_upload,
                ),
            )
        created.append(str(doc_id))
        return doc_id

    yield _factory

    # cleanup
    with raw_connection.cursor() as cur:
        for did in created:
            # Clear the current pointer first: ``fk_documents_current_revision``
            # blocks deleting a revision row a document still points at. Both
            # columns must move in one statement — the stable-pointer trigger's
            # carve-out only fires when ``source_deleted_at`` becomes non-null.
            cur.execute(
                "UPDATE documents SET current_revision_id = NULL, "
                "source_deleted_at = NOW() WHERE id = %s",
                (did,),
            )
            cur.execute(
                "DELETE FROM revision_ingestion_attempts WHERE document_id = %s",
                (did,),
            )
            # Build rows FK-RESTRICT their revision; delete them first.
            cur.execute(
                "DELETE FROM document_revision_builds WHERE revision_id IN "
                "(SELECT revision_id FROM document_revisions WHERE document_id = %s)",
                (did,),
            )
            cur.execute(
                "DELETE FROM document_revisions WHERE document_id = %s",
                (did,),
            )
            cur.execute("DELETE FROM documents WHERE id = %s", (did,))
