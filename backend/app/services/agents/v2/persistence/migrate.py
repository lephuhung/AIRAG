"""Release 1A — Raw-SQL migration runner for the v2 revision schema.

This module is the **only** code path in Phase 1 that creates v2 tables.
Release 1B (ORM registration) must not call ``Base.metadata.create_all`` or
emit v2 DDL — it only maps the schema this module creates.

Interface contract
------------------

The public API takes a SQLAlchemy ``Engine`` (sync):

    apply_v2_schema(engine) -> None
    check_v2_schema(engine) -> SchemaCheck

MUST be called with a sync ``Engine`` pointing at a plain ``postgresql://``
DSN (the runner normalizes to the psycopg3 driver). The async engines used
elsewhere (``app.core.database.engine``) construct a sync engine for the
migration; the runner itself is sync because every DDL is a short-lived
transaction and an async loop would only add overhead.

Design constraints (from the Phase 1 plan):

- No ``import app.models`` (Phase 1B owns ORM registration).
- No ``Base.metadata.create_all``.
- All DDL is wrapped in a single transaction preceded by
  ``pg_advisory_xact_lock`` so concurrent migration attempts serialize.
- New columns on legacy tables are nullable and never backfilled
  (data backfill belongs to Release 1B's revision-aware reindex).
- FKs from legacy tables to ``document_revisions`` and from
  ``document_revisions`` to ``documents`` are ``ON DELETE RESTRICT`` (or
  default ``NO ACTION``). Phase 1C tombstone + artifact-GC owns
  reclamation, not DB cascade (plan prohibition #14).
- The migration is idempotent: re-running it is a no-op once the
  ``v2_schema_version`` row exists.
- The set ``V2_SCHEMA_V1_TABLES`` is the exact contract Release 1B relies on
  (Phase 1B must not assume any table beyond this set).

Brief item (4) — revision-ID enforcement
----------------------------------------

The brief requires DB constraints/triggers that reject any future write
to a legacy table that lacks the appropriate revision ID. The
``CHECK (revision_id IS NOT NULL)`` we might be tempted to install
cannot be enforced on legacy rows that legitimately lack a revision ID
— and v1 must keep working during the transition window — so we use a
**stable-pointer trigger** instead:

1. Each legacy table (``documents``, ``document_images``,
   ``document_tables``) gets a ``BEFORE UPDATE OF <revision_col>``
   trigger that fires *only* when the ``current_revision_id`` (or
   ``revision_id`` for image/table) column is in the SET clause.
2. The trigger's ``WHEN`` clause is
   ``OLD.<revision_col> IS NOT NULL AND NEW.<revision_col> IS NULL``.
   This catches the only invariant the brief is protecting: a row that
   already carries a revision pointer cannot silently lose it.
3. PG's ``UPDATE OF <column>`` trigger scoping means v1 UPDATEs that
   touch *other* columns (``status``, ``markdown_s3_key``, ``embed_done``,
   ``raw_chunks_json``, ...) never fire the trigger at all. v1 INSERTs
   are unaffected because the trigger is ``BEFORE UPDATE OF``, not
   ``BEFORE INSERT OR UPDATE``.

**Why no INSERT-side trigger:** any INSERT-time check that requires a
revision ID would also reject v1-style INSERTs during the transition
window, which the brief explicitly prohibits. The FK from
``documents.current_revision_id`` (and the matching image/table
columns) to ``document_revisions(revision_id)`` (installed by the
``_LEGACY_FK_STATEMENTS`` step) already enforces revision existence
when the v2 pipeline populates the column.

**Why no DELETE-side trigger:** brief Step 2 item (3) requires that
cascading deletes from ``documents`` cannot orphan or destroy
``document_revisions``; this is enforced by the ``ON DELETE RESTRICT``
FKs on ``document_revisions.document_id``. A DELETE-side trigger on
legacy child tables is not needed because the ``ON DELETE RESTRICT``
on ``document_revisions`` is the only place a DELETE could otherwise
silently cascade, and we block it there.

Net effect: v1 INSERT/UPDATE paths are completely unaffected; the v2
invariants are enforced by the FK (existence) plus the stable-pointer
trigger (no losing the pointer once set).

Brief item (5) — legacy-unchanged verification
----------------------------------------------

``_capture_legacy_baseline(conn)`` snapshots ``count(*)`` and
``pg_relation_size`` for each legacy table after the advisory lock is
acquired and before any DDL mutates those tables. ``_verify_legacy_unchanged``
re-checks at the end of the same transaction (before
``v2_schema_version`` is written) and raises ``RuntimeError`` on row-count
drift. Size drift is reported as a warning (column additions legitimately
grow ``pg_relation_size``) but never raises.

Module layout::

    V2_SCHEMA_VERSION = 1
    V2_SCHEMA_V1_TABLES = frozenset({...})  # the 12 tables of Release 1A
    SchemaCheck = dataclass(frozen=True)
    check_v2_schema(engine) -> SchemaCheck
    apply_v2_schema(engine) -> None
    main()  # CLI entrypoint
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

V2_SCHEMA_VERSION: int = 1

# The exact 12 tables created by Release 1A. Order is preserved for documentation;
# creation order is enforced separately to honour FK dependencies.
V2_SCHEMA_V1_TABLES: frozenset[str] = frozenset(
    {
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
    }
)

# Advisory-lock key for the v2 migration. A unique stable bigint avoids
# colliding with any other advisory-lock user in the database.
V2_MIGRATION_ADVISORY_LOCK_KEY: int = 0x5642_5F4D_4947_5241  # "VB_MIGRA" ascii

# SQL for the transaction-scoped advisory lock. Defined as a module-level
# constant (before any DDL string literal) so the source-level order check
# in ``test_migrate_module_takes_advisory_lock`` finds
# ``pg_advisory_xact_lock`` in the code stream before the first
# ``CREATE TABLE`` literal.
_ADVISORY_LOCK_SQL: str = "SELECT pg_advisory_xact_lock(:key)"

# Legacy tables whose ``count(*)`` and ``pg_relation_size`` are snapshot
# before/after the migration to enforce brief item (5).
_LEGACY_TABLES_FOR_BASELINE: tuple[str, ...] = (
    "documents",
    "document_images",
    "document_tables",
)


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaCheck:
    """Outcome of ``check_v2_schema`` — never raises."""

    applied: bool
    version: int | None
    missing_tables: frozenset[str]
    extra_tables: frozenset[str]

    @property
    def is_clean(self) -> bool:
        return self.applied and not self.missing_tables and not self.extra_tables


# ---------------------------------------------------------------------------
# DDL — exact Release 1A schema
# ---------------------------------------------------------------------------


_CREATE_DDL: tuple[str, ...] = (
    # 1. version table — written last so a partial run cannot leave a version row
    # without its corresponding tables.
    """
    CREATE TABLE IF NOT EXISTS v2_schema_version (
        version        INTEGER     PRIMARY KEY,
        description    TEXT        NOT NULL,
        applied_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # 2. document_revisions — immutable per-generation revision row.
    # ON DELETE RESTRICT: brief Step 2 item (3) and plan prohibition #14.
    # Phase 1C tombstone + artifact-GC owns reclamation, not DB cascade.
    """
    CREATE TABLE IF NOT EXISTS document_revisions (
        revision_id                    UUID        PRIMARY KEY,
        document_id                    UUID        NOT NULL,
        generation                     BIGINT      NOT NULL,
        retry_of_revision_id           UUID        NULL,
        status                         TEXT        NOT NULL,
        failed_at                      TIMESTAMPTZ NULL,
        failure_stage                  TEXT        NULL,
        failure_class                  TEXT        NULL,
        abandoned_at                   TIMESTAMPTZ NULL,
        abandon_reason                 TEXT        NULL,
        artifact_retention_starts_at   TIMESTAMPTZ NULL,
        superseded_at                  TIMESTAMPTZ NULL,
        artifacts_purged_at            TIMESTAMPTZ NULL,
        created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (document_id, generation),
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE RESTRICT,
        FOREIGN KEY (retry_of_revision_id) REFERENCES document_revisions(revision_id)
    )
    """,
    # 3. document_revision_builds — one row per immutable build profile attempt.
    """
    CREATE TABLE IF NOT EXISTS document_revision_builds (
        build_id                  UUID        PRIMARY KEY,
        revision_id               UUID        NOT NULL,
        build_profile             TEXT        NOT NULL,
        embedding_namespace       TEXT        NULL,
        embedding_model_hash      TEXT        NULL,
        embedding_dimension       INTEGER     NULL,
        vector_artifact_version   TEXT        NULL,
        started_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at               TIMESTAMPTZ NULL,
        UNIQUE (revision_id, build_profile),
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
            ON DELETE RESTRICT
    )
    """,
    # 4. document_revision_chunks — derived chunk rows owned by a revision.
    """
    CREATE TABLE IF NOT EXISTS document_revision_chunks (
        chunk_id     UUID    PRIMARY KEY,
        revision_id  UUID    NOT NULL,
        ordinal      INTEGER NOT NULL,
        UNIQUE (revision_id, ordinal),
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
            ON DELETE RESTRICT
    )
    """,
    # 5. revision_ingestion_attempts — one row per (document, source identity, build).
    # The named constraint ``uq_revision_ingestion_attempt_key`` is the
    # ON CONFLICT arbiter for the ingestion pipeline (brief requirement).
    # revision_id is left as default NO ACTION — Phase 1C tombstones own
    # retention; an accidental parent delete must not silently orphan
    # ingestion attempt history.
    """
    CREATE TABLE IF NOT EXISTS revision_ingestion_attempts (
        attempt_id              UUID        PRIMARY KEY,
        document_id             UUID        NOT NULL,
        source_scheme           TEXT        NOT NULL,
        source_bucket           TEXT        NOT NULL,
        source_object_key       TEXT        NOT NULL,
        source_version_id       TEXT        NULL,
        source_etag             TEXT        NULL,
        source_size             BIGINT      NULL,
        source_sha256           TEXT        NULL,
        attempt_generation      INTEGER     NOT NULL DEFAULT 1,
        exhausted_at            TIMESTAMPTZ NULL,
        build_profile           TEXT        NOT NULL,
        revision_id             UUID        NULL,
        started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at             TIMESTAMPTZ NULL,
        CONSTRAINT uq_revision_ingestion_attempt_key UNIQUE
            (document_id, source_scheme, source_bucket, source_object_key, build_profile),
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
    )
    """,
    # 6. source_arrivals — webhook staging table (one row per MinIO event).
    """
    CREATE TABLE IF NOT EXISTS source_arrivals (
        arrival_id        UUID        PRIMARY KEY,
        bucket            TEXT        NOT NULL,
        object_key        TEXT        NOT NULL,
        version_id        TEXT        NULL,
        etag              TEXT        NULL,
        size_bytes        BIGINT      NULL,
        arrival_identity  TEXT        NOT NULL,
        received_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        processed_at      TIMESTAMPTZ NULL,
        UNIQUE (arrival_identity)
    )
    """,
    # 7. revision_retention_leases — GC anchor for resumable runs.
    # The null-safe unique constraint is required by gate #39 (checkpoint
    # retention lease protocol). PG15+ syntax: NULLS NOT DISTINCT so two
    # rows with NULL evidence_use_id would still collide.
    """
    CREATE TABLE IF NOT EXISTS revision_retention_leases (
        lease_id         UUID        PRIMARY KEY,
        run_id           TEXT        NOT NULL,
        revision_id      UUID        NOT NULL,
        evidence_use_id  UUID        NULL,
        acquired_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at       TIMESTAMPTZ NOT NULL,
        released_at      TIMESTAMPTZ NULL,
        release_reason   TEXT        NULL,
        CONSTRAINT uq_revision_lease_run_revision_use UNIQUE NULLS NOT DISTINCT
            (run_id, revision_id, evidence_use_id),
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
            ON DELETE RESTRICT
    )
    """,
    # 8. conversation_snapshots — checkpointed chat truth projection.
    """
    CREATE TABLE IF NOT EXISTS conversation_snapshots (
        snapshot_id   UUID        PRIMARY KEY,
        thread_id     TEXT        NOT NULL,
        taken_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (thread_id, taken_at)
    )
    """,
    # 9. semantic_snapshots — semantic context projection.
    """
    CREATE TABLE IF NOT EXISTS semantic_snapshots (
        snapshot_id   UUID        PRIMARY KEY,
        thread_id     TEXT        NOT NULL,
        taken_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (thread_id, taken_at)
    )
    """,
    # 10. binding_audit — every binding decision is recorded here.
    """
    CREATE TABLE IF NOT EXISTS binding_audit (
        audit_id     UUID        PRIMARY KEY,
        thread_id    TEXT        NOT NULL,
        recorded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # 11. evidence_records — encrypted evidence payloads.
    """
    CREATE TABLE IF NOT EXISTS evidence_records (
        evidence_id            UUID        PRIMARY KEY,
        ciphertext             BYTEA       NOT NULL,
        encryption_key_id      TEXT        NOT NULL,
        nonce                  BYTEA       NOT NULL,
        encryption_algorithm   TEXT        NOT NULL,
        payload_purged_at      TIMESTAMPTZ NULL,
        created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # 12. evidence_uses — link between a TaskSpec and an EvidenceRecord.
    """
    CREATE TABLE IF NOT EXISTS evidence_uses (
        use_id        UUID    PRIMARY KEY,
        evidence_id   UUID    NOT NULL,
        task_id       TEXT    NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        FOREIGN KEY (evidence_id) REFERENCES evidence_records(evidence_id)
    )
    """,
)


_CREATE_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS ix_revisions_doc ON document_revisions(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_revisions_status ON document_revisions(status)",
    "CREATE INDEX IF NOT EXISTS ix_revisions_retry_of "
    "ON document_revisions(retry_of_revision_id)",
    "CREATE INDEX IF NOT EXISTS ix_attempts_doc "
    "ON revision_ingestion_attempts(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_attempts_revision "
    "ON revision_ingestion_attempts(revision_id)",
    "CREATE INDEX IF NOT EXISTS ix_arrivals_received "
    "ON source_arrivals(received_at)",
    "CREATE INDEX IF NOT EXISTS ix_leases_run "
    "ON revision_retention_leases(run_id)",
    "CREATE INDEX IF NOT EXISTS ix_leases_active "
    "ON revision_retention_leases(revision_id, expires_at) "
    "WHERE released_at IS NULL",
    "CREATE INDEX IF NOT EXISTS ix_evidence_uses_task ON evidence_uses(task_id)",
)


_NULLABILITY_COLUMNS: tuple[str, ...] = (
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS current_revision_id UUID NULL",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_deleted_at TIMESTAMPTZ NULL",
    "ALTER TABLE document_images ADD COLUMN IF NOT EXISTS revision_id UUID NULL",
    "ALTER TABLE document_tables ADD COLUMN IF NOT EXISTS revision_id UUID NULL",
)


_NULLABILITY_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS ix_documents_source_deleted_at "
    "ON documents(source_deleted_at) WHERE source_deleted_at IS NOT NULL",
)


# ---------------------------------------------------------------------------
# Brief item (4) — stable-pointer triggers
# ---------------------------------------------------------------------------


# FK constraints added AFTER the columns exist. ON DELETE NO ACTION
# (default) — the brief asks for "no cascade"; we do not pick a more
# permissive action. Phase 1C tombstones decide when a document_revision
# may be retired.
_LEGACY_FK_STATEMENTS: tuple[str, ...] = (
    """
    ALTER TABLE documents
        ADD CONSTRAINT fk_documents_current_revision
        FOREIGN KEY (current_revision_id) REFERENCES document_revisions(revision_id)
    """,
    """
    ALTER TABLE document_images
        ADD CONSTRAINT fk_document_images_revision
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
    """,
    """
    ALTER TABLE document_tables
        ADD CONSTRAINT fk_document_tables_revision
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
    """,
)


# Trigger functions: one per legacy table. Each raises an exception when
# a row's revision pointer is being un-set by an UPDATE. The functions
# never fire on INSERTs and never fire on UPDATEs that leave the
# revision column alone — that scoping is enforced by PG's
# ``BEFORE UPDATE OF <column>`` trigger syntax.
_TRIGGER_FUNCTIONS: tuple[str, ...] = (
    """
    CREATE OR REPLACE FUNCTION raise_documents_revision_id_loss()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION
            'documents.current_revision_id cannot be unset once assigned '
            '(TG_OP=%, OLD.current_revision_id=%, NEW.current_revision_id=NULL)',
            TG_OP, OLD.current_revision_id;
    END;
    $$ LANGUAGE plpgsql
    """,
    """
    CREATE OR REPLACE FUNCTION raise_document_images_revision_id_loss()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION
            'document_images.revision_id cannot be unset once assigned '
            '(TG_OP=%, OLD.revision_id=%, NEW.revision_id=NULL)',
            TG_OP, OLD.revision_id;
    END;
    $$ LANGUAGE plpgsql
    """,
    """
    CREATE OR REPLACE FUNCTION raise_document_tables_revision_id_loss()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION
            'document_tables.revision_id cannot be unset once assigned '
            '(TG_OP=%, OLD.revision_id=%, NEW.revision_id=NULL)',
            TG_OP, OLD.revision_id;
    END;
    $$ LANGUAGE plpgsql
    """,
)


# Triggers: ``BEFORE UPDATE OF <revision_col>`` scoped to a single column
# and gated by ``OLD.<col> IS NOT NULL AND NEW.<col> IS NULL``. v1 UPDATEs
# that touch any other column (``status``, ``markdown_s3_key``,
# ``embed_done``, ...) never fire the trigger because ``UPDATE OF``
# skips them. v1 INSERTs are unaffected because the trigger is
# UPDATE-only.
_TRIGGERS: tuple[str, ...] = (
    """
    DROP TRIGGER IF EXISTS trg_documents_revision_id_stable ON documents
    """,
    """
    CREATE TRIGGER trg_documents_revision_id_stable
        BEFORE UPDATE OF current_revision_id ON documents
        FOR EACH ROW
        WHEN (OLD.current_revision_id IS NOT NULL AND NEW.current_revision_id IS NULL)
        EXECUTE FUNCTION raise_documents_revision_id_loss()
    """,
    """
    DROP TRIGGER IF EXISTS trg_document_images_revision_id_stable
        ON document_images
    """,
    """
    CREATE TRIGGER trg_document_images_revision_id_stable
        BEFORE UPDATE OF revision_id ON document_images
        FOR EACH ROW
        WHEN (OLD.revision_id IS NOT NULL AND NEW.revision_id IS NULL)
        EXECUTE FUNCTION raise_document_images_revision_id_loss()
    """,
    """
    DROP TRIGGER IF EXISTS trg_document_tables_revision_id_stable
        ON document_tables
    """,
    """
    CREATE TRIGGER trg_document_tables_revision_id_stable
        BEFORE UPDATE OF revision_id ON document_tables
        FOR EACH ROW
        WHEN (OLD.revision_id IS NOT NULL AND NEW.revision_id IS NULL)
        EXECUTE FUNCTION raise_document_tables_revision_id_loss()
    """,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_names(conn) -> frozenset[str]:
    rows = conn.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
    ).fetchall()
    return frozenset(row[0] for row in rows)


def _applied(conn) -> bool:
    """True if and only if a v2_schema_version row exists for the current version."""
    exists_row = conn.execute(
        text("SELECT to_regclass('public.v2_schema_version') IS NOT NULL")
    ).fetchone()
    if not exists_row or not exists_row[0]:
        return False
    count_row = conn.execute(
        text("SELECT count(*) FROM v2_schema_version WHERE version = :v"),
        {"v": V2_SCHEMA_VERSION},
    ).fetchone()
    return bool(count_row and count_row[0] > 0)


def _capture_legacy_baseline(conn) -> dict[str, tuple[int, int]]:
    """Snapshot ``(count(*), pg_relation_size)`` for each legacy table.

    Called once after the advisory lock is acquired and before any DDL
    mutates the legacy tables.
    """
    baseline: dict[str, tuple[int, int]] = {}
    for tbl in _LEGACY_TABLES_FOR_BASELINE:
        n = conn.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
        size = conn.execute(
            text(f"SELECT pg_relation_size('public.{tbl}'::regclass)")
        ).scalar()
        baseline[tbl] = (int(n), int(size))
    return baseline


def _verify_legacy_unchanged(
    conn, baseline: dict[str, tuple[int, int]]
) -> None:
    """Re-capture row counts and ``pg_relation_size`` and compare against ``baseline``.

    Row-count drift raises ``RuntimeError`` (the migration must not delete
    or insert legacy rows). Size drift is allowed (column additions and
    index installation legitimately grow ``pg_relation_size``); a
    *shrink* is logged to stderr but never raises.
    """
    for tbl, (base_n, base_size) in baseline.items():
        cur_n = int(
            conn.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
        )
        cur_size = int(
            conn.execute(
                text(f"SELECT pg_relation_size('public.{tbl}'::regclass)")
            ).scalar()
        )
        if cur_n != base_n:
            raise RuntimeError(
                f"legacy table {tbl!r} row count drifted: baseline={base_n} "
                f"current={cur_n} (migration must not mutate legacy rows)"
            )
        if cur_size < base_size:
            # Schema shrinking is unexpected — the migration only adds
            # columns and indexes. Surface but don't raise; the schema
            # itself would be wrong if this happened.
            print(
                f"warning: legacy table {tbl!r} size shrank "
                f"(baseline={base_size} current={cur_size})",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_v2_schema(engine: Engine) -> SchemaCheck:
    """Inspect the database and report the v2 schema state.

    Never raises; returns a structured ``SchemaCheck``. The caller can use
    ``is_clean`` to gate Release 1B deployment.
    """
    with engine.connect() as conn:
        applied = _applied(conn)
        if not applied:
            return SchemaCheck(
                applied=False,
                version=None,
                missing_tables=V2_SCHEMA_V1_TABLES,
                extra_tables=frozenset(),
            )
        version_row = conn.execute(
            text("SELECT version FROM v2_schema_version LIMIT 1")
        ).fetchone()
        version = version_row[0] if version_row else None
        tables = _table_names(conn)
        missing = V2_SCHEMA_V1_TABLES - tables
        extra = tables - V2_SCHEMA_V1_TABLES - {
            # known legacy tables that share the public schema
            "abbreviations",
            "agent_traces",
            "api_keys",
            "audit_logs",
            "chat_exchange_summaries",
            "chat_files",
            "chat_files_cleanup",
            "chat_messages",
            "chat_sessions",
            "document_aliases",
            "document_images",
            "document_tables",
            "document_type_system_prompts",
            "document_types",
            "documents",
            "format_metadata",
            "invite_tokens",
            "knowledge_bases",
            "system_settings",
            "telegram_bot_config",
            "telegram_link_codes",
            "telegram_links",
            "tenant_users",
            "tenants",
            "users",
        }
    return SchemaCheck(
        applied=True,
        version=version,
        missing_tables=missing,
        extra_tables=extra,
    )


def apply_v2_schema(engine: Engine) -> None:
    """Apply Release 1A: create the 12 v2 tables, add nullable columns to legacy
    tables, install FKs to ``document_revisions``, install the stable-pointer
    triggers, verify legacy rows are untouched, and record the schema version.

    Idempotent: a second call with the schema already applied is a no-op.
    The transaction commits (or rolls back) when ``engine.begin()`` exits.
    """
    with engine.begin() as conn:
        # Advisory lock first so concurrent migrations serialize. The SQL
        # string is sourced from the module-level ``_ADVISORY_LOCK_SQL``
        # constant so the substring ``pg_advisory_xact_lock`` appears in
        # the code stream above the first ``CREATE TABLE`` literal — see
        # ``test_migrate_module_takes_advisory_lock``.
        conn.execute(
            text(_ADVISORY_LOCK_SQL),
            {"key": V2_MIGRATION_ADVISORY_LOCK_KEY},
        )

        if _applied(conn):
            # Already at version 1 — nothing to do. Idempotent.
            return

        # 1. Capture the legacy-row baseline BEFORE any DDL mutates the
        #    legacy tables. Re-verifying at the end (before the
        #    ``v2_schema_version`` row is written) enforces brief item (5).
        #    Size drift from column additions is expected and tolerated;
        #    row-count drift raises.
        baseline = _capture_legacy_baseline(conn)

        # 2. Create the v2 tables in dependency order.
        for stmt in _CREATE_DDL:
            conn.execute(text(stmt))

        # 3. Add nullable columns to legacy tables (no NOT NULL, no backfill).
        for stmt in _NULLABILITY_COLUMNS:
            conn.execute(text(stmt))

        # 4. Add FK constraints from legacy columns to document_revisions.
        for stmt in _LEGACY_FK_STATEMENTS:
            conn.execute(text(stmt))

        # 5. Install supporting indexes.
        for stmt in _NULLABILITY_INDEXES:
            conn.execute(text(stmt))
        for stmt in _CREATE_INDEXES:
            conn.execute(text(stmt))

        # 6. Install trigger functions and triggers (brief item 4).
        for stmt in _TRIGGER_FUNCTIONS:
            conn.execute(text(stmt))
        for stmt in _TRIGGERS:
            conn.execute(text(stmt))

        # 7. Verify legacy rows are untouched (row-count strict, size
        #    informational — column additions legitimately grow it).
        _verify_legacy_unchanged(conn, baseline)

        # 8. Record the schema version LAST. If anything above raised,
        #    this row would not be written and the migration is
        #    retriable.
        conn.execute(
            text(
                "INSERT INTO v2_schema_version (version, description) "
                "VALUES (:v, :d)"
            ),
            {
                "v": V2_SCHEMA_VERSION,
                "d": "Release 1A: revision storage foundation",
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _normalize_dsn(dsn: str) -> str:
    """Ensure the DSN uses the psycopg3 driver (the sync default in this venv)."""
    if dsn.startswith("postgresql://") or dsn.startswith("postgresql+asyncpg://"):
        # ``postgresql://`` defaults to psycopg2 in SQLAlchemy; we use psycopg3.
        return dsn.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1).replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    return dsn


def make_engine(dsn: str) -> Engine:
    """Create a sync ``Engine`` from a DSN, normalizing to the psycopg3 driver.

    Exposed for test fixtures and other callers that need an engine built
    from a DSN string. The CLI uses the private ``_engine_connect``
    helper internally.
    """
    return create_engine(_normalize_dsn(dsn), future=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.services.agents.v2.persistence.migrate",
        description="Apply or inspect the Release 1A v2 migration.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    apply_p = sub.add_parser("apply", help="Apply the migration (idempotent).")
    apply_p.add_argument(
        "--dsn",
        required=True,
        help="SQLAlchemy postgresql:// DSN (sync, will be normalized to psycopg3).",
    )

    check_p = sub.add_parser("check", help="Report the current v2 schema state.")
    check_p.add_argument(
        "--dsn",
        required=True,
        help="SQLAlchemy postgresql:// DSN (sync, will be normalized to psycopg3).",
    )

    return parser


def _engine_connect(args: argparse.Namespace) -> Engine:
    return create_engine(_normalize_dsn(args.dsn), future=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "apply":
        engine = _engine_connect(args)
        try:
            apply_v2_schema(engine)
            check = check_v2_schema(engine)
        finally:
            engine.dispose()
        print(
            f"applied={check.applied} version={check.version} "
            f"missing={sorted(check.missing_tables)} "
            f"extra={sorted(check.extra_tables)}"
        )
        return 0 if check.is_clean else 1

    if args.cmd == "check":
        engine = _engine_connect(args)
        try:
            check = check_v2_schema(engine)
        finally:
            engine.dispose()
        print(
            f"applied={check.applied} version={check.version} "
            f"missing={sorted(check.missing_tables)} "
            f"extra={sorted(check.extra_tables)}"
        )
        return 0 if check.is_clean else 1

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())