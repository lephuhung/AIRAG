"""Release 1A — Raw-SQL migration runner for the v2 revision schema.

This module is the **only** code path in Phase 1 that creates v2 tables.
Release 1B (ORM registration) must not call ``Base.metadata.create_all`` or
emit v2 DDL — it only maps the schema this module creates.

Design constraints (from the Phase 1 plan):

- No ``import app.models`` (Phase 1B owns ORM registration).
- No ``Base.metadata.create_all``.
- All DDL is wrapped in a single transaction preceded by
  ``pg_advisory_xact_lock`` so concurrent migration attempts serialize.
- New columns on legacy tables are nullable and never backfilled.
- The migration is idempotent: re-running it is a no-op once the
  ``v2_schema_version`` row exists.
- The set ``V2_SCHEMA_V1_TABLES`` is the exact contract Release 1B relies on
  (Phase 1B must not assume any table beyond this set).

Module layout:

    V2_SCHEMA_VERSION = 1
    V2_SCHEMA_V1_TABLES = frozenset({...})  # the 12 tables of Release 1A
    SchemaCheck = NamedTuple  # structured result of check_v2_schema
    check_v2_schema(engine) -> SchemaCheck
    apply_v2_schema(engine_or_conn) -> None
    main()  # CLI entrypoint
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

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
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
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
            ON DELETE CASCADE
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
            ON DELETE CASCADE
    )
    """,
    # 5. revision_ingestion_attempts — one row per (document, source identity, build).
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
        UNIQUE (document_id, source_scheme, source_bucket, source_object_key,
                build_profile),
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
        FOREIGN KEY (revision_id) REFERENCES document_revisions(revision_id)
            ON DELETE CASCADE
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
# Helpers
# ---------------------------------------------------------------------------


def _coerce_conn(conn: psycopg.Connection | psycopg.extensions.connection):
    """Accept either a ``psycopg.Connection`` or a plain DB-API connection.

    The migration is deliberately DB-API only — no SQLAlchemy session.
    """
    return conn


def _table_names(conn) -> frozenset[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        rows = cur.fetchall()
    return frozenset(r[0] if not isinstance(r, dict) else r["table_name"] for r in rows)


def _applied(conn) -> bool:
    """True if and only if a v2_schema_version row exists."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.v2_schema_version') IS NOT NULL"
        )
        exists = cur.fetchone()
    if not exists or not (exists[0] if not isinstance(exists, dict) else exists["to_regclass"]):
        return False
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM v2_schema_version WHERE version = %s",
                    (V2_SCHEMA_VERSION,))
        row = cur.fetchone()
    return bool(row and (row[0] if not isinstance(row, dict) else row["count"]) > 0)


def _advisory_lock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (V2_MIGRATION_ADVISORY_LOCK_KEY,),
        )


def _exec_all(conn, statements: Sequence[str]) -> None:
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_v2_schema(conn) -> SchemaCheck:
    """Inspect the database and report the v2 schema state.

    Never raises; returns a structured ``SchemaCheck``. The caller can use
    ``is_clean`` to gate Release 1B deployment.
    """
    conn = _coerce_conn(conn)
    applied = _applied(conn)
    if not applied:
        return SchemaCheck(
            applied=False,
            version=None,
            missing_tables=V2_SCHEMA_V1_TABLES,
            extra_tables=frozenset(),
        )
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM v2_schema_version LIMIT 1")
        row = cur.fetchone()
    version = row[0] if row else None
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


def apply_v2_schema(conn) -> None:
    """Apply Release 1A: create the 12 v2 tables, add nullable columns to legacy
    tables, install indexes, and record the schema version.

    Idempotent: a second call with the schema already applied is a no-op.
    The caller is responsible for committing the transaction.
    """
    conn = _coerce_conn(conn)
    # Advisory lock first so concurrent migrations serialize.
    _advisory_lock(conn)

    if _applied(conn):
        # Already at version 1 — nothing to do. Idempotent.
        return

    # 1. Create the v2 tables in dependency order.
    _exec_all(conn, _CREATE_DDL)

    # 2. Add nullable columns to legacy tables (no NOT NULL, no backfill).
    _exec_all(conn, _NULLABILITY_COLUMNS)

    # 3. Install supporting indexes.
    _exec_all(conn, _NULLABILITY_INDEXES)
    _exec_all(conn, _CREATE_INDEXES)

    # 4. Record the schema version LAST. If anything above raised, the row
    #    would not be written and the migration is retriable.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO v2_schema_version (version, description) "
            "VALUES (%s, %s)",
            (V2_SCHEMA_VERSION, "Release 1A: revision storage foundation"),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.services.agents.v2.persistence.migrate",
        description="Apply or inspect the Release 1A v2 migration.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    apply_p = sub.add_parser("apply", help="Apply the migration (idempotent).")
    apply_p.add_argument(
        "--dsn",
        default=os.environ.get(
            "V2_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/hrag_test_v2"
        ),
    )

    check_p = sub.add_parser("check", help="Report the current v2 schema state.")
    check_p.add_argument(
        "--dsn",
        default=os.environ.get(
            "V2_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/hrag_test_v2"
        ),
    )

    return parser


def _dsn_connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(args.dsn, autocommit=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "apply":
        with _dsn_connect(args) as conn:
            apply_v2_schema(conn)
            conn.commit()
            check = check_v2_schema(conn)
        print(
            f"applied={check.applied} version={check.version} "
            f"missing={sorted(check.missing_tables)} "
            f"extra={sorted(check.extra_tables)}"
        )
        return 0 if check.is_clean else 1

    if args.cmd == "check":
        with _dsn_connect(args) as conn:
            check = check_v2_schema(conn)
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
