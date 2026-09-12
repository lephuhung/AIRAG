"""Phase 1B — Post-migration v2 ORM registry.

Release 1B only maps the v2 tables that Release 1A (``app.services.
agents.v2.persistence.migrate``) created. This module is the
*post-migration entrypoint* for the v2 ORM mappings: it imports the 11
v2 model classes (which register themselves on
``app.core.database.Base.metadata``) and exposes the runtime gates
that the application uses to refuse startup on a database that has not
been migrated to schema version 1.

Public surface
--------------

- ``LEGACY_STARTUP_TABLES`` — the *exact* set of tables the startup
  ``Base.metadata.create_all(tables=...)`` allowlist may create or
  alter. Every v2 table is **excluded** from this set; legacy v1
  tables are the only ones that ``create_all`` is allowed to touch.
  This is the runtime guarantee that no v2 DDL can leak into
  ``lifespan`` even if ``AUTO_CREATE_TABLES=true``.

- ``register_v2_models()`` — idempotent marker that has already run
  at import time. Provided so the application can prove that v2
  mappings are present (e.g. from a test).

- ``assert_v2_readiness(check)`` — gate called by ``app.main.lifespan``
  after ``check_v2_schema(engine)`` returns. Raises ``RuntimeError``
  with a message naming the Release-1A migration command if the
  schema is missing, at the wrong version, or has missing tables.
  Application code MUST call this before any v2 repository /
  service initializes.

Why import-time registration is safe here
-----------------------------------------

SQLAlchemy declarative ``Base`` subclasses register themselves on
``Base.metadata`` at *class definition* (i.e. at module-import time).
We cannot avoid this without ugly ``__init_subclass__`` tricks, and
the brief's gating requirement is fully satisfied by:

1. ``assert_v2_readiness`` runs before any v2 service starts work.
2. ``create_all(tables=LEGACY_STARTUP_TABLES)`` restricts the startup
   legacy-table allowlist to v1 tables only, so even if the v2 models
   are mapped on ``Base.metadata``, ``create_all`` will not create or
   alter any v2 table.
3. ``check_v2_schema`` is the single source of truth for "is v2
   actually applied?", and ``lifespan`` refuses to start the app
   without ``applied=True, version=1, missing=∅``.

This module never opens a DB connection, never issues DDL, and never
calls ``Base.metadata.create_all``. The brief forbids all of these
("no v2 DDL in startup", "register them in ``app.models.__init__``
only in this post-migration release").

Naming conventions
------------------

- File ``document_ingestion_attempt.py`` maps the table
  ``revision_ingestion_attempts`` to the class
  ``DocumentIngestionAttempt``. The brief uses
  ``DocumentIngestionAttempt`` consistently; the table is named
  ``revision_ingestion_attempts`` because it is owned by a revision
  (the pipeline can have many attempts per revision).
"""

from __future__ import annotations

from typing import Iterable

from app.core.database import Base
from app.services.agents.v2.persistence.migrate import (
    SchemaCheck,
    V2_SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Legacy startup-table allowlist
# ---------------------------------------------------------------------------


# The legacy v1 tables that ``app.main.lifespan`` is allowed to create
# or alter via ``Base.metadata.create_all(tables=...)``. This list is
# the *only* code path that may issue v1 DDL on startup; v2 tables are
# deliberately excluded. Adding a v2 table here would be a regression
# of the Phase 1B deploy gate (it would let startup silently re-create
# a v2 table that the migration owns).
#
# We deliberately include both the v1 tables and any v2 tables'
# nullable-column additions (``documents``, ``document_images``,
# ``document_tables``) that are *legacy* tables (the v2 columns on
# them are nullable per Task 1 brief Step 2).
LEGACY_STARTUP_TABLES: frozenset[str] = frozenset(
    {
        # Legacy v1 tables — these are the only tables ``create_all``
        # may create/alter. Mirrors the legacy whitelist in
        # ``app.services.agents.v2.persistence.migrate.check_v2_schema``
        # (which subtracts this same set when computing ``extra_tables``)
        # so the two stay in sync.
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
        # v2 tables that ``create_all`` MUST NOT touch — listed here
        # as a comment so future maintainers see the explicit ban:
        #   "v2_schema_version",
        #   "document_revisions",
        #   "document_revision_builds",
        #   "document_revision_chunks",
        #   "revision_ingestion_attempts",
        #   "source_arrivals",
        #   "revision_retention_leases",
        #   "conversation_snapshots",
        #   "semantic_snapshots",
        #   "binding_audit",
        #   "evidence_records",
        #   "evidence_uses",
    }
)


# ---------------------------------------------------------------------------
# v2 model imports
# ---------------------------------------------------------------------------
#
# Importing the 11 v2 model modules registers the corresponding ORM
# classes on ``Base.metadata``. This is the single source of truth
# for "v2 models are present" — if any model file is renamed, removed,
# or fails to import, ``Base.metadata.tables`` will reflect the change
# and ``test_v2_orm_tables_are_exactly_v2_schema_v1_tables`` will
# fail loudly.
#
# We import at module scope (not lazily inside ``register_v2_models``)
# because SQLAlchemy declarative metadata is class-definition-time
# global; lazy import after a runtime check would still register the
# same classes on the same metadata. The post-migration gating
# guarantee comes from ``assert_v2_readiness`` (called by ``lifespan``
# before any v2 service runs) plus the
# ``create_all(tables=LEGACY_STARTUP_TABLES)`` allowlist in
# ``lifespan`` — see the module docstring for the full rationale.


# A sentinel ``register_v2_models`` callable: import-side-effect
# triggers ``Base.metadata`` registration. Provided as a public
# function so tests and downstream code can prove registration ran.
def register_v2_models() -> None:
    """Idempotent marker that v2 model imports have already run.

    The v2 model classes register themselves on
    ``app.core.database.Base.metadata`` at class-definition time (i.e.
    at module import). Importing ``app.models.v2_registry`` therefore
    registers them automatically; calling ``register_v2_models()`` is
    a no-op that lets callers prove the registration ran.

    Never opens a DB connection, never issues DDL.
    """
    return None


# Importing the 11 v2 model modules registers the corresponding
# classes on ``Base.metadata``. The import order is preserved for
# readability; SQLAlchemy resolves inter-model FKs lazily so order
# is not significant at runtime.
from app.models.document_revision import (  # noqa: E402,F401
    DocumentRevision,
)
from app.models.document_revision_build import (  # noqa: E402,F401
    DocumentRevisionBuild,
)
from app.models.document_revision_chunk import (  # noqa: E402,F401
    DocumentRevisionChunk,
)
from app.models.document_ingestion_attempt import (  # noqa: E402,F401
    DocumentIngestionAttempt,
)
from app.models.source_arrival import (  # noqa: E402,F401
    SourceArrival,
)
from app.models.revision_retention_lease import (  # noqa: E402,F401
    RevisionRetentionLease,
)
from app.models.conversation_snapshot import (  # noqa: E402,F401
    ConversationSnapshot,
)
from app.models.semantic_snapshot import (  # noqa: E402,F401
    SemanticSnapshot,
)
from app.models.binding_audit import (  # noqa: E402,F401
    BindingAudit,
)
from app.models.evidence_record import (  # noqa: E402,F401
    EvidenceRecord,
)
from app.models.evidence_use import (  # noqa: E402,F401
    EvidenceUse,
)


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


def assert_v2_readiness(check: SchemaCheck) -> None:
    """Refuse startup if the v2 schema is not at exact version 1.

    Called by ``app.main.lifespan`` after ``check_v2_schema(engine)``.
    Raises ``RuntimeError`` with a message that names the Release-1A
    migration command (``python -m app.services.agents.v2.persistence
    .migrate apply --dsn <DSN>``) so an operator can immediately run
    it.

    Failure modes:

    - ``applied=False``: the database has no ``v2_schema_version``
      row. The error message names the migration command.
    - ``applied=True, version != V2_SCHEMA_VERSION``: a future schema
      version is installed but the application is pinned to v1. The
      error message references both the installed version and the
      required version.
    - ``applied=True, version == V2_SCHEMA_VERSION, missing_tables
      non-empty``: the version row was written but the schema is
      incomplete (e.g. partial / divergent environment). The error
      message lists the missing tables.

    On success, the function returns ``None``.
    """
    if not check.applied:
        raise RuntimeError(
            "v2 schema is not applied to this database. "
            "Run the Release-1A migration first: "
            "'python -m app.services.agents.v2.persistence.migrate "
            "apply --dsn <DSN>'. "
            f"SchemaCheck(applied=False, version=None, "
            f"missing_tables={sorted(check.missing_tables)}, "
            f"extra_tables={sorted(check.extra_tables)})"
        )
    if check.version != V2_SCHEMA_VERSION:
        raise RuntimeError(
            f"v2 schema version mismatch: installed={check.version} "
            f"required={V2_SCHEMA_VERSION}. This backend is pinned to "
            f"v2 schema version {V2_SCHEMA_VERSION}; the database is at "
            f"version {check.version}. Either upgrade the backend or "
            f"downgrade the schema."
        )
    if check.missing_tables:
        raise RuntimeError(
            f"v2 schema version {V2_SCHEMA_VERSION} is recorded but the "
            f"following tables are missing: {sorted(check.missing_tables)}. "
            f"This is a divergent / partial migration environment; "
            f"re-run 'python -m app.services.agents.v2.persistence.migrate "
            f"apply --dsn <DSN>' to repair."
        )


def legacy_startup_tables() -> Iterable[str]:
    """Return the ``create_all(tables=...)`` allowlist.

    Returned as an iterable (the frozen set itself is iterable) so
    callers can pass it straight into ``Base.metadata.create_all(tables=
    ...)``.
    """
    return LEGACY_STARTUP_TABLES


def v2_models_registered() -> bool:
    """``True`` iff every expected v2 ORM class is present in ``Base.metadata``.

    This is a runtime check intended for tests and diagnostics; it
    does not require a DB connection. It complements
    ``assert_v2_readiness`` (which checks the live DB) by proving the
    ORM metadata is in sync with the schema contract.
    """
    expected = {
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
    return expected.issubset(set(Base.metadata.tables.keys()))


__all__ = [
    "LEGACY_STARTUP_TABLES",
    "assert_v2_readiness",
    "register_v2_models",
    "legacy_startup_tables",
    "v2_models_registered",
]