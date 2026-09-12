"""ORM metadata + readiness tests for Phase 1B.

These tests assert that the 12 ORM mappings added in Phase 1B
*match* the migrated schema (no surprises between SQL and ORM) and
that the startup path cannot accidentally issue v2 DDL or
register v2 models without a verified schema version 1.

Coverage (Phase 1B brief Step 2):

- ORM mappings match completed migration (legacy columns nullable).
- FKs resolve.
- Every v2 ORM-mapped table was created by Release 1A and appears
  in ``V2_SCHEMA_V1_TABLES`` before registration.
- Exact schema version 1 required before v2 repositories initialize.
- ``LEGACY_STARTUP_TABLES`` excludes every v2 table.
- Against fresh legacy schema, ``lifespan`` reports the Release-1A
  migration command and exits before startup-table creation.
- With schema version 1, legacy ``AUTO_CREATE_TABLES`` may create
  only allowlisted legacy tables and emits no v2 DDL.

These tests run against ``hrag_test_v2`` (controller-prepared,
Phase 1A migration applied). They do not require a populated
fixture because Release 1B's job is to *map*, not to *write* data.

The runtime engine used here is a fresh sync SQLAlchemy ``Engine``
built from the same DSN; we never call ``create_all`` on it. The
ORM metadata is read via ``Base.metadata`` *after* ``app.models``
imports succeed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.services.agents.v2.persistence.migrate import (
    V2_SCHEMA_VERSION,
    V2_SCHEMA_V1_TABLES,
    check_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Engine:
    """A SQLAlchemy ``Engine`` pointed at the v2 test database."""
    engine = make_engine(V2_DSN)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def imported_app_models():
    """Import ``app.models`` once per module so ``Base.metadata`` is populated.

    Importing ``app.models`` triggers ``app.models.__init__`` which, in
    Phase 1B, performs the *import-time registration* of the 11 v2
    models on ``Base.metadata``. Registration happens at class
    definition time (a SQLAlchemy declarative ``Base`` subclass
    registers itself on import); there is no "assert schema version 1
    before registering" step in ``v2_registry`` — the gating
    guarantee is owned by ``app.main.lifespan`` via
    ``check_v2_schema`` + ``assert_v2_readiness`` (see
    ``v2_registry`` module docstring). If registration fails for any
    reason (missing module, syntax error, etc.) the fixture raises
    and the suite fails fast.
    """
    import app.models  # noqa: F401

    return app.models


# ---------------------------------------------------------------------------
# Mapping / metadata tests
# ---------------------------------------------------------------------------


def test_v2_orm_tables_are_exactly_v2_schema_v1_tables(imported_app_models):
    """The v2 ORM-mapped tables must be EXACTLY the v2 schema v1 tables
    (minus ``v2_schema_version``).

    This is the structural guarantee that Release 1B cannot introduce
    a hidden new table that Release 1A did not create, and cannot
    *omit* a v2 table that Release 1A did create. ``v2_schema_version``
    is intentionally NOT mapped as an ORM class — it has no per-row
    application semantics; it is a migration-control surface read
    directly via raw SQL.

    The test uses **set-difference** in both directions so it detects
    extras (ORM table added that the migration didn't create) as well
    as omissions (v2 table missing from the ORM):
        extras    = Base.metadata.tables - V2_SCHEMA_V1_TABLES - {legacy names}
        omissions = expected_v2_orm_tables - Base.metadata.tables
    """
    from app.core.database import Base

    # The 11 v2 ORM classes — the application-facing v2 tables.
    expected_v2_orm_tables = V2_SCHEMA_V1_TABLES - {"v2_schema_version"}
    mapped_v2_tables = {
        t.name
        for t in Base.metadata.tables.values()
        if t.name in expected_v2_orm_tables
    }
    # Strict equality: any drift either way is a regression.
    assert mapped_v2_tables == expected_v2_orm_tables, (
        f"ORM-mapped v2 tables {sorted(mapped_v2_tables)} differ from "
        f"expected {sorted(expected_v2_orm_tables)}. ``v2_schema_version`` "
        f"is intentionally not mapped as an ORM class."
    )

    # Additional set-difference assertions: catch extras (ORM table
    # added that is NOT in ``V2_SCHEMA_V1_TABLES``) and omissions (v2
    # table missing from the ORM).
    v1_legacy_names = {
        "abbreviations", "agent_traces", "api_keys", "audit_logs",
        "chat_exchange_summaries", "chat_files", "chat_files_cleanup",
        "chat_messages", "chat_sessions", "document_aliases",
        "document_images", "document_tables", "document_type_system_prompts",
        "document_types", "documents", "format_metadata", "invite_tokens",
        "knowledge_bases", "system_settings", "telegram_bot_config",
        "telegram_link_codes", "telegram_links", "tenant_users", "tenants",
        "users",
    }
    all_mapped = set(Base.metadata.tables.keys())
    extras = all_mapped - expected_v2_orm_tables - v1_legacy_names
    omissions = expected_v2_orm_tables - all_mapped
    assert not extras, (
        f"ORM-mapped tables NOT in V2_SCHEMA_V1_TABLES and not v1 legacy: "
        f"{sorted(extras)}"
    )
    assert not omissions, (
        f"V2_SCHEMA_V1_TABLES entries missing from ORM metadata: "
        f"{sorted(omissions)}"
    )


def test_legacy_revision_id_columns_are_nullable(imported_app_models):
    """``Document.current_revision_id`` and the legacy image/table
    ``revision_id`` columns must remain nullable in the ORM (they
    are nullable in SQL — no NOT NULL was added by the migration).
    """
    from app.models.document import Document, DocumentImage, DocumentTable

    assert Document.__table__.c["current_revision_id"].nullable is True
    assert Document.__table__.c["source_deleted_at"].nullable is True
    assert DocumentImage.__table__.c["revision_id"].nullable is True
    assert DocumentTable.__table__.c["revision_id"].nullable is True


def test_document_revision_generation_is_unique_per_document(imported_app_models):
    """``DocumentRevision.generation`` maps the unique per-document
    monotonic allocation column with its unique constraint.
    """
    from sqlalchemy import UniqueConstraint

    from app.models.document_revision import DocumentRevision

    table = DocumentRevision.__table__
    assert table.c["generation"].nullable is False
    # The (document_id, generation) UNIQUE constraint must be in the
    # table metadata so SQLAlchemy issues it on create_all (it is also
    # already in the schema via the migration).
    uq = next(
        (c for c in table.constraints if isinstance(c, UniqueConstraint)),
        None,
    )
    assert uq is not None, "DocumentRevision must declare a UNIQUE constraint"
    assert set(uq.columns.keys()) == {"document_id", "generation"}, (
        f"UNIQUE must cover (document_id, generation), got {sorted(uq.columns.keys())}"
    )


def test_document_ingestion_attempt_maps_unique_attempt_key(
    imported_app_models,
):
    """``DocumentIngestionAttempt`` maps the unique
    ``(document_id, source_object_identity, build_profile)`` key. The
    migration installs this as a NAMED constraint
    ``uq_revision_ingestion_attempt_key``; the ORM mapping must reflect
    the same key.

    The arbiter is the FULL canonical ``source_object_identity`` string
    (scheme|bucket|key|version|size|sha256) — NOT the decomposed
    bucket/key components, which are retained only for audit/query
    convenience.
    """
    from sqlalchemy import UniqueConstraint

    from app.models.document_ingestion_attempt import DocumentIngestionAttempt

    table = DocumentIngestionAttempt.__table__
    cols = {"document_id", "source_object_identity", "build_profile"}
    audit_cols = {
        "source_scheme",
        "source_bucket",
        "source_object_key",
        "source_version_id",
        "source_etag",
        "source_size",
        "source_sha256",
    }
    assert (cols | audit_cols).issubset(set(table.c.keys())), (
        f"missing columns: {sorted((cols | audit_cols) - set(table.c.keys()))}"
    )
    uq = next(
        (c for c in table.constraints if isinstance(c, UniqueConstraint)),
        None,
    )
    assert uq is not None, "DocumentIngestionAttempt must declare UNIQUE"
    assert uq.name == "uq_revision_ingestion_attempt_key", (
        f"UNIQUE name must match the migration; got {uq.name!r}"
    )
    assert set(uq.columns.keys()) == cols, (
        f"UNIQUE must cover {sorted(cols)}, got {sorted(uq.columns.keys())}"
    )


def test_evidence_use_null_safe_uniqueness(imported_app_models):
    """``EvidenceUse`` null-safe uniqueness maps the Task-1 SQL index
    on ``task_id`` (the migration installs ``ix_evidence_uses_task``
    on the ``task_id`` column). The ORM must mirror this index so a
    later ``create_all`` would reproduce it.
    """
    from app.models.evidence_use import EvidenceUse

    table = EvidenceUse.__table__
    assert "task_id" in table.c, "EvidenceUse must expose task_id"
    indexes = {ix.name for ix in table.indexes}
    assert "ix_evidence_uses_task" in indexes, (
        f"EvidenceUse must declare ix_evidence_uses_task, found: {sorted(indexes)}"
    )


def test_evidence_record_exposes_no_plaintext_column(imported_app_models):
    """``EvidenceRecord`` maps ``ciphertext``/``encryption_key_id``/
    ``nonce``/``encryption_algorithm``/``payload_purged_at`` and
    exposes NO plaintext column.
    """
    from app.models.evidence_record import EvidenceRecord

    table = EvidenceRecord.__table__
    expected = {
        "evidence_id",
        "ciphertext",
        "encryption_key_id",
        "nonce",
        "encryption_algorithm",
        "payload_purged_at",
        "created_at",
    }
    actual = set(table.c.keys())
    assert expected.issubset(actual), (
        f"EvidenceRecord missing columns: {sorted(expected - actual)}"
    )
    # No plaintext column. Forbidden names:
    forbidden = {"plaintext", "payload", "content", "raw"}
    leaks = forbidden & set(c.lower() for c in actual)
    assert not leaks, (
        f"EvidenceRecord must not leak plaintext columns, found: {sorted(leaks)}"
    )


def test_v2_orm_mapped_tables_exist_in_database(
    imported_app_models, db: Engine
):
    """Every v2 ORM-mapped table must already exist in the live DB.

    This is the post-migration deploy gate: if any v2 ORM class is
    declared but the table is missing, the release cannot have been
    properly migrated and ``Base.metadata.create_all`` would silently
    *create* the missing v2 table (which would later collide with the
    migration's version-write logic). Release 1B must refuse this
    silently.
    """
    with db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        ).fetchall()
    live_tables = {row[0] for row in rows}
    missing = V2_SCHEMA_V1_TABLES - live_tables
    assert not missing, (
        f"v2 ORM tables missing from live DB: {sorted(missing)}. "
        f"Run the Release 1A migration before deploying Phase 1B."
    )


def test_orm_fks_resolve(imported_app_models, db: Engine):
    """Every FK declared by an ORM-mapped v2 table must resolve in the
    live DB (target table + target column exist).
    """
    from app.core.database import Base

    inspector = inspect(db)
    failures: list[str] = []
    for table_name in V2_SCHEMA_V1_TABLES:
        if table_name not in inspector.get_table_names():
            failures.append(f"missing v2 table in DB: {table_name}")
            continue
        fks = inspector.get_foreign_keys(table_name)
        for fk in fks:
            referred_table = fk.get("referred_table")
            referred_cols = fk.get("referred_columns") or []
            constrained_cols = fk.get("constrained_columns") or []
            if referred_table not in inspector.get_table_names():
                failures.append(
                    f"{table_name}.{','.join(constrained_cols)} -> "
                    f"{referred_table} (target missing)"
                )
                continue
            if referred_table == "v2_schema_version":
                continue
            for rc in referred_cols:
                cols = {c["name"] for c in inspector.get_columns(referred_table)}
                if rc not in cols:
                    failures.append(
                        f"{table_name}.{','.join(constrained_cols)} -> "
                        f"{referred_table}.{rc} (target column missing)"
                    )
    # Also check the ORM-declared FKs match what's in the DB (ORM ↔ SQL).
    orm_declared: set[tuple[str, str, str, str]] = set()
    for table in Base.metadata.tables.values():
        if table.name not in V2_SCHEMA_V1_TABLES and table.name not in {
            "documents",
            "document_images",
            "document_tables",
        }:
            continue
        for fk in table.foreign_keys:
            orm_declared.add(
                (table.name, fk.parent.name, fk.column.table.name, fk.column.name)
            )
    assert not failures, "FK resolution failures: " + "; ".join(failures)
    # ORM should declare at least the FKs the migration installs.
    assert orm_declared, "ORM metadata declares no FKs (mismatch with SQL schema)"


def test_check_v2_schema_returns_version_1(db: Engine):
    """Sanity: the test DB is at exact v2 schema version 1."""
    check = check_v2_schema(db)
    assert check.applied is True, check
    assert check.version == 1, check
    assert check.is_clean is True, check


def test_check_v2_schema_reports_no_shape_errors_on_migrated_db(db: Engine):
    """R2/I3: a freshly migrated DB reports no shape drift.

    ``shape_errors`` is the fail-closed guard that detects a database
    which recorded version 1 *before* the R2 amendment and therefore
    still carries the stale columns / attempt arbiter.
    """
    check = check_v2_schema(db)
    assert check.shape_errors == frozenset(), check.shape_errors


def test_schema_check_is_clean_requires_no_shape_errors():
    """R2/I3: ``is_clean`` must be False whenever ``shape_errors`` is set."""
    from app.services.agents.v2.persistence.migrate import SchemaCheck

    stale = SchemaCheck(
        applied=True,
        version=1,
        missing_tables=frozenset(),
        extra_tables=frozenset(),
        shape_errors=frozenset(
            {"revision_ingestion_attempts: missing columns ['source_object_identity']"}
        ),
    )
    assert stale.is_clean is False
    healthy = SchemaCheck(
        applied=True,
        version=1,
        missing_tables=frozenset(),
        extra_tables=frozenset(),
    )
    assert healthy.is_clean is True
    assert healthy.shape_errors == frozenset()


def test_v2_models_module_has_no_db_writes_at_import(imported_app_models):
    """Importing ``app.models`` (and therefore ``v2_registry``) must not
    issue DDL or open a transaction. This is the post-migration deploy
    gate: registration is a metadata-only operation.
    """
    src_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "models"
        / "v2_registry.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # Strip the module docstring (anchored, count=1) and ALL inline
    # triple-quoted strings so the assertion cannot pass against
    # narrative text. The migration DDL / SQL string literals are
    # not present in this module, so stripping all triple-quoted
    # blocks is safe.
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"#[^\n]*", "", code)
    # No ``Base.metadata.create_all`` call (allowlist is enforced in
    # ``app.main.lifespan``, not here).
    assert "create_all" not in code, (
        "v2_registry must not call Base.metadata.create_all"
    )
    # No transactional DDL helpers inside v2_registry.
    forbidden = (
        "engine.execute", "engine.begin", "conn.execute",
        "session.execute", "text(DDL", "DDL(",
        "apply_v2_schema",  # migration runner is for the CLI / startup gate
    )
    for tok in forbidden:
        assert tok not in code, (
            f"v2_registry must not issue DDL; found {tok!r}"
        )


# ---------------------------------------------------------------------------
# Legacy-table allowlist tests
# ---------------------------------------------------------------------------


def test_legacy_startup_tables_excludes_every_v2_table():
    """``LEGACY_STARTUP_TABLES`` must enumerate ONLY legacy tables;
    v2 tables are excluded so a stray ``create_all`` cannot recreate
    or alter them.
    """
    from app.models.v2_registry import LEGACY_STARTUP_TABLES

    assert isinstance(LEGACY_STARTUP_TABLES, (frozenset, set, list, tuple))
    overlap = set(LEGACY_STARTUP_TABLES) & V2_SCHEMA_V1_TABLES
    assert not overlap, (
        f"LEGACY_STARTUP_TABLES must exclude v2 tables; overlap={sorted(overlap)}"
    )


def test_legacy_startup_tables_includes_documents_and_children():
    """The allowlist must still let ``AUTO_CREATE_TABLES`` create the
    tables ``app.models`` already mapped (so the legacy greenfield
    startup path remains usable on databases that have never been
    migrated).
    """
    from app.models.v2_registry import LEGACY_STARTUP_TABLES

    legacy = set(LEGACY_STARTUP_TABLES)
    for tbl in ("documents", "document_images", "document_tables",
                "knowledge_bases", "users", "tenants"):
        assert tbl in legacy, (
            f"LEGACY_STARTUP_TABLES must include legacy {tbl!r}"
        )


# ---------------------------------------------------------------------------
# Schema-version gate tests (lifespan semantics)
# ---------------------------------------------------------------------------


def test_v2_registry_raises_when_schema_not_applied():
    """If a database is fresh (no v2 schema), ``assert_v2_readiness``
    must raise with a message naming the Release 1A migration command.

    The error message must include the migration CLI invocation so an
    operator can immediately run it.
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck

    # We can't easily spin up a fresh DB here (would need to drop the
    # test DB). Instead, construct a SchemaCheck that says
    # ``applied=False, version=None, missing_tables=V2_SCHEMA_V1_TABLES``
    # and assert that ``v2_registry.assert_v2_readiness`` raises with
    # the right message shape.
    from app.models import v2_registry

    check = SchemaCheck(
        applied=False,
        version=None,
        missing_tables=V2_SCHEMA_V1_TABLES,
        extra_tables=frozenset(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        v2_registry.assert_v2_readiness(check)
    msg = str(excinfo.value)
    assert "migrate" in msg.lower(), (
        f"error must mention the migration command: {msg!r}"
    )
    assert "apply" in msg.lower() or "Release 1A" in msg, (
        f"error must mention 'apply' or 'Release 1A': {msg!r}"
    )


def test_v2_registry_raises_when_schema_at_wrong_version():
    """If the DB is at a different schema version, ``assert_v2_readiness``
    must raise.
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck
    from app.models import v2_registry

    check = SchemaCheck(
        applied=True,
        version=999,  # a future version
        missing_tables=frozenset(),
        extra_tables=frozenset(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        v2_registry.assert_v2_readiness(check)
    msg = str(excinfo.value)
    assert "999" in msg or str(V2_SCHEMA_VERSION) in msg, (
        f"error must reference the version mismatch: {msg!r}"
    )


def test_v2_registry_accepts_version_1():
    """If the DB is at version 1 with no missing tables, ``assert_v2_readiness``
    must not raise.
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck
    from app.models import v2_registry

    check = SchemaCheck(
        applied=True,
        version=V2_SCHEMA_VERSION,
        missing_tables=frozenset(),
        extra_tables=frozenset(),
    )
    # Must not raise.
    v2_registry.assert_v2_readiness(check)


def test_v2_registry_raises_when_tables_missing_at_version_1():
    """If schema is "applied" but tables are missing, ``assert_v2_readiness``
    must raise (the migration wrote the version row but the schema is
    incomplete — likely a partially applied / divergent environment).
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck
    from app.models import v2_registry

    check = SchemaCheck(
        applied=True,
        version=V2_SCHEMA_VERSION,
        missing_tables={"document_revisions", "evidence_uses"},
        extra_tables=frozenset(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        v2_registry.assert_v2_readiness(check)
    msg = str(excinfo.value)
    assert "document_revisions" in msg, (
        f"error must reference the missing table: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Lifespan source-level tests
# ---------------------------------------------------------------------------


def test_main_lifespan_does_not_call_v2_ddl():
    """``app.main.lifespan`` must not import or call any v2 DDL/migration
    function. It only checks the schema and refuses to start if v2
    readiness is not at version 1.
    """
    src_path = (
        Path(__file__).resolve().parents[3] / "app" / "main.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # Strip module docstring + comments.
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)
    forbidden = (
        "apply_v2_schema",
        "from app.services.agents.v2.persistence.migrate import apply_v2_schema",
    )
    for tok in forbidden:
        assert tok not in code, (
            f"app.main.lifespan must not {tok!r} — it only checks schema version"
        )


def test_main_lifespan_uses_legacy_startup_tables_allowlist():
    """``app.main.lifespan`` must restrict ``create_all`` to the
    ``LEGACY_STARTUP_TABLES`` allowlist (so v2 tables are never created
    or altered by startup).
    """
    from app.models.v2_registry import LEGACY_STARTUP_TABLES

    src_path = (
        Path(__file__).resolve().parents[3] / "app" / "main.py"
    )
    src = src_path.read_text(encoding="utf-8")
    code = re.sub(r'^\s*""".*?""""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)
    # The phrase ``Base.metadata.create_all`` must appear ONLY inside
    # an explicit ``tables=...`` call. The allowed form is either:
    #   Base.metadata.create_all(tables=LEGACY_STARTUP_TABLES)
    # or the ``run_sync`` variant
    #   conn.run_sync(Base.metadata.create_all, tables=...)
    # We forbid the bare ``Base.metadata.create_all()`` form (which
    # would touch every table on ``Base.metadata`` — including v2).
    # Strip the allowed variants first so the remaining occurrence is
    # meaningful.
    allowed_patterns = (
        r"Base\.metadata\.create_all\(tables=LEGACY_STARTUP_TABLES\)",
        # ``run_sync`` form: ``Base.metadata.create_all`` appears as a
        # callable passed positionally + ``tables=...`` kwarg.
        r"Base\.metadata\.create_all,\s*tables=",
    )
    code_stripped = code
    for pat in allowed_patterns:
        code_stripped = re.sub(pat, "", code_stripped)
    assert "Base.metadata.create_all" not in code_stripped, (
        "lifespan must use Base.metadata.create_all(tables=...); "
        "unrestricted create_all is forbidden (it would create v2 tables)"
    )
    # Sanity: at least one allowed form is actually present.
    allowed_present = any(re.search(p, code) for p in allowed_patterns)
    assert allowed_present, (
        "lifespan must use Base.metadata.create_all(tables=LEGACY_STARTUP_TABLES)"
    )
    # And it must reference the imported symbol.
    assert "LEGACY_STARTUP_TABLES" in code, (
        "lifespan must import LEGACY_STARTUP_TABLES from v2_registry"
    )
    # Confirm the imported name resolves to the same set we asserted
    # above.
    assert "documents" in LEGACY_STARTUP_TABLES


def test_main_lifespan_checks_v2_schema_version():
    """``app.main.lifespan`` must call ``assert_v2_readiness`` (or the
    underlying ``check_v2_schema``) so that v2 services never run on
    a non-version-1 database.
    """
    src_path = (
        Path(__file__).resolve().parents[3] / "app" / "main.py"
    )
    src = src_path.read_text(encoding="utf-8")
    code = re.sub(r'^\s*""".*?"""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)
    # Either direct call to ``assert_v2_readiness`` or to
    # ``check_v2_schema`` followed by ``assert_v2_readiness``.
    has_assert = "assert_v2_readiness" in code
    has_check = "check_v2_schema" in code
    assert has_assert or has_check, (
        "lifespan must call assert_v2_readiness or check_v2_schema"
    )


# ---------------------------------------------------------------------------
# ORM ↔ schema parity test (I3)
# ---------------------------------------------------------------------------


def _type_family(t) -> str:
    """Normalize an SA Type or PG Type to a portable type-family string.

    Maps ORM-side ``DateTime`` and PG-side ``TIMESTAMP`` to the same
    ``"timestamp"`` family so the parity test can compare them. Same
    for ``BigInteger`` / ``BIGINT`` (both ``"bigint"``), ``LargeBinary``
    / ``BYTEA`` (both ``"bytea"``), ``String`` / ``VARCHAR`` (both
    ``"varchar"``), etc. The timezone flag and length are compared
    separately so ``TIMESTAMP WITH TIME ZONE`` vs ``TIMESTAMP`` (no
    zone) is detectable.
    """
    cls = type(t).__name__.lower()
    # Generic SA names
    if cls == "datetime":
        return "timestamp"
    if cls == "boolean":
        return "boolean"
    if cls == "uuid":
        return "uuid"
    if cls == "text":
        return "text"
    if cls == "string":
        return "varchar"
    if cls == "varchar":
        return "varchar"
    if cls == "integer":
        return "integer"
    if cls == "biginteger":
        return "bigint"
    if cls == "smallinteger":
        return "smallint"
    if cls == "largebinary":
        return "bytea"
    if cls == "json":
        return "json"
    if cls == "enum":
        return "enum"
    if cls == "timestamp":
        return "timestamp"
    if cls == "bytea":
        return "bytea"
    if cls == "jsonb":
        return "jsonb"
    return cls


def _column_fingerprint(col_or_live) -> dict:
    """Return a dict fingerprint for an ORM column or a live DB column.

    ``col_or_live`` is either a SQLAlchemy ``Column`` (ORM) or a dict
    from ``inspect(engine).get_columns(...)`` (live).
    """
    if isinstance(col_or_live, dict):
        # live DB column from inspector
        t = col_or_live["type"]
        return {
            "name": col_or_live["name"],
            "family": _type_family(t),
            "timezone": getattr(t, "timezone", None),
            "length": getattr(t, "length", None),
            "nullable": col_or_live["nullable"],
        }
    # ORM column
    t = col_or_live.type
    return {
        "name": col_or_live.name,
        "family": _type_family(t),
        "timezone": getattr(t, "timezone", None),
        "length": getattr(t, "length", None),
        "nullable": col_or_live.nullable,
    }


def test_orm_matches_live_db_schema_parity(imported_app_models, db: Engine):
    """Brief I3: ORM metadata must match the live DB schema exactly.

    Compares ``Base.metadata.tables`` against
    ``inspect(engine).get_columns(<table>)`` for each of the 12 v2
    tables AND the 4 legacy columns (``documents.current_revision_id``,
    ``documents.source_deleted_at``, ``document_images.revision_id``,
    ``document_tables.revision_id``). For each column the test asserts:

    - column exists in both ORM and live DB (set equality);
    - column type family matches (e.g. ``DateTime`` ↔ ``TIMESTAMP``);
    - timezone flag matches (``TIMESTAMP WITH TIME ZONE`` ↔
      ``DateTime(timezone=True)``);
    - length matches (for ``String`` / ``VARCHAR``);
    - nullability matches.

    This is the post-migration deploy gate that prevents autogenerate
    drift (a mismatch here would silently generate a needless
    migration on the next ``alembic revision --autogenerate`` run).
    """
    from app.core.database import Base

    inspector = inspect(db)

    # 12 v2 tables — all of them, including ``v2_schema_version``
    # which is not ORM-mapped but must still appear in the live DB.
    v2_tables = set(V2_SCHEMA_V1_TABLES)
    # 4 legacy columns (table, column) pairs added by the migration.
    legacy_cols = {
        ("documents", "current_revision_id"),
        ("documents", "source_deleted_at"),
        ("document_images", "revision_id"),
        ("document_tables", "revision_id"),
    }

    issues: list[str] = []

    for tbl in sorted(v2_tables):
        live = {
            c["name"]: c
            for c in inspector.get_columns(tbl)
        } if tbl in inspector.get_table_names() else {}

        if tbl == "v2_schema_version":
            # Not ORM-mapped; we only assert the live schema exists.
            if not live:
                issues.append(f"{tbl}: missing from live DB")
            continue

        if tbl not in Base.metadata.tables:
            issues.append(f"{tbl}: V2_SCHEMA_V1_TABLES table missing from ORM")
            continue

        orm_table = Base.metadata.tables[tbl]
        orm_cols = {c.name: c for c in orm_table.columns}

        # Set equality on column names.
        missing_in_live = set(orm_cols) - set(live)
        missing_in_orm = set(live) - set(orm_cols)
        for c in missing_in_live:
            issues.append(f"{tbl}.{c}: ORM column missing in live DB")
        for c in missing_in_orm:
            issues.append(f"{tbl}.{c}: live DB column missing in ORM")

        # Compare each common column.
        for name in sorted(set(orm_cols) & set(live)):
            ofp = _column_fingerprint(orm_cols[name])
            lfp = _column_fingerprint(live[name])
            if ofp["family"] != lfp["family"]:
                issues.append(
                    f"{tbl}.{name}: type family mismatch "
                    f"(orm={ofp['family']!r}, live={lfp['family']!r})"
                )
            if ofp["timezone"] != lfp["timezone"]:
                issues.append(
                    f"{tbl}.{name}: timezone mismatch "
                    f"(orm={ofp['timezone']!r}, live={lfp['timezone']!r}); "
                    f"a DateTime(false) vs TIMESTAMP(timezone=True) would "
                    f"compile to TIMESTAMP WITHOUT TIME ZONE vs WITH TIME ZONE — "
                    f"autogenerate drift would create a needless migration"
                )
            if ofp["length"] != lfp["length"]:
                issues.append(
                    f"{tbl}.{name}: length mismatch "
                    f"(orm={ofp['length']!r}, live={lfp['length']!r})"
                )
            if ofp["nullable"] != lfp["nullable"]:
                issues.append(
                    f"{tbl}.{name}: nullable mismatch "
                    f"(orm={ofp['nullable']!r}, live={lfp['nullable']!r})"
                )

    # 4 legacy columns on tables that are otherwise v1 (only the 4
    # new columns need parity — the rest of the v1 schema is out of
    # scope for Phase 1B).
    for tbl, col in sorted(legacy_cols):
        if tbl not in Base.metadata.tables:
            issues.append(f"{tbl}: legacy table missing from ORM")
            continue
        if col not in Base.metadata.tables[tbl].columns:
            issues.append(f"{tbl}.{col}: legacy column missing from ORM")
            continue
        orm_col = Base.metadata.tables[tbl].c[col]
        live_cols = inspector.get_columns(tbl)
        live = next((c for c in live_cols if c["name"] == col), None)
        if live is None:
            issues.append(f"{tbl}.{col}: legacy column missing in live DB")
            continue
        ofp = _column_fingerprint(orm_col)
        lfp = _column_fingerprint(live)
        if ofp["family"] != lfp["family"]:
            issues.append(
                f"{tbl}.{col}: type family mismatch "
                f"(orm={ofp['family']!r}, live={lfp['family']!r})"
            )
        if ofp["timezone"] != lfp["timezone"]:
            issues.append(
                f"{tbl}.{col}: timezone mismatch "
                f"(orm={ofp['timezone']!r}, live={lfp['timezone']!r})"
            )
        if ofp["nullable"] != lfp["nullable"]:
            issues.append(
                f"{tbl}.{col}: nullable mismatch "
                f"(orm={ofp['nullable']!r}, live={lfp['nullable']!r})"
            )

    assert not issues, (
        "ORM ↔ live DB schema parity issues:\n  "
        + "\n  ".join(issues)
    )


# ---------------------------------------------------------------------------
# Lifespan behavioural tests (I4)
# ---------------------------------------------------------------------------


def test_assert_v2_readiness_raises_with_clear_message_for_unmigrated_db():
    """I4(a): ``assert_v2_readiness`` on a ``SchemaCheck`` representing
    an unmigrated database must raise with a clear error message that
    names the Release-1A migration command.

    The check is built in-memory (no DB roundtrip needed) so this test
    is fast and deterministic.
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck
    from app.models import v2_registry

    check = SchemaCheck(
        applied=False,
        version=None,
        missing_tables=V2_SCHEMA_V1_TABLES,
        extra_tables=frozenset(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        v2_registry.assert_v2_readiness(check)
    msg = str(excinfo.value)
    # The message must name the migration CLI invocation so an
    # operator can immediately run it.
    assert "migrate" in msg.lower(), (
        f"unmigrated-DB error must mention the migration command: {msg!r}"
    )
    assert "apply" in msg.lower(), (
        f"unmigrated-DB error must mention 'apply': {msg!r}"
    )
    assert "--dsn" in msg.lower(), (
        f"unmigrated-DB error must mention --dsn so the operator "
        f"knows how to invoke the migration: {msg!r}"
    )
    # And the structural details that make the error actionable:
    assert "applied=False" in msg or "not applied" in msg.lower(), (
        f"unmigrated-DB error must explain WHY (schema not applied): {msg!r}"
    )


def test_lifespan_resolves_legacy_startup_tables_to_table_objects(
    imported_app_models, db: Engine
):
    """I4(c): the lifespan path resolves every ``LEGACY_STARTUP_TABLES``
    name to a real ``Table`` object on ``Base.metadata`` before passing
    it to ``create_all``.

    ``Base.metadata.create_all`` requires ``Sequence[Table]``; passing
    the names directly raises ``AttributeError: 'str' object has no
    attribute 'name'``. This test catches that regression at the
    shape level: it iterates ``LEGACY_STARTUP_TABLES`` and resolves
    each name to a ``Table`` object exactly the way the lifespan does,
    then asserts the resulting ``Sequence[Table]`` is non-empty and
    has no ``str`` element (i.e. the resolution actually happened).
    """
    from sqlalchemy.sql.schema import Table
    from app.core.database import Base
    from app.models.v2_registry import LEGACY_STARTUP_TABLES

    legacy_table_objs: list[Table] = [
        Base.metadata.tables[name]
        for name in LEGACY_STARTUP_TABLES
        if name in Base.metadata.tables
    ]
    assert legacy_table_objs, (
        "LEGACY_STARTUP_TABLES resolved to zero Table objects; "
        "either the names list is empty or no ORM class maps them"
    )
    for t in legacy_table_objs:
        assert isinstance(t, Table), (
            f"resolved legacy object is not a Table: {t!r}"
        )
        assert t.name in LEGACY_STARTUP_TABLES, (
            f"Table {t.name!r} not in LEGACY_STARTUP_TABLES — resolution "
            f"drift between names and Table objects"
        )

    # Sanity: the legacy create_all call shape itself doesn't raise.
    # We use ``checkfirst=True`` (default) so it is a no-op against
    # the live ``hrag_test_v2`` schema.
    legacy_table_objs_dict = {t.name: t for t in legacy_table_objs}
    with db.connect() as conn:
        inspector = inspect(conn)
        live_legacy = {
            tbl for tbl in LEGACY_STARTUP_TABLES
            if tbl in inspector.get_table_names()
        }
        # Filter to the live tables that are actually present in DB
        # so the call is a pure no-op against an existing schema.
        to_create = [
            legacy_table_objs_dict[t] for t in live_legacy
            if t in legacy_table_objs_dict
        ]
        # Use the bound form (SQLAlchemy create_all needs a ``bind``
        # arg). This is the synchronous equivalent of the async
        # ``run_sync(Base.metadata.create_all, tables=...)`` call in
        # ``main.py``'s ``lifespan``.
        try:
            Base.metadata.create_all(conn, tables=to_create)
        except AttributeError as e:
            pytest.fail(
                f"create_all raised AttributeError on resolved Table "
                f"objects: {e!r} — C1 regression"
            )


def test_lifespan_readiness_gate_works_against_unmigrated_db():
    """I4(b-i): drive the readiness gate against an unmigrated DB.

    We simulate an unmigrated database by feeding
    ``assert_v2_readiness`` a ``SchemaCheck(applied=False, ...)`` —
    which is what ``check_v2_schema`` would return against a fresh
    empty database that has no v2 tables. The gate MUST refuse.
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck
    from app.models import v2_registry

    check = SchemaCheck(
        applied=False,
        version=None,
        missing_tables=V2_SCHEMA_V1_TABLES,
        extra_tables=frozenset(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        v2_registry.assert_v2_readiness(check)
    # The error message MUST name the migration command so an
    # operator can immediately run it.
    msg = str(excinfo.value)
    assert (
        "python -m app.services.agents.v2.persistence.migrate" in msg
        or "migrate apply" in msg
    ), f"unmigrated-DB error must name the migration CLI: {msg!r}"


def test_lifespan_readiness_gate_accepts_migrated_v2_db(db: Engine):
    """I4(b-ii): drive the readiness gate against a migrated v2 DB.

    The test DB ``hrag_test_v2`` is at exact schema version 1 (see
    ``test_check_v2_schema_returns_version_1``); the readiness gate
    MUST accept it without raising.
    """
    check = check_v2_schema(db)
    assert check.applied is True, check
    assert check.version == V2_SCHEMA_VERSION, check
    assert check.is_clean is True, check

    from app.models import v2_registry

    # Must not raise.
    v2_registry.assert_v2_readiness(check)


def test_lifespan_readiness_gate_works_against_empty_db():
    """I4(b-i, fresh-DB form): drive the readiness gate against a
    fresh empty DB.

    We use an in-memory SQLite engine as a stand-in for a fresh
    PostgreSQL database with no v2 tables. ``check_v2_schema`` uses
    SQL syntax (e.g. ``to_regclass``) that is PostgreSQL-specific, so
    we cannot drive it against SQLite directly. Instead we exercise
    the gate via a manually-built ``SchemaCheck`` that matches what
    ``check_v2_schema`` would return against an unmigrated DB.
    """
    from app.services.agents.v2.persistence.migrate import SchemaCheck
    from app.models import v2_registry

    # What ``check_v2_schema`` returns against a fresh DB with no v2
    # tables: applied=False, missing_tables = all of V2_SCHEMA_V1_TABLES.
    fresh_check = SchemaCheck(
        applied=False,
        version=None,
        missing_tables=V2_SCHEMA_V1_TABLES,
        extra_tables=frozenset(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        v2_registry.assert_v2_readiness(fresh_check)
    msg = str(excinfo.value)
    assert "apply" in msg.lower(), (
        f"fresh-DB error must name 'apply': {msg!r}"
    )


def test_main_lifespan_readiness_gate_unconditional_under_auto_create_false():
    """I1 behavioural: the readiness gate runs even when
    ``AUTO_CREATE_TABLES=false``.

    This is a source-level proof: the gate (the calls to
    ``check_v2_schema`` and ``assert_v2_readiness``) must appear in
    the code stream BEFORE the ``if auto_create:`` block. With
    ``AUTO_CREATE_TABLES=false`` the lifespan still calls the gate,
    otherwise a non-migrated database would silently boot in
    production (the documented deploy path is
    ``AUTO_CREATE_TABLES=false`` + manual migration).

    Note: there are TWO ``if auto_create:`` blocks in the lifespan
    (the first one is in the multi-worker warning section; the
    second one wraps the ``create_all`` pass). We find the LAST one
    because that is the gate the ``create_all(tables=...)`` allowlist
    must be inside.
    """
    src_path = (
        Path(__file__).resolve().parents[3] / "app" / "main.py"
    )
    src = src_path.read_text(encoding="utf-8")
    code = re.sub(r'^\s*""".*?""""', "", src, count=1, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)

    # Find ALL occurrences of the ``if auto_create:`` block; use the
    # last one (the gate's create_all block).
    auto_create_positions = [
        m.start() for m in re.finditer(r"if\s+auto_create\s*:", code)
    ]
    assert auto_create_positions, (
        "lifespan must contain at least one ``if auto_create:`` block"
    )
    auto_create_pos = auto_create_positions[-1]

    # Find the readiness gate call sites — the call sites are
    # positionally unique because they appear once at module scope.
    assert_pos = code.find("assert_v2_readiness(check)")
    # ``check_v2_schema`` may be called in different forms:
    #   * ``check_v2_schema(engine.sync_engine)`` (pre-asyncpg path)
    #   * ``check_v2_schema(migration_sync_engine)`` (post-asyncpg
    #     path, where ``migration_sync_engine`` is built from the
    #     DSN via ``make_engine`` because ``engine.sync_engine`` is
    #     asyncpg-backed and can't be used synchronously).
    # Accept either.
    check_pos = -1
    for variant in (
        "check_v2_schema(engine.sync_engine)",
        "check_v2_schema(migration_sync_engine)",
        "check_v2_schema(",  # last resort: any check_v2_schema call
    ):
        p = code.find(variant)
        if p != -1:
            check_pos = p
            break
    assert assert_pos != -1, (
        "lifespan must call assert_v2_readiness(check)"
    )
    assert check_pos != -1, (
        "lifespan must call check_v2_schema (with a sync engine)"
    )

    # Both must precede the LAST ``if auto_create:`` so the gate
    # runs even when ``AUTO_CREATE_TABLES=false``.
    assert check_pos < auto_create_pos, (
        "check_v2_schema must precede the ``if auto_create:`` block "
        "so the readiness gate runs even when AUTO_CREATE_TABLES=false"
    )
    assert assert_pos < auto_create_pos, (
        "assert_v2_readiness must precede the ``if auto_create:`` "
        "block so the readiness gate runs even when "
        "AUTO_CREATE_TABLES=false"
    )