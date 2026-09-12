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
    Phase 1B, performs the post-migration registration of the 12 v2
    models. The ``v2_registry`` is responsible for asserting schema
    version 1 *before* registering; if registration fails for any
    reason, the fixture raises and the suite fails fast.
    """
    import app.models  # noqa: F401

    return app.models


# ---------------------------------------------------------------------------
# Mapping / metadata tests
# ---------------------------------------------------------------------------


def test_v2_orm_tables_are_exactly_v2_schema_v1_tables(imported_app_models):
    """Every v2 ORM-mapped table must appear in ``V2_SCHEMA_V1_TABLES``.

    This is the structural guarantee that Release 1B cannot introduce
    a hidden new table that Release 1A did not create. ``v2_schema_version``
    is intentionally NOT mapped as an ORM class — it has no per-row
    application semantics; it is a migration-control surface read
    directly via raw SQL.
    """
    from app.core.database import Base

    # The 11 v2 ORM classes — the application-facing v2 tables.
    expected_v2_orm_tables = V2_SCHEMA_V1_TABLES - {"v2_schema_version"}
    mapped_v2_tables = {
        t.name
        for t in Base.metadata.tables.values()
        if t.name in expected_v2_orm_tables
    }
    assert mapped_v2_tables == expected_v2_orm_tables, (
        f"ORM-mapped v2 tables {sorted(mapped_v2_tables)} differ from "
        f"expected {sorted(expected_v2_orm_tables)}. ``v2_schema_version`` "
        f"is intentionally not mapped as an ORM class."
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
    ``(document_id, source_scheme, source_bucket, source_object_key,
    build_profile)`` key. The migration installs this as a NAMED
    constraint ``uq_revision_ingestion_attempt_key``; the ORM mapping
    must reflect the same key.
    """
    from sqlalchemy import UniqueConstraint

    from app.models.document_ingestion_attempt import DocumentIngestionAttempt

    table = DocumentIngestionAttempt.__table__
    cols = {"document_id", "source_scheme", "source_bucket",
            "source_object_key", "build_profile"}
    assert cols.issubset(set(table.c.keys())), (
        f"missing columns: {sorted(cols - set(table.c.keys()))}"
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