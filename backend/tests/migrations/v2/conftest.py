"""Session bootstrap for the v2 migration / model-metadata tests.

The R2 amendment changes the Release 1A shape (new ``published_at`` /
``superseded_by`` / artifact columns + a re-keyed attempt arbiter) under
the current ``V2_SCHEMA_VERSION``. Because the DDL uses
``CREATE TABLE IF NOT EXISTS`` and ``apply_v2_schema`` short-circuits on
the version row, a database that applied the pre-R2 schema is never
upgraded in place — it would keep the stale shape while the version row
reports version 1.

To keep the suite order-independent and to guarantee it exercises the
*current* shape, rebuild the v2 state once per session before any test
runs. ``check_v2_schema`` also fails closed on a stale shape (see
``_shape_errors`` in ``migrate.py``), so a real deploy that skipped the
rebuild is reported rather than silently accepted.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from app.services.agents.v2.persistence.migrate import (
    V2_SCHEMA_V3_TABLES,
    apply_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)


def _drop_v2_state() -> None:
    """Drop v2 tables, the legacy FKs, and the stable-pointer triggers.

    The migration-owned legacy **columns** (``documents.current_revision_id``
    / ``source_deleted_at``, ``document_images.revision_id``,
    ``document_tables.revision_id``) are deliberately left in place. Dropping
    and re-adding a column consumes a fresh Postgres attribute slot on every
    cycle and the 1600-column limit is reached after a few dozen suite runs;
    dropping only the FK constraints (which ``apply_v2_schema`` re-adds via
    ``ADD CONSTRAINT``) keeps this bootstrap idempotent and churn-free.
    ``apply_v2_schema``'s ``ADD COLUMN IF NOT EXISTS`` is then a no-op for the
    retained columns.
    """
    with psycopg.connect(V2_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            # v2 tables first — CASCADE removes FKs pointing at them.
            # Task 7A: the v3 set (V1 + rollout) so no residue survives.
            for tbl in V2_SCHEMA_V3_TABLES:
                cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
            for trigger, table in (
                ("trg_documents_revision_id_stable", "documents"),
                ("trg_document_images_revision_id_stable", "document_images"),
                ("trg_document_tables_revision_id_stable", "document_tables"),
            ):
                cur.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
            for fn in (
                "raise_documents_revision_id_loss()",
                "raise_document_images_revision_id_loss()",
                "raise_document_tables_revision_id_loss()",
            ):
                cur.execute(f"DROP FUNCTION IF EXISTS {fn}")
            # The migration re-adds these named constraints (it does not use
            # IF NOT EXISTS for ADD CONSTRAINT), so drop them explicitly.
            cur.execute(
                "ALTER TABLE documents DROP CONSTRAINT IF EXISTS "
                "fk_documents_current_revision"
            )
            cur.execute(
                "ALTER TABLE document_images DROP CONSTRAINT IF EXISTS "
                "fk_document_images_revision"
            )
            cur.execute(
                "ALTER TABLE document_tables DROP CONSTRAINT IF EXISTS "
                "fk_document_tables_revision"
            )
            # The retained revision-pointer columns may still hold pointers
            # into revisions that the DROP above removed; clear them so the
            # migration's ADD CONSTRAINT can be re-installed.
            cur.execute(
                "UPDATE documents SET current_revision_id = NULL, "
                "source_deleted_at = NULL"
            )
            cur.execute("UPDATE document_images SET revision_id = NULL")
            cur.execute("UPDATE document_tables SET revision_id = NULL")
        conn.commit()


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_v2_schema() -> None:
    """Rebuild the v2 schema from scratch once per session (current shape)."""
    _drop_v2_state()
    engine = make_engine(V2_DSN)
    try:
        apply_v2_schema(engine)
    finally:
        engine.dispose()
