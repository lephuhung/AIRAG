"""Session bootstrap for the v2 migration / model-metadata tests.

The R2 amendment changes the Release 1A shape (new ``published_at`` /
``superseded_by`` / artifact columns + a re-keyed attempt arbiter) under
the **same** ``V2_SCHEMA_VERSION = 1``. Because the DDL uses
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
    V2_SCHEMA_V1_TABLES,
    apply_v2_schema,
    make_engine,
)

V2_DSN = os.environ.get(
    "V2_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/hrag_test_v2",
)


def _drop_v2_state() -> None:
    """Drop v2 tables, legacy-column additions, and stable-pointer triggers."""
    with psycopg.connect(V2_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            # v2 tables first — CASCADE removes FKs pointing at them.
            for tbl in V2_SCHEMA_V1_TABLES:
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
            for stmt in (
                "ALTER TABLE documents DROP COLUMN IF EXISTS current_revision_id",
                "ALTER TABLE documents DROP COLUMN IF EXISTS source_deleted_at",
                "ALTER TABLE document_images DROP COLUMN IF EXISTS revision_id",
                "ALTER TABLE document_tables DROP COLUMN IF EXISTS revision_id",
            ):
                cur.execute(stmt)
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
