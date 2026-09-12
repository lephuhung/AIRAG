"""Reuse the v2 test-database fixtures for the API-level Task 5 tests.

The v2 schema bootstrap lives with the persistence suite (it owns the DSN,
the legacy-schema reset, and the SAVEPOINT session pattern). Importing the
fixtures here registers them for this directory instead of duplicating the
bootstrap.
"""

from tests.agents.v2.persistence.conftest import (  # noqa: F401
    _bootstrap_v2_schema,
    async_db,
    async_engine,
    document_factory,
    raw_connection,
    v2_sync_engine,
)
