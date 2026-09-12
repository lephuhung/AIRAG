"""Worker-pipeline test fixtures.

``tests/workers/test_revision_pipeline.py`` exercises the same Postgres v2
schema as Task 3's persistence suite, so this conftest re-exports that
suite's session bootstrap and per-test fixtures. Pytest registers fixtures
from a conftest module's namespace, so importing them here is sufficient
(and keeps a single authoritative bootstrap rather than a second copy).
"""

from tests.agents.v2.persistence.conftest import (  # noqa: F401
    _bootstrap_v2_schema,
    async_db,
    async_engine,
    document_factory,
    raw_connection,
    v2_sync_engine,
)
