"""Bounded fix: real-Postgres proof for the builder revision repair.

The REAL ``GovernorEvidenceBuilder`` over the REAL ``EvidenceGovernor``
persists document evidence from ``source.document_revision`` (string form of
the revision UUID) and the stored row carries that authoritative
``revision_id``; hydration through the pinned binding then round-trips the
content. Uses the ``async_db`` SAVEPOINT session + ``document_factory`` (a
legacy document + one published revision row), so nothing is committed.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.evidence_record import EvidenceRecord as EvidenceRecordRow
from app.services.agent.runtime_selector import GovernorEvidenceBuilder
from app.services.agents.v2.contracts.binding import ScopedDocument
from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
from app.services.agents.v2.contracts.evidence import (
    DocumentSourceIdentity,
    Provenance,
)
from app.services.agents.v2.contracts.locators import ChunkRangeLocator
from app.services.agents.v2.evidence_store.governance import (
    EvidenceGovernor,
    EvidenceHydrationRequest,
    EvidenceKeyring,
)

KEY_1 = bytes([1]) * 32
RUN_ID = "run-builder-revision"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_builder_persists_document_evidence_with_authoritative_revision(
    async_db: AsyncSession, document_factory
):
    document_id = document_factory()
    workspace_id = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == document_id)
    )
    revision_id = uuid.uuid4()
    async_db.add(
        DocumentRevision(
            revision_id=revision_id,
            document_id=document_id,
            generation=1,
            status="published",
            published_at=_now(),
        )
    )
    await async_db.flush()

    governor = EvidenceGovernor(
        async_db, keyring=EvidenceKeyring({"k1": KEY_1}, "k1")
    )
    builder = GovernorEvidenceBuilder(governor, run_id=RUN_ID)
    ref = await builder.persist_use(
        source=DocumentSourceIdentity(
            kind="document",
            document_id=document_id,
            document_revision=str(revision_id),
            locator=ChunkRangeLocator(kind="chunk_range", start="c1", end="c1"),
        ),
        content="revision-pinned chunk",
        provenance=Provenance(
            acquisition_id=uuid.uuid4(),
            fetcher="document.retrieve",
            fetched_at=_now(),
        ),
        task_id="task-1",
        purpose="coverage",
        target_id="t1",
    )

    envelope = await governor.repository.load_use(ref.use_id)
    assert envelope is not None
    assert envelope.run_id == RUN_ID
    assert envelope.use.purpose == "coverage"
    assert envelope.use.target_id == "t1"
    row = await async_db.get(EvidenceRecordRow, envelope.use.evidence_id)
    assert row is not None
    assert row.revision_id == revision_id

    runtime = CapabilityRuntimeContext(
        request_id="req-1",
        run_id=RUN_ID,
        user_id=uuid.uuid4(),
        workspace_ids=(workspace_id,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.retrieve"}),
        deadline_at=_now(),
    )
    content = await governor.hydrate_use(
        use_id=ref.use_id,
        request=EvidenceHydrationRequest(
            runtime=runtime,
            required_binding=ScopedDocument(
                binding_id="b_t1",
                document_id=document_id,
                document_revision=str(revision_id),
                role="target",
            ),
        ),
    )
    assert content == "revision-pinned chunk"
