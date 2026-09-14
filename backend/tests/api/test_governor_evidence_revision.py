"""Bounded fix: GovernorEvidenceBuilder must persist the authoritative revision.

Live root cause (commit 46009bd): hard-scoped retrieval reaches Chroma but
``DocumentRetrieveCapability`` returns INTERNAL_ERROR ``EvidenceValidationError``
because ``GovernorEvidenceBuilder.persist_use`` calls
``EvidenceGovernor.persist_record`` for ``DocumentSourceIdentity`` without the
required ``revision_id``.

Contract under test (production edit lives in ``runtime_selector.py`` only):

- document source → ``revision_id`` is the authoritative UUID parsed ONLY from
  ``source.document_revision`` (the string form of the revision UUID, per
  ``adapters/document.py``); no fallback to any current pointer/request value;
- invalid / non-UUID ``document_revision`` fails closed BEFORE persistence
  (``persist_record`` is never called);
- people and non-document sources pass no ``revision_id`` (behavior unchanged).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest


class _RecordingRepository:
    """In-test governed-use sink: keeps every appended envelope."""

    def __init__(self) -> None:
        self.envelopes: list = []

    async def append_use(self, envelope):
        self.envelopes.append(envelope)
        return envelope.use.use_id


class _RecordingGovernor:
    """Recording governor that enforces the REAL persistence invariant.

    Mirrors ``EvidenceGovernor._require_valid_governance_fields`` for the
    revision rule (document evidence REQUIRES a revision_id; every other
    kind MUST NOT carry one) so this unit fails with the exact missing
    revision metadata the live path hits, without needing Postgres.
    """

    def __init__(self) -> None:
        from app.services.agents.v2.evidence_store.governance import (
            EvidenceValidationError,
        )

        self._validation_error = EvidenceValidationError
        self.record_calls: list[dict] = []
        self.people_calls: list[dict] = []
        self.repository = _RecordingRepository()

    async def persist_record(self, *, source, content, provenance,
                             revision_id=None, **kwargs):
        from app.services.agents.v2.contracts.evidence import (
            DocumentSourceIdentity,
        )

        if isinstance(source, DocumentSourceIdentity):
            if revision_id is None:
                raise self._validation_error(
                    "document evidence must reference the authoritative "
                    "revision_id it was read from (workspace resolves through "
                    "the revision, never a copied allowlist)"
                )
        elif revision_id is not None:
            raise self._validation_error(
                f"{source.kind!r} evidence must not carry a document revision_id"
            )
        self.record_calls.append(
            {"source": source, "revision_id": revision_id}
        )
        return uuid4()

    async def persist_people_evidence(self, **kwargs):
        self.people_calls.append(kwargs)
        return uuid4()


def _provenance():
    from datetime import datetime, timezone

    from app.services.agents.v2.contracts.evidence import Provenance

    return Provenance(
        acquisition_id=uuid4(),
        fetcher="document.retrieve",
        fetched_at=datetime.now(timezone.utc),
    )


def _document_source(document_revision: str):
    from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity
    from app.services.agents.v2.contracts.locators import ChunkRangeLocator

    return DocumentSourceIdentity(
        kind="document",
        document_id=uuid4(),
        document_revision=document_revision,
        locator=ChunkRangeLocator(kind="chunk_range", start="c1", end="c1"),
    )


def _builder(governor, run_id: str = "run-revision-fix"):
    from app.services.agent.runtime_selector import GovernorEvidenceBuilder

    return GovernorEvidenceBuilder(governor, run_id=run_id)


def test_document_source_passes_authoritative_revision_uuid():
    """The builder parses ONLY ``source.document_revision`` into ``revision_id``."""
    revision_id = uuid4()
    governor = _RecordingGovernor()
    builder = _builder(governor, run_id="run-doc-rev")
    source = _document_source(str(revision_id))

    async def _main():
        return await builder.persist_use(
            source=source,
            content="revision-pinned chunk",
            provenance=_provenance(),
            task_id="task-1",
            purpose="coverage",
            target_id="t1",
        )

    ref = asyncio.run(_main())
    assert len(governor.record_calls) == 1
    recorded = governor.record_calls[0]
    assert recorded["revision_id"] == revision_id
    assert recorded["revision_id"] == UUID(source.document_revision)
    # The governed target-bound use is appended for this run.
    assert len(governor.repository.envelopes) == 1
    envelope = governor.repository.envelopes[0]
    assert envelope.run_id == "run-doc-rev"
    assert envelope.use.target_id == "t1"
    assert envelope.use.purpose == "coverage"
    assert ref.use_id == envelope.use.use_id


def test_invalid_document_revision_fails_closed_before_persistence():
    """A non-UUID revision never reaches ``persist_record`` (no fallback)."""
    from app.services.agents.v2.evidence_store.governance import (
        EvidenceValidationError,
    )

    governor = _RecordingGovernor()
    builder = _builder(governor)
    source = _document_source("rev-1")

    async def _main():
        return await builder.persist_use(
            source=source,
            content="revision-pinned chunk",
            provenance=_provenance(),
            task_id="task-1",
            purpose="coverage",
            target_id="t1",
        )

    with pytest.raises(EvidenceValidationError):
        asyncio.run(_main())
    assert governor.record_calls == []
    assert governor.repository.envelopes == []


def test_people_source_passes_no_revision():
    """People evidence keeps the minimized re-persist path with no revision."""
    import json

    from app.services.agents.v2.contracts.evidence import PeopleSourceIdentity

    governor = _RecordingGovernor()
    builder = _builder(governor)
    source = PeopleSourceIdentity(kind="people", record_id="p-1")

    async def _main():
        return await builder.persist_use(
            source=source,
            content=json.dumps({"name": "Nguyen Van A"}),
            provenance=_provenance(),
            task_id="task-1",
            purpose="supporting",
            target_id=None,
        )

    asyncio.run(_main())
    assert len(governor.people_calls) == 1
    assert governor.record_calls == []
    assert "revision_id" not in governor.people_calls[0]
    assert len(governor.repository.envelopes) == 1


def test_non_document_source_passes_no_revision():
    """Ownerless (memory) evidence persists with no revision carried."""
    from app.services.agents.v2.contracts.evidence import MemorySourceIdentity

    governor = _RecordingGovernor()
    builder = _builder(governor)
    source = MemorySourceIdentity(kind="memory", memory_id="m-1")

    async def _main():
        return await builder.persist_use(
            source=source,
            content="memory content",
            provenance=_provenance(),
            task_id="task-1",
            purpose="supporting",
            target_id=None,
        )

    asyncio.run(_main())
    assert len(governor.record_calls) == 1
    assert governor.record_calls[0]["revision_id"] is None
    assert len(governor.repository.envelopes) == 1


def test_builder_namespace_sanity():
    holder = SimpleNamespace(ok=True)
    assert holder.ok
