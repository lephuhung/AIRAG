"""Phase 1D Task 9 — evidence GC worker orchestration.

``run_both_batches`` must run Predicate A (evidence payloads) then Predicate B
(revision artifacts) each in its own transaction, so a failure in one does not
abort the other. The batchers and the session factory are injectable, so the
orchestration is tested without external artifact stores; one end-to-end test
drives the real evidence batcher against committed rows.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.services.agents.v2.evidence_store.gc import EvidencePayloadGcResult
from app.services.agents.v2.persistence.revision_gc import RevisionArtifactGcResult
from app.workers import evidence_gc_worker as worker

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _session_factory(async_engine):
    return async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )


@pytest.mark.asyncio
async def test_worker_runs_both_batches_in_separate_transactions(async_engine):
    session_factory = _session_factory(async_engine)
    seen: list[tuple[str, object, bool, int]] = []

    async def evidence_batcher(session, *, batch_size):
        seen.append(("evidence", session, session.in_transaction(), batch_size))
        return EvidencePayloadGcResult(purged=2)

    async def revision_batcher(session, *, batch_size):
        seen.append(("revision", session, session.in_transaction(), batch_size))
        return RevisionArtifactGcResult(reclaimed=3)

    summary = await worker.run_both_batches(
        batch_size=7,
        session_factory=session_factory,
        evidence_batcher=evidence_batcher,
        revision_batcher=revision_batcher,
    )

    assert [entry[0] for entry in seen] == ["evidence", "revision"]
    # Separate sessions => separate transactions.
    assert seen[0][1] is not seen[1][1]
    assert all(entry[2] for entry in seen)  # both ran inside a transaction
    assert all(entry[3] == 7 for entry in seen)  # batch size passed through
    assert summary.evidence == EvidencePayloadGcResult(purged=2)
    assert summary.revision == RevisionArtifactGcResult(reclaimed=3)
    assert summary.ok


@pytest.mark.asyncio
async def test_evidence_failure_does_not_abort_revision_batch(async_engine):
    session_factory = _session_factory(async_engine)
    revision_calls: list[int] = []

    async def evidence_batcher(session, *, batch_size):
        raise RuntimeError("evidence store down")

    async def revision_batcher(session, *, batch_size):
        revision_calls.append(batch_size)
        return RevisionArtifactGcResult(reclaimed=1)

    summary = await worker.run_both_batches(
        session_factory=session_factory,
        evidence_batcher=evidence_batcher,
        revision_batcher=revision_batcher,
    )

    assert revision_calls == [worker.DEFAULT_BATCH_SIZE]
    assert summary.evidence is None
    assert summary.revision == RevisionArtifactGcResult(reclaimed=1)
    assert not summary.ok
    assert summary.failures and summary.failures[0].startswith("evidence:")


@pytest.mark.asyncio
async def test_revision_failure_does_not_abort_evidence_batch(async_engine):
    session_factory = _session_factory(async_engine)
    evidence_calls: list[int] = []

    async def evidence_batcher(session, *, batch_size):
        evidence_calls.append(batch_size)
        return EvidencePayloadGcResult(purged=4)

    async def revision_batcher(session, *, batch_size):
        raise RuntimeError("neo4j down")

    summary = await worker.run_both_batches(
        session_factory=session_factory,
        evidence_batcher=evidence_batcher,
        revision_batcher=revision_batcher,
    )

    assert evidence_calls == [worker.DEFAULT_BATCH_SIZE]
    assert summary.evidence == EvidencePayloadGcResult(purged=4)
    assert summary.revision is None
    assert not summary.ok
    assert summary.failures and summary.failures[0].startswith("revision:")


@pytest.mark.asyncio
async def test_worker_purges_expired_evidence_with_the_real_batcher(
    async_engine, committed_evidence
):
    """End-to-end: the real Predicate-A batcher runs under the worker's own tx."""
    evidence_id = committed_evidence(expires_at=BASE - timedelta(hours=1))
    session_factory = _session_factory(async_engine)

    summary = await worker.run_both_batches(
        session_factory=session_factory, batch_size=10
    )

    assert summary.evidence is not None
    assert summary.evidence.purged >= 1
    assert evidence_id in summary.evidence.purged_ids


def test_once_flag_exits_zero_on_success(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return worker.GcRunSummary(
            evidence=EvidencePayloadGcResult(purged=1),
            revision=RevisionArtifactGcResult(reclaimed=0),
        )

    monkeypatch.setattr(worker, "run_both_batches", fake_run)

    assert worker.main(["--once", "--batch-size", "5"]) == 0
    assert captured == {"batch_size": 5}


def test_once_flag_exits_nonzero_when_a_batch_failed(monkeypatch):
    async def fake_run(**kwargs):
        return worker.GcRunSummary(failures=("evidence: boom",))

    monkeypatch.setattr(worker, "run_both_batches", fake_run)

    assert worker.main(["--once"]) == 1


def test_default_batch_size_is_100():
    assert worker.DEFAULT_BATCH_SIZE == 100
    args = worker._parse_args([])
    assert args.once is False
    assert args.batch_size == 100
    assert args.interval == worker.DEFAULT_INTERVAL_SECONDS


@pytest.fixture
def committed_evidence(raw_connection):
    """Insert committed evidence rows (no revision FK) visible to any session."""
    created: list[uuid.UUID] = []

    def _make(*, expires_at: datetime) -> uuid.UUID:
        evidence_id = uuid.uuid4()
        with raw_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO evidence_records ("
                "evidence_id, contract_version, ciphertext, encryption_key_id, "
                "nonce, encryption_algorithm, content_hash, classification, "
                "expires_at, revision_id, source, provenance, payload_purged_at"
                ") VALUES (%s, '2.0', %s, 'k1', %s, 'AES-256-GCM', %s, 'normal', "
                "%s, NULL, %s::jsonb, %s::jsonb, NULL)",
                (
                    str(evidence_id),
                    b"secret-payload",
                    b"0123456789ab",
                    uuid.uuid4().hex,
                    expires_at,
                    json.dumps({"kind": "document"}),
                    json.dumps({"fetcher": "document.read"}),
                ),
            )
        created.append(evidence_id)
        return evidence_id

    yield _make

    with raw_connection.cursor() as cur:
        for evidence_id in created:
            cur.execute(
                "DELETE FROM evidence_records WHERE evidence_id = %s",
                (str(evidence_id),),
            )
