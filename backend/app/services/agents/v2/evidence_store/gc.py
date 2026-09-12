"""Evidence *payload* retention GC — Phase 1D Task 9 (Predicate A).

Scope is ``evidence_records.ciphertext`` **only**. Evidence becomes eligible for
payload purge when

  - ``expires_at IS NOT NULL AND expires_at <= now()`` (the storage-policy
    deadline selected once at insertion, spec §15.3), AND
  - ``payload_purged_at IS NULL`` (not already purged), AND
  - no *active* (:mod:`~app.services.agents.v2.persistence.retention_leases`)
    lease references the evidence's use(s) or its revision — a resumable run
    may still hydrate the payload, AND
  - no legal hold blocks it.

Legal hold is intentionally a no-op in Phase 1: the v2 schema models no
legal-hold column/table (nothing in ``migrate.py`` or the ORM), so the predicate
is vacuously satisfied. If a hold model lands, add it to
:func:`_evidence_payload_candidate_query` — do not fork this module.

Action: clear ``ciphertext`` (empty bytes — the column is NOT NULL), set
``payload_purged_at``, and KEEP the row and its key-id/nonce/algorithm/identity
metadata for lineage. The purge is audited through an injectable sink (structured
logging by default), mirroring the audited access boundary in
:mod:`app.services.agents.v2.evidence_store.governance`.

This batcher NEVER deletes objects/vectors/KG/images/tables/chunks/markdown or
any ``DocumentRevision``; revision reclamation is a separate predicate
(:mod:`app.services.agents.v2.persistence.revision_gc`) with its own
eligibility.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evidence_record import EvidenceRecord
from app.services.agents.v2.persistence.retention_leases import (
    RevisionRetentionLeaseRepository,
)

logger = logging.getLogger(__name__)

#: Transaction-scoped advisory lock key for the evidence payload batcher,
#: distinct from the revision batcher and from the v2 migration lock.
EVIDENCE_GC_ADVISORY_LOCK_KEY: int = 0x4547_435F_4556_4944  # "EGC_EVID" ascii


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EvidencePayloadPurge:
    """One audited evidence payload purge."""

    evidence_id: uuid.UUID
    revision_id: Optional[uuid.UUID]
    purged_at: datetime
    reason: str


@runtime_checkable
class EvidencePayloadPurgeAuditor(Protocol):
    """Sink for every evidence payload purge."""

    def record(self, purge: EvidencePayloadPurge) -> None: ...


class LoggingEvidencePayloadPurgeAuditor:
    """Default auditor: structured ``logging`` records, never silent."""

    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)

    def record(self, purge: EvidencePayloadPurge) -> None:
        self._log.info(
            "evidence payload purged evidence_id=%s revision_id=%s reason=%s",
            purge.evidence_id,
            purge.revision_id,
            purge.reason,
        )


@dataclass(frozen=True)
class EvidencePayloadGcResult:
    """Outcome of one :func:`run_evidence_payload_gc_batch` invocation."""

    purged: int = 0
    purged_ids: tuple[uuid.UUID, ...] = ()
    #: True when another transaction held the batcher's advisory lock; the
    #: batch did nothing and the next scheduled run will retry.
    skipped_locked: bool = False


def _evidence_payload_candidate_query(
    lease_repository: RevisionRetentionLeaseRepository,
    *,
    now: datetime,
    batch_size: int,
):
    """The Predicate-A candidate query (``FOR UPDATE SKIP LOCKED``)."""
    return (
        select(EvidenceRecord.evidence_id, EvidenceRecord.revision_id)
        .where(
            EvidenceRecord.expires_at.is_not(None),
            EvidenceRecord.expires_at <= now,
            EvidenceRecord.payload_purged_at.is_(None),
            ~lease_repository.active_evidence_lease_exists(
                EvidenceRecord.evidence_id,
                EvidenceRecord.revision_id,
                now=now,
            ),
        )
        .order_by(EvidenceRecord.expires_at)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )


async def run_evidence_payload_gc_batch(
    session: AsyncSession,
    *,
    batch_size: int = 100,
    now: Optional[datetime] = None,
    auditor: Optional[EvidencePayloadPurgeAuditor] = None,
    lease_repository: Optional[RevisionRetentionLeaseRepository] = None,
) -> EvidencePayloadGcResult:
    """Purge one batch of expired evidence payloads. Mutates+flush only.

    The caller (the ``evidence_gc_worker`` one-shot) owns the transaction. The
    batch takes a transaction-scoped advisory lock and locks candidate rows
    ``FOR UPDATE SKIP LOCKED`` so two workers never purge the same payload and a
    row locked by another transaction is simply deferred to the next run.
    """
    ts = now or _now()
    # Re-entrant for the owning transaction; a competing transaction skips.
    locked = await session.scalar(
        text("SELECT pg_try_advisory_xact_lock(:key)"),
        {"key": EVIDENCE_GC_ADVISORY_LOCK_KEY},
    )
    if not locked:
        logger.info(
            "[evidence_gc] advisory lock held by another transaction — skipping"
        )
        return EvidencePayloadGcResult(skipped_locked=True)

    leases = lease_repository or RevisionRetentionLeaseRepository(session)
    rows = (
        await session.execute(
            _evidence_payload_candidate_query(
                leases, now=ts, batch_size=batch_size
            )
        )
    ).all()
    if not rows:
        return EvidencePayloadGcResult()

    evidence_ids = [row[0] for row in rows]
    revision_ids = {row[0]: row[1] for row in rows}
    # Clear ONLY the ciphertext; keep key-id/nonce/algorithm/identity metadata
    # for lineage. Guarded on ``payload_purged_at IS NULL`` so a concurrent
    # (already purged) row is not re-purged/audited twice.
    await session.execute(
        update(EvidenceRecord)
        .where(
            EvidenceRecord.evidence_id.in_(evidence_ids),
            EvidenceRecord.payload_purged_at.is_(None),
        )
        .values(ciphertext=b"", payload_purged_at=ts)
    )
    await session.flush()

    sink = auditor or LoggingEvidencePayloadPurgeAuditor()
    for evidence_id in evidence_ids:
        sink.record(
            EvidencePayloadPurge(
                evidence_id=evidence_id,
                revision_id=revision_ids[evidence_id],
                purged_at=ts,
                reason="expires_at_elapsed",
            )
        )
    return EvidencePayloadGcResult(
        purged=len(evidence_ids), purged_ids=tuple(evidence_ids)
    )
