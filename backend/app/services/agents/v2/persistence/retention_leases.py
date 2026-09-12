"""Single owner of checkpoint/revision retention-lease SQL — Phase 1D Task 9.

The checkpoint lives in ``CHECKPOINT_DATABASE_URL`` (AsyncPostgresSaver) while
revisions, evidence, and leases live in the application DB. The two databases
never share a transaction, so the retention guarantee is expressed as a **safe
ordering**, not cross-DB atomicity:

    before emitting a graph state update that introduces or retains a revision
    pin:
        acquire_or_refresh(run_id, revision_id, evidence_use_id)   # app DB
        COMMIT the lease
        return the state update                                    # LangGraph then
                                                                   # writes the checkpoint

- lease commit succeeds, checkpoint fails → a harmless orphan lease that
  TTL/:meth:`RevisionRetentionLeaseRepository.sweep_expired` eventually releases;
- lease write fails → the node fails and the checkpoint containing the new pin
  MUST NOT be written;
- terminal ordering: the terminal state checkpoint succeeds first, then
  :meth:`release_run`; never release before the terminal checkpoint;
- interrupt/clarification: the resumable checkpoint keeps its lease active,
  refreshed on resume, released only on terminal completion/cancel or
  clarification expiry.

This module is the **only** implementation of ``revision_retention_leases`` SQL.
GC (``evidence_store/gc.py`` / ``persistence/revision_gc.py``), the supervisor,
the complex subgraph, and every checkpoint writer consume
:class:`RevisionRetentionLeaseRepository`; none duplicates lease SQL.

Like every v2 repository this one mutates and ``flush`` es only — the caller /
unit of work owns the transaction boundary (the node/worker commits the lease
BEFORE returning a checkpointable update).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import ColumnElement, exists, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.evidence_use import EvidenceUse
from app.models.revision_retention_lease import RevisionRetentionLease


def _now() -> datetime:
    """Timezone-aware ``now()`` — the project standard."""
    return datetime.now(timezone.utc)


def retention_lease_ttl() -> timedelta:
    """The configured lease TTL (``CHECKPOINT_RETENTION_LEASE_TTL_HOURS``)."""
    return timedelta(hours=settings.CHECKPOINT_RETENTION_LEASE_TTL_HOURS)


class RevisionRetentionLeaseRepository:
    """Read/write access to the revision retention leases.

    An *active* lease is ``released_at IS NULL AND expires_at > now()``. GC may
    reclaim a revision or evidence payload only when no active lease references
    it; an expired lease never blocks.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        ttl: Optional[timedelta] = None,
    ) -> None:
        self.session = session
        self.ttl = ttl if ttl is not None else retention_lease_ttl()

    # ------------------------------------------------------------------
    # Acquisition / release
    # ------------------------------------------------------------------

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Optional[uuid.UUID] = None,
        evidence_use_id: Optional[uuid.UUID] = None,
        *,
        now: Optional[datetime] = None,
    ) -> RevisionRetentionLease:
        """Acquire the run's lease on ``revision_id``, or refresh its expiry.

        Pass ``revision_id=None`` with an ``evidence_use_id`` for an
        evidence-only (targetless) lease: People/KG/memory evidence has no
        document revision to pin, so the lease anchors on the run + use
        identity alone. The null-safe unique
        ``uq_revision_lease_run_revision_use`` on
        ``(run_id, revision_id, evidence_use_id)`` is the arbiter in both
        cases: a re-delivered / resumed node converges on the SAME lease row
        (``lease_id`` preserved), extends ``expires_at`` to ``now + ttl``,
        and clears any release so the pin is active again.
        ``acquired_at`` is preserved — it records the first acquisition.

        At least one of ``revision_id`` / ``evidence_use_id`` should identify
        the retained artifact; a fully anonymous lease retains nothing and
        is only useful as a run keep-alive.

        The caller MUST commit after this call and before emitting the
        checkpointable state update (see the module docstring).
        """
        ts = now or _now()
        stmt = (
            pg_insert(RevisionRetentionLease)
            .values(
                lease_id=uuid.uuid4(),
                run_id=run_id,
                revision_id=revision_id,
                evidence_use_id=evidence_use_id,
                acquired_at=ts,
                expires_at=ts + self.ttl,
                released_at=None,
                release_reason=None,
            )
            .on_conflict_do_update(
                constraint="uq_revision_lease_run_revision_use",
                set_={
                    "expires_at": ts + self.ttl,
                    "released_at": None,
                    "release_reason": None,
                },
            )
            .returning(RevisionRetentionLease)
        )
        lease = (await self.session.execute(stmt)).scalar_one()
        await self.session.flush()
        # A refresh path updates the existing identity-mapped row; SQLAlchemy
        # returns that instance without re-populating already-loaded attributes,
        # so refresh explicitly or the caller would observe a stale expiry.
        await self.session.refresh(lease)
        return lease

    async def release_run(
        self,
        run_id: str,
        reason: str = "terminal",
        *,
        now: Optional[datetime] = None,
    ) -> int:
        """Release every active lease of ``run_id``. Returns the rowcount.

        Called only AFTER the terminal state checkpoint has been written (see
        the module docstring); never before.
        """
        ts = now or _now()
        result = await self.session.execute(
            update(RevisionRetentionLease)
            .where(
                RevisionRetentionLease.run_id == run_id,
                RevisionRetentionLease.released_at.is_(None),
            )
            .values(released_at=ts, release_reason=reason)
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def sweep_expired(self, *, now: Optional[datetime] = None) -> int:
        """Mark every expired active lease released. Returns the rowcount.

        Expiry already unblocks GC (the predicate requires
        ``expires_at > now()``); this is the hygiene sweep that also gives an
        orphan lease (lease commit succeeded, checkpoint failed) a terminal
        release reason.
        """
        ts = now or _now()
        result = await self.session.execute(
            update(RevisionRetentionLease)
            .where(
                RevisionRetentionLease.released_at.is_(None),
                RevisionRetentionLease.expires_at <= ts,
            )
            .values(released_at=ts, release_reason="expired")
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    # ------------------------------------------------------------------
    # Active-lease queries / SQL predicates
    # ------------------------------------------------------------------

    def active_revision_lease_exists(
        self,
        revision_id_column: ColumnElement,
        *,
        now: Optional[datetime] = None,
    ) -> ColumnElement[bool]:
        """Correlated ``EXISTS`` predicate: an active lease pins this revision.

        ``revision_id_column`` is an outer-query column (e.g.
        ``DocumentRevision.revision_id``); the clause is embedded in the GC
        candidate query so the lease SQL stays owned by this module.
        """
        ts = now or _now()
        return exists().where(
            RevisionRetentionLease.revision_id == revision_id_column,
            RevisionRetentionLease.released_at.is_(None),
            RevisionRetentionLease.expires_at > ts,
        )

    def active_evidence_lease_exists(
        self,
        evidence_id_column: ColumnElement,
        revision_id_column: ColumnElement,
        *,
        now: Optional[datetime] = None,
    ) -> ColumnElement[bool]:
        """Correlated ``EXISTS`` predicate: an active lease pins this evidence.

        A lease references either the evidence's ``revision_id`` or one of the
        evidence's ``evidence_use_id`` rows (a resumable run may still hydrate
        the payload), so both are checked.
        """
        ts = now or _now()
        use_ids = select(EvidenceUse.use_id).where(
            EvidenceUse.evidence_id == evidence_id_column
        )
        return exists().where(
            RevisionRetentionLease.released_at.is_(None),
            RevisionRetentionLease.expires_at > ts,
            or_(
                RevisionRetentionLease.revision_id == revision_id_column,
                RevisionRetentionLease.evidence_use_id.in_(use_ids),
            ),
        )

    async def has_active_revision_lease(
        self,
        revision_id: uuid.UUID,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        """True when an unexpired, unreleased lease pins ``revision_id``."""
        ts = now or _now()
        return bool(
            await self.session.scalar(
                select(
                    exists().where(
                        RevisionRetentionLease.revision_id == revision_id,
                        RevisionRetentionLease.released_at.is_(None),
                        RevisionRetentionLease.expires_at > ts,
                    )
                )
            )
        )

    async def has_active_evidence_lease(
        self,
        evidence_use_id: Optional[uuid.UUID] = None,
        revision_id: Optional[uuid.UUID] = None,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        """True when an unexpired, unreleased lease references the use/revision.

        ``evidence_use_id``/``revision_id`` are OR-ed: a lease on either one
        retains the evidence payload and blocks its purge.
        """
        ts = now or _now()
        conditions = [
            RevisionRetentionLease.released_at.is_(None),
            RevisionRetentionLease.expires_at > ts,
        ]
        references = []
        if evidence_use_id is not None:
            references.append(
                RevisionRetentionLease.evidence_use_id == evidence_use_id
            )
        if revision_id is not None:
            references.append(
                RevisionRetentionLease.revision_id == revision_id
            )
        if not references:
            return False
        return bool(
            await self.session.scalar(
                select(exists().where(*conditions, or_(*references)))
            )
        )
