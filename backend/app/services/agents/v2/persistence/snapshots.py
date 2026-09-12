"""Optimistic-locking persistence for the v2 conversation and semantic snapshots.

Spec §8.2 / §8.3 / §24: the raw chat history (``chat_messages``) is
**authoritative** for conversation truth; :class:`ConversationSnapshot` and
:class:`SemanticSnapshot` are derived projections persisted at the v2
checkpoint boundary. Rolling-summary persistence — not semantic context — owns
optimistic locking through ``summary_version`` and the monotonic
``built_through_message_id`` and the persistence-only
``built_through_ordinal``.

This module is the persistence owner for those projections. Like every v2
repository it only mutates and ``flush`` es; the caller / unit of work owns
the transaction boundary and must never be bypassed here.

Versioning (spec §3, §25): a persisted ``contract_version`` is a boundary
declaration, not a migration hint. A row carrying an unsupported version is
rejected (:class:`IncompatibleSnapshotVersion`) rather than best-effort
migrated.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_snapshot import (
    ConversationSnapshot as ConversationSnapshotRow,
)
from app.models.semantic_snapshot import SemanticSnapshot as SemanticSnapshotRow
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.conversation import ConversationSnapshot
from app.services.agents.v2.contracts.semantic import SemanticSnapshot
from app.services.agents.v2.contracts.validation import (
    validate_conversation_snapshot,
    validate_semantic_context,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SnapshotPersistenceError(Exception):
    """Base class for typed snapshot-persistence failures."""


class IncompatibleSnapshotVersion(SnapshotPersistenceError):
    """A persisted snapshot declares a ``contract_version`` this code does not
    support. The row is rejected, never migrated (spec §25)."""


class StaleSummaryVersion(SnapshotPersistenceError):
    """The writer's expected ``summary_version`` no longer matches the stored
    snapshot, so another writer advanced it first. The write is refused."""


class BuiltThroughMessageRegression(SnapshotPersistenceError):
    """A stale writer tried to clear ``built_through_message_id`` once it had
    advanced; the pointer is monotonic and can never move backwards to NULL."""


class BuiltThroughOrdinalRegression(SnapshotPersistenceError):
    """The caller-supplied ``built_through_ordinal`` would move the built-through
    pointer backwards or clear it; the pointer only ever advances."""


class SnapshotNotFound(SnapshotPersistenceError):
    """A CAS update named a thread with no persisted snapshot."""


class ConversationSnapshotAlreadyExists(SnapshotPersistenceError):
    """``save_first`` was called for a thread that already has a snapshot."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    """Timezone-aware ``now()`` — the project standard."""
    return datetime.now(timezone.utc)


def _require_supported_version(version: str, *, where: str) -> None:
    if version != CONTRACT_VERSION:
        raise IncompatibleSnapshotVersion(
            f"{where} declares contract_version {version!r}; "
            f"only {CONTRACT_VERSION!r} is supported and incompatible "
            "snapshots are rejected rather than migrated"
        )


# ---------------------------------------------------------------------------
# Conversation snapshots
# ---------------------------------------------------------------------------


class ConversationSnapshotRepository:
    """Persist the rolling-summary projection with optimistic locking.

    The repository holds exactly one current snapshot per ``thread_id``. The
    first write uses :meth:`save_first` (insert-arbitrated by the
    ``UNIQUE(thread_id)`` key); every subsequent write uses
    :meth:`cas_update`, whose conditional ``UPDATE ... WHERE thread_id =
    :t AND summary_version = :expected`` is the arbiter — a writer whose
    expectation lost the race affects zero rows and is refused.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def load(self, thread_id: str) -> Optional[ConversationSnapshot]:
        """Load and deserialize the thread's current snapshot, or ``None``."""
        row = await self.session.scalar(
            select(ConversationSnapshotRow).where(
                ConversationSnapshotRow.thread_id == thread_id
            )
        )
        if row is None:
            return None
        return self._deserialize(row)

    async def save_first(
        self,
        snapshot: ConversationSnapshot,
        *,
        built_through_ordinal: Optional[int] = None,
    ) -> ConversationSnapshot:
        """Insert the first snapshot for ``snapshot.thread_id``.

        ``built_through_ordinal`` is the persistence-only monotonic ordinal the
        caller (the rolling-summary writer) derives from the authoritative raw
        chat messages; it is not part of the frozen contract.

        Insert-arbitrated by ``UNIQUE(thread_id)``: when another writer already
        created the thread's snapshot the insert is a no-op and
        :class:`ConversationSnapshotAlreadyExists` is raised, so the first
        write can never silently clobber an established snapshot.
        """
        self._validate_for_write(snapshot)
        stmt = (
            pg_insert(ConversationSnapshotRow)
            .values(
                snapshot_id=uuid.uuid4(),
                thread_id=snapshot.thread_id,
                contract_version=snapshot.contract_version,
                summary_version=snapshot.summary_version,
                built_through_message_id=snapshot.built_through_message_id,
                built_through_ordinal=built_through_ordinal,
                context=snapshot.context.model_dump(mode="json"),
                taken_at=_now(),
            )
            .on_conflict_do_nothing(index_elements=["thread_id"])
            .returning(ConversationSnapshotRow.snapshot_id)
        )
        inserted = (await self.session.execute(stmt)).scalar_one_or_none()
        await self.session.flush()
        if inserted is None:
            raise ConversationSnapshotAlreadyExists(
                f"thread {snapshot.thread_id!r} already has a conversation snapshot"
            )
        return snapshot

    async def cas_update(
        self,
        snapshot: ConversationSnapshot,
        *,
        expected_summary_version: int,
        built_through_ordinal: Optional[int] = None,
    ) -> ConversationSnapshot:
        """Advance the thread's snapshot iff it is still at the expected version.

        The ``summary_version`` must strictly advance and both monotonic
        built-through pointers may only move forward: the
        ``built_through_message_id`` may never regress to ``NULL`` and
        ``built_through_ordinal`` must strictly exceed the stored ordinal (and
        may never be cleared once set). The caller — the rolling-summary
        writer — owns ordinal derivation from the authoritative raw chat
        messages, so the persisted ordinal is the ordering authority the opaque
        message id cannot provide.

        :raises StaleSummaryVersion: the stored version moved on (the CAS
          affected zero rows, or the stored version already differed).
        :raises SnapshotNotFound: no snapshot exists for the thread.
        :raises BuiltThroughOrdinalRegression: the new ordinal is not strictly
          greater than the stored ordinal, or it clears a stored ordinal.
        """
        self._validate_for_write(snapshot)
        if snapshot.summary_version <= expected_summary_version:
            raise ValueError(
                "summary_version must strictly advance: "
                f"new={snapshot.summary_version} "
                f"expected={expected_summary_version}"
            )
        current = await self.session.scalar(
            select(ConversationSnapshotRow).where(
                ConversationSnapshotRow.thread_id == snapshot.thread_id
            )
        )
        if current is None:
            raise SnapshotNotFound(
                f"thread {snapshot.thread_id!r} has no conversation snapshot"
            )
        # Fail closed on an incompatible persisted row instead of overwriting it.
        _require_supported_version(
            current.contract_version, where="conversation_snapshots"
        )
        if current.summary_version != expected_summary_version:
            raise StaleSummaryVersion(
                f"thread {snapshot.thread_id!r} is at summary_version "
                f"{current.summary_version}, not the expected "
                f"{expected_summary_version}"
            )
        if (
            current.built_through_message_id is not None
            and snapshot.built_through_message_id is None
        ):
            raise BuiltThroughMessageRegression(
                f"thread {snapshot.thread_id!r} already built through "
                f"{current.built_through_message_id!r}; it cannot be cleared"
            )
        if current.built_through_ordinal is not None:
            if built_through_ordinal is None:
                raise BuiltThroughOrdinalRegression(
                    f"thread {snapshot.thread_id!r} already built through "
                    f"ordinal {current.built_through_ordinal}; it cannot be "
                    "cleared"
                )
            if built_through_ordinal <= current.built_through_ordinal:
                raise BuiltThroughOrdinalRegression(
                    f"thread {snapshot.thread_id!r} already built through "
                    f"ordinal {current.built_through_ordinal}; the new ordinal "
                    f"{built_through_ordinal} must strictly advance"
                )
        stmt = (
            update(ConversationSnapshotRow)
            .where(
                ConversationSnapshotRow.thread_id == snapshot.thread_id,
                ConversationSnapshotRow.summary_version == expected_summary_version,
            )
            .values(
                contract_version=snapshot.contract_version,
                summary_version=snapshot.summary_version,
                built_through_message_id=snapshot.built_through_message_id,
                built_through_ordinal=built_through_ordinal,
                context=snapshot.context.model_dump(mode="json"),
                taken_at=_now(),
            )
            .returning(ConversationSnapshotRow.snapshot_id)
        )
        updated = (await self.session.execute(stmt)).scalar_one_or_none()
        if updated is None:
            # A concurrent writer advanced the version between our read and
            # this conditional UPDATE — the CAS is the arbiter.
            raise StaleSummaryVersion(
                f"thread {snapshot.thread_id!r} advanced past summary_version "
                f"{expected_summary_version} during the update"
            )
        await self.session.flush()
        return snapshot

    @staticmethod
    def _validate_for_write(snapshot: ConversationSnapshot) -> None:
        _require_supported_version(
            snapshot.contract_version, where="ConversationSnapshot"
        )
        validate_conversation_snapshot(snapshot)

    @staticmethod
    def _deserialize(row: ConversationSnapshotRow) -> ConversationSnapshot:
        _require_supported_version(
            row.contract_version, where="conversation_snapshots"
        )
        # JSONB materializes tuples as JSON arrays, so a strict tuple check
        # would reject our own projection; strictness is still enforced by the
        # version gate and the typed columns.
        snapshot = ConversationSnapshot.model_validate(
            {
                "contract_version": row.contract_version,
                "thread_id": row.thread_id,
                "summary_version": row.summary_version,
                "built_through_message_id": row.built_through_message_id,
                "context": row.context,
            },
            strict=False,
        )
        validate_conversation_snapshot(snapshot)
        return snapshot


# ---------------------------------------------------------------------------
# Semantic snapshots
# ---------------------------------------------------------------------------


class SemanticSnapshotRepository:
    """Persist the finalized semantic projection, one current row per thread.

    ``SemanticSnapshot`` carries no ``thread_id`` (spec §8.3), so the caller
    supplies the persistence key. There is no version field to lock against;
    a save overwrites the thread's current semantic projection — but only one
    whose persisted ``contract_version`` this code can also load.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def load(self, thread_id: str) -> Optional[SemanticSnapshot]:
        """Load and deserialize the thread's current semantic snapshot, or ``None``."""
        row = await self.session.scalar(
            select(SemanticSnapshotRow).where(
                SemanticSnapshotRow.thread_id == thread_id
            )
        )
        if row is None:
            return None
        return self._deserialize(row)

    async def save(
        self, thread_id: str, snapshot: SemanticSnapshot
    ) -> SemanticSnapshot:
        """Upsert the thread's current semantic projection.

        Fail closed on an incompatible persisted row: the conflict update is
        gated on ``contract_version``, so a row this code would refuse to
        ``load`` is never silently overwritten. When the gate refuses the
        update the stored row is re-read and
        :class:`IncompatibleSnapshotVersion` is raised.
        """
        _require_supported_version(
            snapshot.contract_version, where="SemanticSnapshot"
        )
        validate_semantic_context(snapshot.semantic)
        payload = snapshot.semantic.model_dump(mode="json")
        stmt = (
            pg_insert(SemanticSnapshotRow)
            .values(
                snapshot_id=uuid.uuid4(),
                thread_id=thread_id,
                contract_version=snapshot.contract_version,
                semantic=payload,
                taken_at=_now(),
            )
            .on_conflict_do_update(
                index_elements=["thread_id"],
                set_={
                    "contract_version": snapshot.contract_version,
                    "semantic": payload,
                    "taken_at": _now(),
                },
                # Only overwrite a row whose declared version this code supports.
                where=(SemanticSnapshotRow.contract_version == CONTRACT_VERSION),
            )
            .returning(SemanticSnapshotRow.snapshot_id)
        )
        written = (await self.session.execute(stmt)).scalar_one_or_none()
        if written is None:
            # The insert conflicted and the gated update declined: the existing
            # row declares a version this code must not clobber.
            stored_version = await self.session.scalar(
                select(SemanticSnapshotRow.contract_version).where(
                    SemanticSnapshotRow.thread_id == thread_id
                )
            )
            if stored_version is None:
                raise SnapshotPersistenceError(
                    f"semantic_snapshots upsert for thread {thread_id!r} neither "
                    "inserted nor updated a row"
                )
            _require_supported_version(stored_version, where="semantic_snapshots")
            raise SnapshotPersistenceError(
                f"semantic_snapshots row for thread {thread_id!r} was not "
                "overwritten although it declares a supported contract_version"
            )
        await self.session.flush()
        return snapshot

    @staticmethod
    def _deserialize(row: SemanticSnapshotRow) -> SemanticSnapshot:
        _require_supported_version(row.contract_version, where="semantic_snapshots")
        snapshot = SemanticSnapshot.model_validate(
            {
                "contract_version": row.contract_version,
                "semantic": row.semantic,
            },
            strict=False,
        )
        validate_semantic_context(snapshot.semantic)
        return snapshot
