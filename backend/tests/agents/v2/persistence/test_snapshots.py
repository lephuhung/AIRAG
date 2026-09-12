"""Tests for the Phase 1D v2 snapshot and binding-audit repositories.

These tests cover the Task 7 brief's Step 1 named behaviours:

- stale ``summary_version`` writers fail with a typed error;
- the raw ``ChatMessage`` row remains authoritative (the snapshot is a
  derived projection that never mutates chat history);
- incompatible persisted snapshot ``contract_version`` values are
  rejected, never migrated;
- binding audit is not needed to resolve hot-path revision policy;
- a successful CAS advances both ``summary_version`` and the monotonic
  ``built_through_message_id``;
- a stale writer cannot clobber a newer snapshot;
- ``ConversationSnapshot`` / ``SemanticSnapshot`` / ``BindingAuditRow``
  round-trip through the DB and stay valid frozen contracts
  (``model_validate`` succeeds).

All tests use the ``async_db`` SAVEPOINT-wrapped ``AsyncSession`` fixture
so each test starts with a clean slate; the repositories only ``flush``,
so nothing is ever committed to the shared test database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import Update, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.binding_audit import BindingAudit
from app.models.conversation_snapshot import ConversationSnapshot as ConversationSnapshotRow
from app.models.semantic_snapshot import SemanticSnapshot as SemanticSnapshotRow
from app.services.agents.v2.contracts.binding import (
    BindingAuditRow,
    DiscoveredBindingProvenance,
    PromotedBindingProvenance,
    UserBindingProvenance,
)
from app.services.agents.v2.contracts.conversation import (
    ActiveEntity,
    ConversationContext,
    ConversationSnapshot,
    ConversationTurn,
    EntityReference,
)
from app.services.agents.v2.contracts.semantic import (
    AbbreviationResolution,
    BlockingAmbiguity,
    DocumentReference,
    PinnedRevisionRequirement,
    SemanticContext,
    SemanticSnapshot,
)
from app.services.agents.v2.persistence.binding_audit import (
    BindingAuditRepository,
    IncompatibleBindingAuditVersion,
)
from app.services.agents.v2.persistence.snapshots import (
    BuiltThroughMessageRegression,
    BuiltThroughOrdinalRegression,
    ConversationSnapshotAlreadyExists,
    ConversationSnapshotRepository,
    IncompatibleSnapshotVersion,
    SemanticSnapshotRepository,
    StaleSummaryVersion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conversation_repo(async_db: AsyncSession) -> ConversationSnapshotRepository:
    return ConversationSnapshotRepository(async_db)


@pytest.fixture
def semantic_repo(async_db: AsyncSession) -> SemanticSnapshotRepository:
    return SemanticSnapshotRepository(async_db)


@pytest.fixture
def audit_repo(async_db: AsyncSession) -> BindingAuditRepository:
    return BindingAuditRepository(async_db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context(*, summary: str = "sum", turn: str = "hello") -> ConversationContext:
    return ConversationContext(
        summary=summary,
        active_entities=(ActiveEntity(ref_id="e1", kind="document", label="Nghị định 1"),),
        last_focus=EntityReference(ref_id="e1", kind="document", label="Nghị định 1"),
        recent_turns=(ConversationTurn(role="user", content=turn),),
    )


def _conversation_snapshot(
    thread_id: str,
    *,
    summary_version: int,
    built_through_message_id: str | None,
    summary: str = "sum",
) -> ConversationSnapshot:
    return ConversationSnapshot(
        contract_version="2.0",
        thread_id=thread_id,
        summary_version=summary_version,
        built_through_message_id=built_through_message_id,
        context=_context(summary=summary),
    )


def _semantic_context(
    *, revision_requirement=None, resolved_document_id: uuid.UUID | None = None
) -> SemanticContext:
    if resolved_document_id is None:
        resolved_document_id = uuid.uuid4()
    reference = DocumentReference(
        ref_id="ref-1",
        original_span="file thứ hai",
        normalized_reference="tep-2.pdf",
        requested_role="target",
        revision_requirement=revision_requirement,
        resolution_status="resolved",
        resolved_document_id=resolved_document_id,
    )
    return SemanticContext(
        contextualized_query="so sánh file thứ hai",
        normalized_query="so sanh tep-2.pdf",
        abbreviations=(AbbreviationResolution(abbreviation="UBND", expansion="Ủy ban nhân dân"),),
        coreferences=(),
        document_refs=(reference,),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(BlockingAmbiguity(ambiguity_id="amb-1", description="?") ,),
    )


async def _row_count(session: AsyncSession, table: str) -> int:
    return int(
        (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar()
    )


async def _stored_ordinal(session: AsyncSession, thread_id: str) -> int | None:
    """Read the persistence-only built-through ordinal straight from the row."""
    return await session.scalar(
        select(ConversationSnapshotRow.built_through_ordinal).where(
            ConversationSnapshotRow.thread_id == thread_id
        )
    )


async def _semantic_row_state(session: AsyncSession, thread_id: str):
    """Every persisted semantic-snapshot column, read as raw column values.

    A column-level ``SELECT`` deliberately bypasses the identity map so the
    assertion observes the database row, not a cached ORM instance.
    """
    return (
        await session.execute(
            select(
                SemanticSnapshotRow.snapshot_id,
                SemanticSnapshotRow.thread_id,
                SemanticSnapshotRow.contract_version,
                SemanticSnapshotRow.semantic,
                SemanticSnapshotRow.taken_at,
            ).where(SemanticSnapshotRow.thread_id == thread_id)
        )
    ).one()


# ---------------------------------------------------------------------------
# Step 1 — ConversationSnapshot CAS / monotonicity
# ---------------------------------------------------------------------------


class TestConversationSnapshotCAS:
    @pytest.mark.asyncio
    async def test_cas_success_advances_summary_version_and_built_through(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        first = await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m1"
            )
        )
        assert first.summary_version == 1

        advanced = await conversation_repo.cas_update(
            _conversation_snapshot(
                thread_id, summary_version=2, built_through_message_id="m2"
            ),
            expected_summary_version=1,
        )
        assert advanced.summary_version == 2
        assert advanced.built_through_message_id == "m2"

        loaded = await conversation_repo.load(thread_id)
        assert loaded is not None
        assert loaded.summary_version == 2
        assert loaded.built_through_message_id == "m2"

    @pytest.mark.asyncio
    async def test_stale_summary_version_fails(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m1"
            )
        )
        await conversation_repo.cas_update(
            _conversation_snapshot(
                thread_id, summary_version=2, built_through_message_id="m2"
            ),
            expected_summary_version=1,
        )

        with pytest.raises(StaleSummaryVersion):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=2, built_through_message_id="m2"
                ),
                expected_summary_version=1,
            )

    @pytest.mark.asyncio
    async def test_stale_writer_cannot_clobber_newer_snapshot(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m1"
            )
        )
        # Writer A wins the race from version 1 -> 2.
        await conversation_repo.cas_update(
            _conversation_snapshot(
                thread_id, summary_version=2, built_through_message_id="m2-A"
            ),
            expected_summary_version=1,
        )
        # Writer B still believes it holds version 1 and must lose.
        with pytest.raises(StaleSummaryVersion):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=2, built_through_message_id="m2-B"
                ),
                expected_summary_version=1,
            )
        loaded = await conversation_repo.load(thread_id)
        assert loaded is not None
        assert loaded.built_through_message_id == "m2-A"
        assert loaded.summary_version == 2

    @pytest.mark.asyncio
    async def test_built_through_message_id_cannot_regress_to_none(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m1"
            )
        )
        with pytest.raises(BuiltThroughMessageRegression):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=2, built_through_message_id=None
                ),
                expected_summary_version=1,
            )
        loaded = await conversation_repo.load(thread_id)
        assert loaded is not None
        assert loaded.summary_version == 1
        assert loaded.built_through_message_id == "m1"

    @pytest.mark.asyncio
    async def test_new_summary_version_must_advance(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=3, built_through_message_id="m3"
            )
        )
        with pytest.raises(ValueError):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=3, built_through_message_id="m3"
                ),
                expected_summary_version=3,
            )

    @pytest.mark.asyncio
    async def test_duplicate_first_snapshot_is_rejected(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m1"
            )
        )
        with pytest.raises(ConversationSnapshotAlreadyExists):
            await conversation_repo.save_first(
                _conversation_snapshot(
                    thread_id, summary_version=1, built_through_message_id="m1"
                )
            )

    @pytest.mark.asyncio
    async def test_round_trip_preserves_frozen_contract(
        self, conversation_repo: ConversationSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        original = _conversation_snapshot(
            thread_id, summary_version=4, built_through_message_id="m9"
        )
        await conversation_repo.save_first(original)
        loaded = await conversation_repo.load(thread_id)
        assert loaded == original
        # The loaded DB projection must itself re-validate as the frozen contract.
        assert (
            ConversationSnapshot.model_validate(
                loaded.model_dump(mode="json"), strict=False
            )
            == original
        )

    @pytest.mark.asyncio
    async def test_incompatible_contract_version_is_rejected_not_migrated(
        self, conversation_repo: ConversationSnapshotRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        async_db.add(
            ConversationSnapshotRow(
                snapshot_id=uuid.uuid4(),
                thread_id=thread_id,
                contract_version="1.0",
                summary_version=1,
                built_through_message_id="m1",
                context=_context().model_dump(mode="json"),
            )
        )
        await async_db.flush()
        with pytest.raises(IncompatibleSnapshotVersion):
            await conversation_repo.load(thread_id)
        # The rejected row is untouched — never best-effort migrated.
        stored = await async_db.scalar(
            select(ConversationSnapshotRow.contract_version).where(
                ConversationSnapshotRow.thread_id == thread_id
            )
        )
        assert stored == "1.0"


class TestBuiltThroughOrdinalMonotonicity:
    """Finding #2: the monotonic built-through pointer is enforced on the
    persistence-only ``built_through_ordinal`` column.

    ``built_through_ordinal`` is not part of the frozen contract: the caller
    (the rolling-summary writer) derives it from the **authoritative** raw
    ``chat_messages`` rows and supplies it on every write. The repository
    refuses any write that would move the pointer backwards or clear it.
    """

    @pytest.mark.asyncio
    async def test_forward_ordinal_move_is_allowed(
        self, conversation_repo: ConversationSnapshotRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m10"
            ),
            built_through_ordinal=10,
        )
        assert await _stored_ordinal(async_db, thread_id) == 10

        await conversation_repo.cas_update(
            _conversation_snapshot(
                thread_id, summary_version=2, built_through_message_id="m11"
            ),
            expected_summary_version=1,
            built_through_ordinal=11,
        )
        assert await _stored_ordinal(async_db, thread_id) == 11

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rewind_to", [9, 10])
    async def test_same_or_earlier_ordinal_rewind_is_rejected(
        self,
        conversation_repo: ConversationSnapshotRepository,
        async_db: AsyncSession,
        rewind_to: int,
    ):
        """A matching-version writer cannot rewind the built-through pointer."""
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m10"
            ),
            built_through_ordinal=10,
        )

        with pytest.raises(BuiltThroughOrdinalRegression):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=2, built_through_message_id="m-rewind"
                ),
                expected_summary_version=1,
                built_through_ordinal=rewind_to,
            )

        # The refused write changed nothing.
        assert await _stored_ordinal(async_db, thread_id) == 10
        stored_version = await async_db.scalar(
            select(ConversationSnapshotRow.summary_version).where(
                ConversationSnapshotRow.thread_id == thread_id
            )
        )
        assert stored_version == 1

    @pytest.mark.asyncio
    async def test_stored_null_ordinal_accepts_a_first_value(
        self, conversation_repo: ConversationSnapshotRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id=None
            )
        )
        assert await _stored_ordinal(async_db, thread_id) is None

        await conversation_repo.cas_update(
            _conversation_snapshot(
                thread_id, summary_version=2, built_through_message_id="m7"
            ),
            expected_summary_version=1,
            built_through_ordinal=7,
        )
        assert await _stored_ordinal(async_db, thread_id) == 7

    @pytest.mark.asyncio
    async def test_stored_ordinal_cannot_be_cleared(
        self, conversation_repo: ConversationSnapshotRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m5"
            ),
            built_through_ordinal=5,
        )

        with pytest.raises(BuiltThroughOrdinalRegression):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=2, built_through_message_id="m5"
                ),
                expected_summary_version=1,
                built_through_ordinal=None,
            )

        assert await _stored_ordinal(async_db, thread_id) == 5

    @pytest.mark.asyncio
    async def test_message_id_cannot_be_cleared_while_ordinal_advances(
        self, conversation_repo: ConversationSnapshotRepository, async_db: AsyncSession
    ):
        """The existing non-NULL -> NULL ``message_id`` rule is unchanged."""
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m6"
            ),
            built_through_ordinal=6,
        )

        with pytest.raises(BuiltThroughMessageRegression):
            await conversation_repo.cas_update(
                _conversation_snapshot(
                    thread_id, summary_version=2, built_through_message_id=None
                ),
                expected_summary_version=1,
                built_through_ordinal=7,
            )

        assert await _stored_ordinal(async_db, thread_id) == 6


class TestCasRowcountArbiter:
    """Finding #3: the conditional UPDATE's zero-row branch is the CAS arbiter.

    Every other stale test fails earlier in the Python pre-read, so only a test
    that interleaves a *committed* advance between the pre-read and the UPDATE
    exercises the ``rowcount == 0`` branch.
    """

    @pytest.mark.asyncio
    async def test_zero_row_update_raises_stale_summary_version(
        self,
        conversation_repo: ConversationSnapshotRepository,
        async_db: AsyncSession,
        raw_connection,
        monkeypatch: pytest.MonkeyPatch,
    ):
        thread_id = f"t-{uuid.uuid4()}"
        # A *committed* row, so a second (autocommit) writer can advance it
        # while the repository's SAVEPOINT transaction is still open.
        with raw_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_snapshots ("
                "snapshot_id, thread_id, contract_version, summary_version, "
                "built_through_message_id, built_through_ordinal, context) "
                "VALUES (%s, %s, %s, %s, %s, %s, '{}'::jsonb)",
                (str(uuid.uuid4()), thread_id, "2.0", 1, "m1", 1),
            )

        try:
            real_execute = async_db.execute

            async def execute_with_interleaved_advance(stmt, *args, **kwargs):
                # Interleave a committed advance exactly between the
                # repository's pre-read and its conditional UPDATE.
                if (
                    isinstance(stmt, Update)
                    and stmt.table.name == "conversation_snapshots"
                ):
                    with raw_connection.cursor() as cur:
                        cur.execute(
                            "UPDATE conversation_snapshots SET "
                            "summary_version = 2, "
                            "built_through_message_id = 'winner', "
                            "built_through_ordinal = 2 "
                            "WHERE thread_id = %s",
                            (thread_id,),
                        )
                return await real_execute(stmt, *args, **kwargs)

            monkeypatch.setattr(
                async_db, "execute", execute_with_interleaved_advance
            )

            with pytest.raises(StaleSummaryVersion):
                await conversation_repo.cas_update(
                    _conversation_snapshot(
                        thread_id,
                        summary_version=2,
                        built_through_message_id="m2",
                    ),
                    expected_summary_version=1,
                    built_through_ordinal=2,
                )

            # The interleaved winner's row is intact — the loser wrote nothing.
            stored = (
                await async_db.execute(
                    select(
                        ConversationSnapshotRow.summary_version,
                        ConversationSnapshotRow.built_through_message_id,
                        ConversationSnapshotRow.built_through_ordinal,
                    ).where(ConversationSnapshotRow.thread_id == thread_id)
                )
            ).one()
            assert tuple(stored) == (2, "winner", 2)
        finally:
            # Roll the SAVEPOINT transaction back first: if the repository ever
            # regressed and did apply the UPDATE, the test would otherwise
            # block forever on the row lock the async session still holds.
            await async_db.rollback()
            with raw_connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_snapshots WHERE thread_id = %s",
                    (thread_id,),
                )


# ---------------------------------------------------------------------------
# Step 1 — SemanticSnapshot persistence / version gate
# ---------------------------------------------------------------------------


class TestSemanticSnapshotPersistence:
    @pytest.mark.asyncio
    async def test_round_trip_preserves_frozen_contract(
        self, semantic_repo: SemanticSnapshotRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        original = SemanticSnapshot(contract_version="2.0", semantic=_semantic_context())
        await semantic_repo.save(thread_id, original)
        loaded = await semantic_repo.load(thread_id)
        assert loaded == original
        assert (
            SemanticSnapshot.model_validate(
                loaded.model_dump(mode="json"), strict=False
            )
            == original
        )

    @pytest.mark.asyncio
    async def test_save_is_idempotent_per_thread(
        self, semantic_repo: SemanticSnapshotRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await semantic_repo.save(
            thread_id, SemanticSnapshot(contract_version="2.0", semantic=_semantic_context())
        )
        await semantic_repo.save(
            thread_id,
            SemanticSnapshot(
                contract_version="2.0",
                semantic=_semantic_context().model_copy(update={"normalized_query": "x"}),
            ),
        )
        assert await _row_count(async_db, "semantic_snapshots") == 1
        loaded = await semantic_repo.load(thread_id)
        assert loaded is not None
        assert loaded.semantic.normalized_query == "x"

    @pytest.mark.asyncio
    async def test_incompatible_contract_version_is_rejected_not_migrated(
        self, semantic_repo: SemanticSnapshotRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        async_db.add(
            SemanticSnapshotRow(
                snapshot_id=uuid.uuid4(),
                thread_id=thread_id,
                contract_version="1.0",
                semantic=_semantic_context().model_dump(mode="json"),
            )
        )
        await async_db.flush()
        with pytest.raises(IncompatibleSnapshotVersion):
            await semantic_repo.load(thread_id)
        stored = await async_db.scalar(
            select(SemanticSnapshotRow.contract_version).where(
                SemanticSnapshotRow.thread_id == thread_id
            )
        )
        assert stored == "1.0"

    @pytest.mark.asyncio
    async def test_save_cannot_overwrite_a_foreign_version_row(
        self, semantic_repo: SemanticSnapshotRepository, async_db: AsyncSession
    ):
        """Finding #1: the semantic upsert must fail closed on a row that
        declares a contract version this code cannot load."""
        thread_id = f"t-{uuid.uuid4()}"
        foreign_version = "1.0"
        async_db.add(
            SemanticSnapshotRow(
                snapshot_id=uuid.uuid4(),
                thread_id=thread_id,
                contract_version=foreign_version,
                semantic=_semantic_context().model_dump(mode="json"),
            )
        )
        await async_db.flush()
        before = await _semantic_row_state(async_db, thread_id)

        with pytest.raises(IncompatibleSnapshotVersion):
            await semantic_repo.save(
                thread_id,
                SemanticSnapshot(
                    contract_version="2.0",
                    semantic=_semantic_context().model_copy(
                        update={"normalized_query": "replacement"}
                    ),
                ),
            )

        # Fail closed: the foreign row is byte-identical — same snapshot_id,
        # contract_version, payload, and taken_at. Never silently overwritten.
        after = await _semantic_row_state(async_db, thread_id)
        assert after == before
        assert after.contract_version == foreign_version
        assert after.semantic["normalized_query"] != "replacement"

    @pytest.mark.asyncio
    async def test_binding_audit_not_needed_to_resolve_hot_path_revision_policy(
        self, semantic_repo: SemanticSnapshotRepository, audit_repo: BindingAuditRepository, async_db: AsyncSession
    ):
        """Revision policy lives on the semantic reference, not in the audit trail."""
        thread_id = f"t-{uuid.uuid4()}"
        document_id = uuid.uuid4()
        snapshot = SemanticSnapshot(
            contract_version="2.0",
            semantic=_semantic_context(
                revision_requirement=PinnedRevisionRequirement(
                    kind="pinned", document_revision="rev-7"
                ),
                resolved_document_id=document_id,
            ),
        )
        await semantic_repo.save(thread_id, snapshot)
        # No audit rows were written to resolve the policy.
        assert await _row_count(async_db, "binding_audit") == 0

        loaded = await semantic_repo.load(thread_id)
        assert loaded is not None
        requirement = loaded.semantic.document_refs[0].revision_requirement
        assert requirement is not None
        assert requirement.kind == "pinned"
        assert requirement.document_revision == "rev-7"
        assert loaded.semantic.document_refs[0].resolved_document_id == document_id
        assert await _row_count(async_db, "binding_audit") == 0


# ---------------------------------------------------------------------------
# Step 1 — raw ChatMessage remains authoritative
# ---------------------------------------------------------------------------


class TestRawChatMessageAuthoritative:
    @pytest.mark.asyncio
    async def test_snapshot_is_a_projection_over_authoritative_raw_messages(
        self, conversation_repo: ConversationSnapshotRepository, raw_connection
    ):
        thread_id = f"t-{uuid.uuid4()}"
        session_id = uuid.uuid4()
        with raw_connection.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS chat_messages ("
                "id UUID PRIMARY KEY, session_id UUID, message_id TEXT, "
                "role TEXT NOT NULL, content TEXT NOT NULL, "
                "created_at TIMESTAMPTZ DEFAULT NOW())"
            )
            cur.execute(
                "INSERT INTO chat_messages (id, session_id, message_id, role, content) "
                "VALUES (%s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), str(session_id), "m1", "user", "câu hỏi đầu"),
            )
            cur.execute(
                "INSERT INTO chat_messages (id, session_id, message_id, role, content) "
                "VALUES (%s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), str(session_id), "m2", "assistant", "câu trả lời"),
            )
        try:
            await conversation_repo.save_first(
                _conversation_snapshot(
                    thread_id, summary_version=1, built_through_message_id="m2"
                )
            )
            # Raw truth advances: a newer message appears after the snapshot.
            with raw_connection.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (id, session_id, message_id, role, content) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), str(session_id), "m3", "user", "câu hỏi mới"),
                )

            # The snapshot did not touch the authoritative raw rows.
            with raw_connection.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM chat_messages WHERE session_id = %s",
                    (str(session_id),),
                )
                assert cur.fetchone()[0] == 3
                cur.execute(
                    "SELECT content FROM chat_messages WHERE session_id = %s AND message_id = 'm2'",
                    (str(session_id),),
                )
                assert cur.fetchone()[0] == "câu trả lời"

            # The projection still lags raw truth (it is derived, not authoritative).
            loaded = await conversation_repo.load(thread_id)
            assert loaded is not None
            assert loaded.built_through_message_id == "m2"
        finally:
            with raw_connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_messages WHERE session_id = %s",
                    (str(session_id),),
                )


# ---------------------------------------------------------------------------
# Step 1 — binding audit append / read / provenance round-trip
# ---------------------------------------------------------------------------


class TestBindingAudit:
    @pytest.mark.asyncio
    async def test_append_and_read_round_trip_preserves_provenance_variants(
        self, audit_repo: BindingAuditRepository
    ):
        thread_id = f"t-{uuid.uuid4()}"
        rows = (
            BindingAuditRow(
                contract_version="2.0",
                provenance=UserBindingProvenance(
                    kind="user_reference", binding_id="b1", source_ref_id="ref-1"
                ),
            ),
            BindingAuditRow(
                contract_version="2.0",
                provenance=DiscoveredBindingProvenance(
                    kind="discovered", binding_id="b2", source_task_id="task-9"
                ),
            ),
            BindingAuditRow(
                contract_version="2.0",
                provenance=PromotedBindingProvenance(
                    kind="promotion",
                    binding_id="b3",
                    source_binding_id="b2",
                    reason="user asked explicitly",
                ),
            ),
        )
        base = datetime(2026, 9, 11, tzinfo=timezone.utc)
        for index, row in enumerate(rows):
            await audit_repo.append(
                row, thread_id=thread_id, recorded_at=base + timedelta(seconds=index)
            )

        loaded = await audit_repo.read_for_thread(thread_id)
        assert loaded == rows

    @pytest.mark.asyncio
    async def test_append_is_ordered_by_recorded_at_and_not_idempotent(
        self, audit_repo: BindingAuditRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        row = BindingAuditRow(
            contract_version="2.0",
            provenance=UserBindingProvenance(
                kind="user_reference", binding_id="b1", source_ref_id="ref-1"
            ),
        )
        base = datetime(2026, 9, 11, tzinfo=timezone.utc)
        await audit_repo.append(row, thread_id=thread_id, recorded_at=base + timedelta(seconds=5))
        await audit_repo.append(row, thread_id=thread_id, recorded_at=base + timedelta(seconds=1))

        # Audit is an append-only trail: the same decision recorded twice
        # produces two rows (no dedup key exists in the schema).
        assert await _row_count(async_db, "binding_audit") == 2
        loaded = await audit_repo.read_for_thread(thread_id)
        assert len(loaded) == 2

        # Deterministic ordering by recorded_at (ascending).
        stamps = (
            await async_db.execute(
                select(BindingAudit.recorded_at)
                .where(BindingAudit.thread_id == thread_id)
                .order_by(BindingAudit.recorded_at)
            )
        ).scalars().all()
        assert list(stamps) == [base + timedelta(seconds=1), base + timedelta(seconds=5)]

    @pytest.mark.asyncio
    async def test_incompatible_contract_version_is_rejected_not_migrated(
        self, audit_repo: BindingAuditRepository, async_db: AsyncSession
    ):
        thread_id = f"t-{uuid.uuid4()}"
        async_db.add(
            BindingAudit(
                audit_id=uuid.uuid4(),
                thread_id=thread_id,
                contract_version="1.0",
                provenance_kind="user_reference",
                provenance={
                    "kind": "user_reference",
                    "binding_id": "b1",
                    "source_ref_id": "ref-1",
                },
            )
        )
        await async_db.flush()
        with pytest.raises(IncompatibleBindingAuditVersion):
            await audit_repo.read_for_thread(thread_id)
        stored = await async_db.scalar(
            select(BindingAudit.contract_version).where(
                BindingAudit.thread_id == thread_id
            )
        )
        assert stored == "1.0"


# ---------------------------------------------------------------------------
# Step 1 — Unit-of-Work boundary
# ---------------------------------------------------------------------------


class TestFlushOnly:
    @pytest.mark.asyncio
    async def test_repository_flushes_without_committing(
        self, conversation_repo: ConversationSnapshotRepository, async_db: AsyncSession, raw_connection
    ):
        thread_id = f"t-{uuid.uuid4()}"
        await conversation_repo.save_first(
            _conversation_snapshot(
                thread_id, summary_version=1, built_through_message_id="m1"
            )
        )
        # Visible inside the still-open unit of work ...
        assert await _row_count(async_db, "conversation_snapshots") == 1
        # ... but NOT visible to a separate autocommit connection, proving the
        # repository flushed rather than committed.
        with raw_connection.cursor() as cur:
            cur.execute("SELECT count(*) FROM conversation_snapshots")
            assert cur.fetchone()[0] == 0
