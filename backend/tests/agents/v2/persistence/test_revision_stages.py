"""P1 Task 1 RED — revision-owned stage repository.

Covers the Task 1 plan's Step 1 contract plus the P1 test strategy:

- ``initialize_stages`` creates the correct pending/skipped rows for
  FULL, CHAT_UPLOAD, and PARSE_ONLY profiles.
- ``required_stages_complete`` is False until every required stage is
  completed (or profile-authorized skipped), then True.
- Duplicate / redelivered transitions converge (idempotent running and
  completed marks; attempt counts only bump on pending -> running).
- Unknown stages fail closed.
- Terminal states (completed / skipped / failed) never transition
  further; skips outside the profile do not satisfy completion.
- Legacy revisions without stage rows fail closed (no inference).
- ``Document.*_done`` mirror flags never authorize completion.

All tests use the ``async_db`` fixture (SAVEPOINT-wrapped AsyncSession)
and ``document_factory`` (legacy ``documents`` rows seeded outside the
SAVEPOINT), mirroring ``test_document_revisions.py``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_revision_stage import DocumentRevisionStage
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
    InvalidStageTransition,
    UnknownRevisionStage,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
    compute_source_object_identity,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(async_db: AsyncSession) -> DocumentRevisionsRepository:
    return DocumentRevisionsRepository(async_db)


def _identity(key: str, sha: str) -> str:
    return compute_source_object_identity(
        bucket="hrag-uploads",
        object_key=key,
        version_id=None,
        etag="abc123",
        size_bytes=1024,
        content_sha256=sha,
    )


async def _allocate(repo: DocumentRevisionsRepository, document_id, key: str,
                    profile: RevisionBuildProfile = RevisionBuildProfile.FULL):
    return await repo.allocate_draft(
        document_id,
        _identity(key, "1" * 64),
        profile,
    )


async def _stages_by_name(repo, revision_id) -> dict[str, DocumentRevisionStage]:
    return {s.stage: s for s in await repo.get_stages(revision_id)}


# ---------------------------------------------------------------------------
# Plan Step 1 — FULL lifecycle through required_stages_complete
# ---------------------------------------------------------------------------


class TestFullProfileStageCompletion:
    @pytest.mark.asyncio
    async def test_full_lifecycle_requires_all_four_stages(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/full.pdf")

        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        assert await repo.required_stages_complete(revision.revision_id) is False

        await repo.mark_stage_completed(revision.revision_id, "parse")
        assert await repo.required_stages_complete(revision.revision_id) is False
        await repo.mark_stage_completed(revision.revision_id, "embed")
        await repo.mark_stage_completed(revision.revision_id, "caption")
        assert await repo.required_stages_complete(revision.revision_id) is False
        await repo.mark_stage_completed(revision.revision_id, "kg")
        assert await repo.required_stages_complete(revision.revision_id) is True

    @pytest.mark.asyncio
    async def test_full_initialize_creates_four_pending_rows(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/rows.pdf")

        rows = await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        assert {r.stage for r in rows} == {"parse", "embed", "caption", "kg"}
        assert {r.state for r in rows} == {"pending"}
        assert all(r.attempt_count == 0 for r in rows)


# ---------------------------------------------------------------------------
# Profile-specific pending / skipped rows
# ---------------------------------------------------------------------------


class TestProfileStageRows:
    @pytest.mark.asyncio
    async def test_chat_upload_skips_caption_and_kg(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(
            repo, document_id, "uploads/chat.pdf",
            profile=RevisionBuildProfile.CHAT_UPLOAD,
        )

        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.CHAT_UPLOAD
        )
        stages = await _stages_by_name(repo, revision.revision_id)
        assert stages["parse"].state == "pending"
        assert stages["embed"].state == "pending"
        assert stages["caption"].state == "skipped"
        assert stages["kg"].state == "skipped"

        await repo.mark_stage_completed(revision.revision_id, "parse")
        await repo.mark_stage_completed(revision.revision_id, "embed")
        assert await repo.required_stages_complete(revision.revision_id) is True

    @pytest.mark.asyncio
    async def test_parse_only_requires_just_parse(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(
            repo, document_id, "uploads/parse.pdf",
            profile=RevisionBuildProfile.PARSE_ONLY,
        )

        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.PARSE_ONLY
        )
        stages = await _stages_by_name(repo, revision.revision_id)
        assert stages["parse"].state == "pending"
        assert stages["embed"].state == "skipped"
        assert stages["caption"].state == "skipped"
        assert stages["kg"].state == "skipped"

        await repo.mark_stage_completed(revision.revision_id, "parse")
        assert await repo.required_stages_complete(revision.revision_id) is True

    @pytest.mark.asyncio
    async def test_full_profile_skip_does_not_satisfy_completion(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A skip outside the profile never counts as complete."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/skip.pdf")

        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        await repo.mark_stage_completed(revision.revision_id, "parse")
        await repo.mark_stage_completed(revision.revision_id, "embed")
        await repo.mark_stage_completed(revision.revision_id, "caption")
        await repo.mark_stage_skipped(revision.revision_id, "kg")
        assert await repo.required_stages_complete(revision.revision_id) is False


# ---------------------------------------------------------------------------
# Idempotent duplicate / redelivery transitions + attempt counts
# ---------------------------------------------------------------------------


class TestIdempotentTransitions:
    @pytest.mark.asyncio
    async def test_duplicate_running_and_completed_converge(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/dupe.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        first = await repo.mark_stage_running(revision.revision_id, "parse")
        assert first.state == "running"
        assert first.attempt_count == 1

        # Redelivered running mark converges without bumping attempts.
        second = await repo.mark_stage_running(revision.revision_id, "parse")
        assert second.state == "running"
        assert second.attempt_count == 1

        completed = await repo.mark_stage_completed(revision.revision_id, "parse")
        assert completed.state == "completed"
        redelivered = await repo.mark_stage_completed(
            revision.revision_id, "parse"
        )
        assert redelivered.state == "completed"
        assert redelivered.attempt_count == 1

    @pytest.mark.asyncio
    async def test_pending_to_completed_without_running_converges(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Out-of-order delivery (completed before running) still converges."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/ooo.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        row = await repo.mark_stage_completed(revision.revision_id, "embed")
        assert row.state == "completed"
        assert row.attempt_count == 0

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent_and_never_resets(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/init.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        await repo.mark_stage_completed(revision.revision_id, "parse")

        # A second initialize must not reset completed work.
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        stages = await _stages_by_name(repo, revision.revision_id)
        assert stages["parse"].state == "completed"
        assert len(stages) == 4

    @pytest.mark.asyncio
    async def test_terminal_states_never_transition(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/term.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        await repo.mark_stage_completed(revision.revision_id, "parse")
        await repo.mark_stage_failed(
            revision.revision_id, "embed", failure_class="timeout"
        )
        await repo.mark_stage_skipped(revision.revision_id, "caption")

        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_running(revision.revision_id, "parse")
        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_completed(revision.revision_id, "embed")
        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_running(revision.revision_id, "caption")
        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_skipped(revision.revision_id, "parse")

    @pytest.mark.asyncio
    async def test_failed_stage_records_failure_class(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/fail.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        row = await repo.mark_stage_failed(
            revision.revision_id, "kg", failure_class="dependency"
        )
        assert row.state == "failed"
        assert row.failure_class == "dependency"
        assert await repo.required_stages_complete(revision.revision_id) is False


# ---------------------------------------------------------------------------
# Fail-closed: unknown stages, missing rows, legacy revisions, mirrors
# ---------------------------------------------------------------------------


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_unknown_stages_fail_closed(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/unk.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        with pytest.raises(UnknownRevisionStage):
            await repo.mark_stage_running(revision.revision_id, "frobnicate")
        with pytest.raises(UnknownRevisionStage):
            await repo.mark_stage_completed(revision.revision_id, "frobnicate")
        with pytest.raises(UnknownRevisionStage):
            await repo.mark_stage_skipped(revision.revision_id, "frobnicate")
        with pytest.raises(UnknownRevisionStage):
            await repo.mark_stage_failed(revision.revision_id, "frobnicate")

    @pytest.mark.asyncio
    async def test_uninitialized_revision_fails_closed(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A revision with no stage rows is never complete (no inference)."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/leg.pdf")

        assert await repo.required_stages_complete(revision.revision_id) is False
        with pytest.raises(UnknownRevisionStage):
            await repo.mark_stage_completed(revision.revision_id, "parse")

    @pytest.mark.asyncio
    async def test_document_mirror_flags_never_authorize_completion(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Stale ``Document.*_done`` mirrors cannot satisfy the stage gate."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/mirror.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        await repo.session.execute(
            text(
                "UPDATE documents SET embed_done = true, captions_done = true, "
                "kg_done = true WHERE id = :did"
            ),
            {"did": str(document_id)},
        )
        assert await repo.required_stages_complete(revision.revision_id) is False


# ---------------------------------------------------------------------------
# P1 Task 1 fix round 1 (I1) — explicit running -> pending retry edge
# ---------------------------------------------------------------------------


class TestStageRetryEdge:
    @pytest.mark.asyncio
    async def test_retry_pending_preserves_attempts_across_two_retries(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``running -> pending`` preserves ``attempt_count``; the next
        ``pending -> running`` increments it (two full retry cycles)."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/retry.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        first = await repo.mark_stage_running(revision.revision_id, "parse")
        assert (first.state, first.attempt_count) == ("running", 1)

        pending = await repo.mark_stage_retry_pending(
            revision.revision_id, "parse"
        )
        assert pending.state == "pending"
        assert pending.attempt_count == 1
        assert pending.failure_class is None

        second = await repo.mark_stage_running(revision.revision_id, "parse")
        assert (second.state, second.attempt_count) == ("running", 2)

        await repo.mark_stage_retry_pending(revision.revision_id, "parse")
        third = await repo.mark_stage_running(revision.revision_id, "parse")
        assert (third.state, third.attempt_count) == ("running", 3)

        # The retried stage still completes normally afterwards.
        completed = await repo.mark_stage_completed(
            revision.revision_id, "parse"
        )
        assert completed.state == "completed"
        assert completed.attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_pending_from_terminal_or_pending_is_illegal(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Only ``running -> pending`` is allowed: ``completed``,
        ``skipped``, exhausted ``failed``, and never-ran ``pending`` all
        raise; unknown stages/rows fail closed."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/retrybad.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )
        await repo.mark_stage_completed(revision.revision_id, "parse")
        await repo.mark_stage_failed(
            revision.revision_id, "embed", failure_class="timeout"
        )
        await repo.mark_stage_skipped(revision.revision_id, "caption")

        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_retry_pending(revision.revision_id, "parse")
        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_retry_pending(revision.revision_id, "embed")
        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_retry_pending(
                revision.revision_id, "caption"
            )
        # Never-ran pending (attempt_count == 0) is not a retry.
        with pytest.raises(InvalidStageTransition):
            await repo.mark_stage_retry_pending(revision.revision_id, "kg")
        with pytest.raises(UnknownRevisionStage):
            await repo.mark_stage_retry_pending(
                revision.revision_id, "frobnicate"
            )

        # Terminal failure keeps its classification (no silent reset).
        stages = await _stages_by_name(repo, revision.revision_id)
        assert stages["embed"].failure_class == "timeout"
        assert stages["embed"].state == "failed"

    @pytest.mark.asyncio
    async def test_retry_edge_does_not_break_duplicate_running_convergence(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Redelivered ``running`` marks still converge without bumping,
        both before and after a genuine retry cycle."""
        document_id = document_factory()
        revision = await _allocate(repo, document_id, "uploads/retrydupe.pdf")
        await repo.initialize_stages(
            revision.revision_id, RevisionBuildProfile.FULL
        )

        await repo.mark_stage_running(revision.revision_id, "parse")
        dupe = await repo.mark_stage_running(revision.revision_id, "parse")
        assert (dupe.state, dupe.attempt_count) == ("running", 1)

        await repo.mark_stage_retry_pending(revision.revision_id, "parse")
        await repo.mark_stage_running(revision.revision_id, "parse")
        dupe_after_retry = await repo.mark_stage_running(
            revision.revision_id, "parse"
        )
        assert dupe_after_retry.state == "running"
        assert dupe_after_retry.attempt_count == 2
