"""Task 8 — governed evidence persistence (spec §15.2/§15.3/§24/§26).

Covers the Task 8 brief's Step 1 behaviours:

- null-safe ``EvidenceUse`` uniqueness including targetless uses;
- People evidence minimization (only task-required fields are stored);
- deterministic classification that a model cannot downgrade;
- AES-256-GCM encryption at rest with no plaintext column, fail-closed key
  management, and key rotation that keeps old records readable;
- expiry / ACL / tombstone / revision-mismatch / derived-validation hydration
  gates, and audited allow/deny reads.

All tests use the ``async_db`` SAVEPOINT-wrapped ``AsyncSession`` fixture, so
each test starts from a clean slate; the repositories only ``flush`` and the
outer transaction rolls back, so nothing is committed to the shared DB.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.evidence_record import EvidenceRecord as EvidenceRecordRow
from app.services.agents.v2.contracts.binding import ScopedDocument
from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
from app.services.agents.v2.contracts.evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceUse,
    EvidenceUseEnvelope,
    PeopleSourceIdentity,
    Provenance,
)
from app.services.agents.v2.contracts.locators import SectionLocator
from app.services.agents.v2.persistence.evidence import EvidenceRepository
from app.services.agents.v2.evidence_store.governance import (
    ENCRYPTION_ALGORITHM,
    EvidenceAccessAuditor,
    EvidenceAccessDecision,
    EvidenceAccessDenied,
    EvidenceDecryptionError,
    EvidenceGovernor,
    EvidenceHydrationRequest,
    EvidenceKeyUnavailable,
    EvidenceKeyring,
    EvidenceMinimizationError,
    EvidenceValidationError,
    classify_evidence,
    minimize_people_record,
)

KEY_1 = bytes([1]) * 32
KEY_2 = bytes([2]) * 32

RUN_ID = "run-1"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _RecordingAuditor:
    """In-test ``EvidenceAccessAuditor`` that keeps every decision."""

    def __init__(self) -> None:
        self.decisions: list[EvidenceAccessDecision] = []

    def record(self, decision: EvidenceAccessDecision) -> None:
        self.decisions.append(decision)

    @property
    def allowed(self) -> list[EvidenceAccessDecision]:
        return [d for d in self.decisions if d.allowed]

    @property
    def denied(self) -> list[EvidenceAccessDecision]:
        return [d for d in self.decisions if not d.allowed]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _provenance(fetcher: str = "document.read") -> Provenance:
    return Provenance(
        acquisition_id=uuid.uuid4(), fetcher=fetcher, fetched_at=_now()
    )


def _runtime(
    *,
    run_id: str = RUN_ID,
    workspace_ids: tuple[uuid.UUID, ...] = (),
    can_read_people: bool = True,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id=run_id,
        user_id=uuid.uuid4(),
        workspace_ids=workspace_ids,
        can_read_people=can_read_people,
        allowed_capabilities=frozenset({"document.read"}),
        deadline_at=_now() + timedelta(minutes=5),
    )


def _keyring(
    *,
    active_key_id: str | None = "k1",
    keys: dict[str, bytes] | None = None,
) -> EvidenceKeyring:
    return EvidenceKeyring(
        keys if keys is not None else {"k1": KEY_1, "k2": KEY_2},
        active_key_id,
    )


def _document_source(
    document_id: uuid.UUID, document_revision: str = "rev-1"
) -> DocumentSourceIdentity:
    return DocumentSourceIdentity(
        kind="document",
        document_id=document_id,
        document_revision=document_revision,
        locator=SectionLocator(kind="section", structure_node_id="n-1"),
    )


def _people_source(record_id: str = "p-1") -> PeopleSourceIdentity:
    return PeopleSourceIdentity(kind="people", record_id=record_id)


@pytest.fixture
def audit_log() -> _RecordingAuditor:
    return _RecordingAuditor()


@pytest.fixture
def make_governor(async_db: AsyncSession, audit_log: _RecordingAuditor):
    """Build a governor bound to the test session with an injectable keyring."""

    def _make(
        *,
        keyring: EvidenceKeyring | None = None,
        auditor: EvidenceAccessAuditor | None = None,
        clock=None,
    ) -> EvidenceGovernor:
        return EvidenceGovernor(
            async_db,
            keyring=keyring if keyring is not None else _keyring(),
            auditor=auditor if auditor is not None else audit_log,
            clock=clock,
        )

    return _make


async def _seed_document(
    session: AsyncSession, document_factory
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """Create a legacy document + one published revision row.

    Returns ``(document_id, revision_id, workspace_id)``. The workspace is read
    back from the authoritative ``documents`` row — the same place the
    governor's ACL check resolves it.
    """
    document_id = document_factory()
    workspace_id = await session.scalar(
        select(Document.workspace_id).where(Document.id == document_id)
    )
    revision_id = uuid.uuid4()
    session.add(
        DocumentRevision(
            revision_id=revision_id,
            document_id=document_id,
            generation=1,
            status="published",
            published_at=_now(),
        )
    )
    await session.flush()
    return document_id, revision_id, workspace_id


async def _append_use(
    session: AsyncSession,
    evidence_id: uuid.UUID,
    *,
    run_id: str = RUN_ID,
    purpose: str = "supporting",
    target_id: str | None = None,
) -> EvidenceUse:
    """Append one EvidenceUse and return the persisted use (with its UUID)."""
    use = EvidenceUse(
        use_id=uuid.uuid4(),
        evidence_id=evidence_id,
        task_id="t1",
        purpose=purpose,  # type: ignore[arg-type]
        target_id=target_id,
    )
    stored = await EvidenceRepository(session).append_use(
        EvidenceUseEnvelope(
            contract_version="2.0", run_id=run_id, use=use
        )
    )
    return use.model_copy(update={"use_id": stored})


async def _hydrate(
    governor: EvidenceGovernor,
    use: EvidenceUse,
    *,
    runtime: CapabilityRuntimeContext | None = None,
    required_binding: ScopedDocument | None = None,
    occurred_at: datetime | None = None,
) -> str:
    return await governor.hydrate_use(
        use_id=use.use_id,
        request=EvidenceHydrationRequest(
            runtime=runtime if runtime is not None else _runtime(),
            required_binding=required_binding,
            occurred_at=occurred_at,
        ),
    )


# ---------------------------------------------------------------------------
# Idempotent record + use persistence
# ---------------------------------------------------------------------------


class TestEvidencePersistence:
    @pytest.mark.asyncio
    async def test_record_insert_is_idempotent_by_source_and_content_hash(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        source = _people_source()
        first = await governor.persist_record(
            source=source, content="same-bytes", provenance=_provenance()
        )
        second = await governor.persist_record(
            source=source, content="same-bytes", provenance=_provenance()
        )
        assert first == second

        rows = await async_db.scalar(
            select(func.count())
            .select_from(EvidenceRecordRow)
            .where(EvidenceRecordRow.evidence_id == first)
        )
        assert rows == 1

        other = await governor.persist_record(
            source=source, content="different-bytes", provenance=_provenance()
        )
        assert other != first

        # A different source identity with the same content is a different record.
        other_source = await governor.persist_record(
            source=_people_source("p-2"),
            content="same-bytes",
            provenance=_provenance(),
        )
        assert other_source != first

    @pytest.mark.asyncio
    async def test_append_use_is_idempotent_and_null_safe_for_targetless_uses(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="p", provenance=_provenance()
        )
        repo = EvidenceRepository(async_db)

        first_use = EvidenceUse(
            use_id=uuid.uuid4(),
            evidence_id=evidence_id,
            task_id="t1",
            purpose="discovery",
            target_id=None,
        )
        first = await repo.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0", run_id=RUN_ID, use=first_use
            )
        )
        # A retry mints a fresh use_id; the five-column key must return the
        # EXISTING row's UUID instead of inserting a duplicate.
        retry = await repo.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0",
                run_id=RUN_ID,
                use=first_use.model_copy(update={"use_id": uuid.uuid4()}),
            )
        )
        assert retry == first == first_use.use_id

        count = await async_db.scalar(
            text(
                "SELECT count(*) FROM evidence_uses "
                "WHERE run_id = :r AND task_id = 't1' AND evidence_id = :e "
                "AND purpose = 'discovery' AND target_id IS NULL"
            ),
            {"r": RUN_ID, "e": evidence_id},
        )
        # NULLS NOT DISTINCT: the targetless retry collided instead of duplicating.
        assert count == 1

    @pytest.mark.asyncio
    async def test_append_use_distinguishes_purpose_and_target(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="p", provenance=_provenance()
        )
        repo = EvidenceRepository(async_db)

        discovery = await repo.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0",
                run_id=RUN_ID,
                use=EvidenceUse(
                    use_id=uuid.uuid4(),
                    evidence_id=evidence_id,
                    task_id="t1",
                    purpose="discovery",
                    target_id=None,
                ),
            )
        )
        coverage = await repo.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0",
                run_id=RUN_ID,
                use=EvidenceUse(
                    use_id=uuid.uuid4(),
                    evidence_id=evidence_id,
                    task_id="t1",
                    purpose="coverage",
                    target_id="target-1",
                ),
            )
        )
        supporting = await repo.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0",
                run_id=RUN_ID,
                use=EvidenceUse(
                    use_id=uuid.uuid4(),
                    evidence_id=evidence_id,
                    task_id="t1",
                    purpose="supporting",
                    target_id="target-1",
                ),
            )
        )
        assert len({discovery, coverage, supporting}) == 3

    @pytest.mark.asyncio
    async def test_append_use_scopes_uniqueness_to_the_run(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="p", provenance=_provenance()
        )
        repo = EvidenceRepository(async_db)
        use = EvidenceUse(
            use_id=uuid.uuid4(),
            evidence_id=evidence_id,
            task_id="t1",
            purpose="discovery",
            target_id=None,
        )
        first = await repo.append_use(
            EvidenceUseEnvelope(contract_version="2.0", run_id="run-a", use=use)
        )
        second = await repo.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0",
                run_id="run-b",
                use=use.model_copy(update={"use_id": uuid.uuid4()}),
            )
        )
        assert first != second

    @pytest.mark.asyncio
    async def test_load_use_round_trips_the_frozen_use_contract(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="p", provenance=_provenance()
        )
        use = await _append_use(
            async_db, evidence_id, purpose="coverage", target_id="target-9"
        )
        loaded = await EvidenceRepository(async_db).load_use(use.use_id)
        assert loaded is not None
        assert loaded.run_id == RUN_ID
        assert loaded.use == use


# ---------------------------------------------------------------------------
# Minimization + classification
# ---------------------------------------------------------------------------


class TestMinimizationAndClassification:
    @pytest.mark.asyncio
    async def test_people_minimization_stores_only_task_required_fields(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        raw = {
            "record_id": "p-1",
            "full_name": "Nguyen Van A",
            "salary": "1000000",
            "email": "a@example.com",
            "notes": "not required",
        }
        evidence_id = await governor.persist_people_evidence(
            record_id="p-1",
            raw_record=raw,
            required_fields=("full_name", "email"),
            provenance=_provenance("people.lookup"),
        )
        use = await _append_use(async_db, evidence_id)
        content = await _hydrate(governor, use)
        assert json.loads(content) == {
            "email": "a@example.com",
            "full_name": "Nguyen Van A",
        }
        assert "salary" not in content
        assert "notes" not in content

        # canonical, deterministic serialization
        assert content == json.dumps(
            {"email": "a@example.com", "full_name": "Nguyen Van A"},
            sort_keys=True,
            separators=(",", ":"),
        )

    @pytest.mark.asyncio
    async def test_people_minimization_rejects_a_missing_required_field(
        self, make_governor
    ):
        governor = make_governor()
        with pytest.raises(EvidenceMinimizationError):
            await governor.persist_people_evidence(
                record_id="p-1",
                raw_record={"full_name": "Nguyen Van A"},
                required_fields=("full_name", "email"),
                provenance=_provenance("people.lookup"),
            )

    def test_minimize_people_record_keeps_only_required_fields(self):
        minimized = minimize_people_record(
            {"full_name": "A", "salary": "1", "extra": "x"},
            required_fields=("full_name",),
        )
        assert minimized.field_names == ("full_name",)
        assert json.loads(minimized.content) == {"full_name": "A"}

    def test_classification_cannot_be_downgraded_by_a_model(self):
        people = _people_source()
        # A model-supplied "normal" can never downgrade People evidence.
        assert (
            classify_evidence(
                people, field_names=("full_name",), detected="normal"
            )
            == "personal"
        )
        # ... but a sensitive field, or an upgraded detector, raises it.
        assert (
            classify_evidence(people, field_names=("national_id",))
            == "sensitive_personal"
        )
        assert (
            classify_evidence(people, field_names=("full_name",), detected="sensitive_personal")
            == "sensitive_personal"
        )
        document = DocumentSourceIdentity(
            kind="document",
            document_id=uuid.uuid4(),
            document_revision="rev-1",
            locator=SectionLocator(kind="section", structure_node_id="n-1"),
        )
        assert classify_evidence(document, field_names=()) == "normal"
        assert (
            classify_evidence(document, field_names=(), detected="personal")
            == "personal"
        )

    @pytest.mark.asyncio
    async def test_people_evidence_is_classified_personal_on_persistence(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_people_evidence(
            record_id="p-1",
            raw_record={"full_name": "A", "salary": "1"},
            required_fields=("full_name",),
            provenance=_provenance("people.lookup"),
        )
        row = await async_db.get(EvidenceRecordRow, evidence_id)
        assert row.classification == "personal"
        assert row.expires_at is None


# ---------------------------------------------------------------------------
# Encryption at rest + key management
# ---------------------------------------------------------------------------


class TestEncryption:
    @pytest.mark.asyncio
    async def test_evidence_plaintext_is_encrypted_and_never_stored(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        secret = "SECRET-PLAINTEXT-VALUE"
        evidence_id = await governor.persist_record(
            source=_people_source(), content=secret, provenance=_provenance()
        )
        row = await async_db.get(EvidenceRecordRow, evidence_id)
        assert row.ciphertext != secret.encode()
        assert secret.encode() not in bytes(row.ciphertext)
        assert row.encryption_algorithm == ENCRYPTION_ALGORITHM == "AES-256-GCM"
        assert row.encryption_key_id == "k1"
        assert len(row.nonce) == 12
        # There is no plaintext column on the record at all.
        columns = set(EvidenceRecordRow.__table__.columns.keys())
        assert not ({"plaintext", "payload", "content", "raw"} & columns)

        use = await _append_use(async_db, evidence_id)
        assert await _hydrate(governor, use) == secret

    @pytest.mark.asyncio
    async def test_evidence_key_unavailable_fails_closed(
        self, make_governor, async_db: AsyncSession
    ):
        # 1) No keyring / no active key id: the write itself fails closed.
        empty = make_governor(keyring=_keyring(active_key_id=None, keys={}))
        with pytest.raises(EvidenceKeyUnavailable):
            await empty.persist_record(
                source=_people_source(),
                content="must-not-persist",
                provenance=_provenance(),
            )
        assert (
            await async_db.scalar(
                select(func.count()).select_from(EvidenceRecordRow)
            )
            == 0
        )

        # 2) An active key id missing from the ring also fails closed.
        missing_active = make_governor(
            keyring=_keyring(active_key_id="k9", keys={"k1": KEY_1})
        )
        with pytest.raises(EvidenceKeyUnavailable):
            await missing_active.persist_record(
                source=_people_source(),
                content="must-not-persist",
                provenance=_provenance(),
            )

        # 3) A record whose recorded key id is unavailable is never decrypted.
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="secret", provenance=_provenance()
        )
        use = await _append_use(async_db, evidence_id)
        unavailable = make_governor(
            keyring=_keyring(active_key_id="k1", keys={"k2": KEY_2})
        )
        with pytest.raises(EvidenceKeyUnavailable) as excinfo:
            await _hydrate(unavailable, use)
        assert "secret" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_evidence_wrong_key_does_not_leak_plaintext(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="secret-value", provenance=_provenance()
        )
        # Simulate a tampered / mismatched key id: the ring has key id "k2"
        # pointing at a different key, but the row claims it was written with
        # "k1" (whose key differs).
        await async_db.execute(
            update(EvidenceRecordRow)
            .where(EvidenceRecordRow.evidence_id == evidence_id)
            .values(encryption_key_id="k2")
        )
        use = await _append_use(async_db, evidence_id)
        wrong = make_governor(
            keyring=_keyring(active_key_id="k1", keys={"k1": KEY_1, "k2": KEY_2})
        )
        with pytest.raises(EvidenceDecryptionError) as excinfo:
            await _hydrate(wrong, use)
        assert "secret-value" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_evidence_key_rotation_keeps_old_records_readable(
        self, make_governor, async_db: AsyncSession
    ):
        # Phase 1: key id "k1" is active.
        before = make_governor(keyring=_keyring(active_key_id="k1"))
        old_id = await before.persist_record(
            source=_people_source("old"),
            content="old-secret",
            provenance=_provenance(),
        )
        old_use = await _append_use(async_db, old_id)
        old_row = await async_db.get(EvidenceRecordRow, old_id)
        assert old_row.encryption_key_id == "k1"

        # Phase 2: the keyring rotates — "k2" becomes active, "k1" is retained.
        after = make_governor(
            keyring=_keyring(active_key_id="k2", keys={"k1": KEY_1, "k2": KEY_2})
        )
        new_id = await after.persist_record(
            source=_people_source("new"),
            content="new-secret",
            provenance=_provenance(),
        )
        new_row = await async_db.get(EvidenceRecordRow, new_id)
        assert new_row.encryption_key_id == "k2"

        # Old evidence stays readable with its recorded key id...
        assert await _hydrate(after, old_use) == "old-secret"
        # ...and new evidence uses the new active key.
        new_use = await _append_use(async_db, new_id)
        assert await _hydrate(after, new_use) == "new-secret"

        # Dropping the retained key id makes the old record fail closed.
        rotated_away = make_governor(keyring=_keyring(active_key_id="k2", keys={"k2": KEY_2}))
        with pytest.raises(EvidenceKeyUnavailable):
            await _hydrate(rotated_away, old_use)

    def test_keyring_from_json_parses_the_keyring_contract(self):
        import base64

        ring = EvidenceKeyring.from_json(
            json.dumps({"k1": base64.b64encode(KEY_1).decode()}),
            "k1",
        )
        assert ring.key("k1") == KEY_1
        from app.services.agents.v2.evidence_store.governance import (
            EvidenceKeyUnavailable as _Unavailable,
        )

        with pytest.raises(_Unavailable):
            ring.key("missing")


# ---------------------------------------------------------------------------
# Hydration gate: expiry / ACL / tombstone / revision / derived / audit
# ---------------------------------------------------------------------------


class TestHydrationGate:
    @pytest.mark.asyncio
    async def test_expired_evidence_is_denied_and_unexpired_is_allowed(
        self, make_governor, async_db: AsyncSession, audit_log
    ):
        governor = make_governor()
        now = _now()
        expired_id = await governor.persist_record(
            source=_people_source("expired"),
            content="expired",
            provenance=_provenance(),
            expires_at=now - timedelta(minutes=1),
        )
        expired_use = await _append_use(async_db, expired_id)
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, expired_use)

        live_id = await governor.persist_record(
            source=_people_source("live"),
            content="live",
            provenance=_provenance(),
            expires_at=now + timedelta(hours=1),
        )
        live_use = await _append_use(async_db, live_id)
        assert await _hydrate(governor, live_use) == "live"

        forever_id = await governor.persist_record(
            source=_people_source("forever"),
            content="forever",
            provenance=_provenance(),
            expires_at=None,
        )
        forever_use = await _append_use(async_db, forever_id)
        assert await _hydrate(governor, forever_use) == "forever"

        reasons = {d.reason for d in audit_log.denied}
        assert "expired" in reasons

        # expires_at is persisted exactly as selected at insertion time.
        row = await async_db.get(EvidenceRecordRow, expired_id)
        assert row.expires_at is not None

    @pytest.mark.asyncio
    async def test_acl_authorizes_document_evidence_through_its_revision(
        self, make_governor, async_db: AsyncSession, document_factory, audit_log
    ):
        governor = make_governor()
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await governor.persist_record(
            source=_document_source(document_id),
            content="document body",
            provenance=_provenance(),
            revision_id=revision_id,
        )
        use = await _append_use(async_db, evidence_id)

        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(
                governor, use, runtime=_runtime(workspace_ids=(uuid.uuid4(),))
            )
        assert {"workspace_not_authorized"} <= {d.reason for d in audit_log.denied}

        allowed = await _hydrate(
            governor, use, runtime=_runtime(workspace_ids=(workspace_id,))
        )
        assert allowed == "document body"

    @pytest.mark.asyncio
    async def test_acl_denies_people_evidence_without_people_permission(
        self, make_governor, async_db: AsyncSession
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="person", provenance=_provenance()
        )
        use = await _append_use(async_db, evidence_id)
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, use, runtime=_runtime(can_read_people=False))
        assert (
            await _hydrate(
                governor, use, runtime=_runtime(can_read_people=True)
            )
            == "person"
        )

    @pytest.mark.asyncio
    async def test_tombstoned_evidence_record_is_denied(
        self, make_governor, async_db: AsyncSession, audit_log
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="gone", provenance=_provenance()
        )
        use = await _append_use(async_db, evidence_id)
        assert await _hydrate(governor, use) == "gone"

        await async_db.execute(
            update(EvidenceRecordRow)
            .where(EvidenceRecordRow.evidence_id == evidence_id)
            .values(payload_purged_at=_now())
        )
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, use)
        assert {"purged"} <= {d.reason for d in audit_log.denied}

    @pytest.mark.asyncio
    async def test_tombstoned_document_denies_hydration(
        self, make_governor, async_db: AsyncSession, document_factory, audit_log
    ):
        governor = make_governor()
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await governor.persist_record(
            source=_document_source(document_id),
            content="body",
            provenance=_provenance(),
            revision_id=revision_id,
        )
        use = await _append_use(async_db, evidence_id)
        runtime = _runtime(workspace_ids=(workspace_id,))
        assert await _hydrate(governor, use, runtime=runtime) == "body"

        await async_db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(source_deleted_at=_now())
        )
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, use, runtime=runtime)
        assert {"document_tombstoned"} <= {d.reason for d in audit_log.denied}

    @pytest.mark.asyncio
    async def test_revision_mismatch_denies_reuse(
        self, make_governor, async_db: AsyncSession, document_factory, audit_log
    ):
        governor = make_governor()
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await governor.persist_record(
            source=_document_source(document_id, "rev-1"),
            content="body",
            provenance=_provenance(),
            revision_id=revision_id,
        )
        use = await _append_use(
            async_db, evidence_id, purpose="coverage", target_id="target-1"
        )
        runtime = _runtime(workspace_ids=(workspace_id,))

        pinned_elsewhere = ScopedDocument(
            binding_id="b-1",
            document_id=document_id,
            document_revision="rev-2",
            role="target",
        )
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(
                governor, use, runtime=runtime, required_binding=pinned_elsewhere
            )
        assert {"revision_mismatch"} <= {d.reason for d in audit_log.denied}

        pinned_here = ScopedDocument(
            binding_id="b-1",
            document_id=document_id,
            document_revision="rev-1",
            role="target",
        )
        assert (
            await _hydrate(
                governor, use, runtime=runtime, required_binding=pinned_here
            )
            == "body"
        )

    @pytest.mark.asyncio
    async def test_document_evidence_requires_the_authoritative_revision_id(
        self, make_governor
    ):
        governor = make_governor()
        with pytest.raises(EvidenceValidationError):
            await governor.persist_record(
                source=_document_source(uuid.uuid4()),
                content="body",
                provenance=_provenance(),
                revision_id=None,
            )
        # A non-document source must not carry a copied revision id.
        with pytest.raises(EvidenceValidationError):
            await governor.persist_record(
                source=_people_source(),
                content="person",
                provenance=_provenance(),
                revision_id=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    async def test_record_source_carries_no_workspace_copy(
        self, make_governor, async_db: AsyncSession, document_factory
    ):
        governor = make_governor()
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        source = _document_source(document_id)
        evidence_id = await governor.persist_record(
            source=source,
            content="body",
            provenance=_provenance(),
            revision_id=revision_id,
        )
        row = await async_db.get(EvidenceRecordRow, evidence_id)
        assert row.source == source.model_dump(mode="json")
        assert "workspace" not in json.dumps(row.source)
        assert row.revision_id == revision_id

    @pytest.mark.asyncio
    async def test_derived_evidence_requires_validated_validation_state(
        self, make_governor, async_db: AsyncSession, audit_log
    ):
        governor = make_governor()
        source_id = await governor.persist_record(
            source=_people_source("source"),
            content="source",
            provenance=_provenance(),
        )

        # A derived record must declare its validation state at persistence.
        with pytest.raises(EvidenceValidationError):
            await governor.persist_record(
                source=DerivedSourceIdentity(
                    kind="derived", source_evidence_ids=(source_id,)
                ),
                content="summary",
                provenance=_provenance(),
                validation_state=None,
            )

        unvalidated_id = await governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(source_id,)
            ),
            content="summary-unvalidated",
            provenance=_provenance(),
            validation_state="unvalidated",
        )
        unvalidated_use = await _append_use(async_db, unvalidated_id)
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, unvalidated_use)

        failed_id = await governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(source_id,)
            ),
            content="summary-failed",
            provenance=_provenance(),
            validation_state="failed",
        )
        failed_use = await _append_use(async_db, failed_id)
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, failed_use)

        # Validated but unresolved lineage is still denied.
        orphan_id = await governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(uuid.uuid4(),)
            ),
            content="summary-orphan",
            provenance=_provenance(),
            validation_state="validated",
        )
        orphan_use = await _append_use(async_db, orphan_id)
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, orphan_use)

        # A validated record with resolved sources is hydratable.
        validated_id = await governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(source_id,)
            ),
            content="summary-validated",
            provenance=_provenance(),
            validation_state="validated",
        )
        validated_use = await _append_use(async_db, validated_id)
        assert await _hydrate(governor, validated_use) == "summary-validated"

        reasons = {d.reason for d in audit_log.denied}
        assert "derived_not_validated" in reasons
        assert "derived_source_unresolved" in reasons

    @pytest.mark.asyncio
    async def test_derived_evidence_rejects_unknown_validation_state(
        self, make_governor
    ):
        governor = make_governor()
        with pytest.raises(EvidenceValidationError):
            await governor.persist_record(
                source=DerivedSourceIdentity(
                    kind="derived", source_evidence_ids=(uuid.uuid4(),)
                ),
                content="summary",
                provenance=_provenance(),
                validation_state="probably-fine",
            )

    @pytest.mark.asyncio
    async def test_cross_run_use_is_rejected(
        self, make_governor, async_db: AsyncSession, audit_log
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="p", provenance=_provenance()
        )
        use = await _append_use(async_db, evidence_id, run_id="run-1")
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, use, runtime=_runtime(run_id="run-2"))
        assert {"cross_run"} <= {d.reason for d in audit_log.denied}

    @pytest.mark.asyncio
    async def test_discovery_use_is_not_hydratable(
        self, make_governor, async_db: AsyncSession, audit_log
    ):
        governor = make_governor()
        evidence_id = await governor.persist_record(
            source=_people_source(), content="p", provenance=_provenance()
        )
        use = await _append_use(
            async_db, evidence_id, purpose="discovery", target_id=None
        )
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(governor, use)
        assert {"purpose_not_hydratable"} <= {d.reason for d in audit_log.denied}

    @pytest.mark.asyncio
    async def test_unknown_use_is_denied_and_audited(
        self, make_governor, audit_log
    ):
        governor = make_governor()
        with pytest.raises(EvidenceAccessDenied):
            await governor.hydrate_use(
                use_id=uuid.uuid4(),
                request=EvidenceHydrationRequest(runtime=_runtime()),
            )
        assert {"unknown_use"} <= {d.reason for d in audit_log.denied}

    @pytest.mark.asyncio
    async def test_audited_allow_and_deny_reads(
        self, make_governor, async_db: AsyncSession, document_factory, audit_log
    ):
        governor = make_governor()
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await governor.persist_record(
            source=_document_source(document_id),
            content="body",
            provenance=_provenance(),
            revision_id=revision_id,
        )
        use = await _append_use(async_db, evidence_id)

        occurred_at = _now()
        assert (
            await _hydrate(
                governor,
                use,
                runtime=_runtime(workspace_ids=(workspace_id,)),
                occurred_at=occurred_at,
            )
            == "body"
        )
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(
                governor,
                use,
                runtime=_runtime(workspace_ids=(uuid.uuid4(),)),
                occurred_at=occurred_at,
            )

        allowed, denied = audit_log.allowed, audit_log.denied
        assert len(allowed) == 1 and len(denied) == 1
        assert allowed[0].use_id == use.use_id
        assert allowed[0].evidence_id == evidence_id
        assert allowed[0].run_id == RUN_ID
        assert allowed[0].reason == "allowed"
        assert allowed[0].occurred_at == occurred_at
        assert denied[0].allowed is False
        assert denied[0].reason == "workspace_not_authorized"
        assert denied[0].evidence_id == evidence_id

    @pytest.mark.asyncio
    async def test_hydration_is_not_authorized_by_copied_record_metadata(
        self, make_governor, async_db: AsyncSession, document_factory
    ):
        """The ACL workspace comes from the revision row, never a record copy."""
        governor = make_governor()
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        # Point the record at a revision that belongs to a DIFFERENT document:
        # the authoritative revision/document consistency check must fail.
        other_document_id, other_revision_id, _ = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await governor.persist_record(
            source=_document_source(document_id),
            content="body",
            provenance=_provenance(),
            revision_id=other_revision_id,
        )
        use = await _append_use(async_db, evidence_id)
        with pytest.raises(EvidenceAccessDenied):
            await _hydrate(
                governor, use, runtime=_runtime(workspace_ids=(workspace_id,))
            )
