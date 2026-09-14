"""Task 4 round 1 — concrete hydrator adapter over the live governed store.

These tests exercise ``evidence_store/hydration.py::GovernorEvidenceHydrator``
against a REAL ``EvidenceGovernor`` on the harness PostgreSQL (the
``async_db`` SAVEPOINT session + ``document_factory`` fixtures from this
directory's conftest), so revision/ACL/expiry/tombstone/derived-faithfulness
and People minimization are verified against production governance — not a
test double. The closing test runs the real node chain
(evaluate → synthesize → ground → finalizer) over scheduler-dispatched,
governor-persisted evidence.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.services.agents.v2.capabilities import EvidenceBuilder
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evaluation import CoverageObservation
from app.services.agents.v2.contracts.evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceSourceIdentity,
    EvidenceUse,
    EvidenceUseEnvelope,
    EvidenceUseRef,
    Provenance,
)
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator, SectionLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.synthesis import SynthesisRuntimeContext
from app.services.agents.v2.evidence_store.governance import (
    EvidenceGovernor,
    EvidenceKeyring,
)
from app.services.agents.v2.evidence_store.hydration import GovernorEvidenceHydrator
from app.services.agents.v2.execution.scheduler import TaskScheduler
from app.services.agents.v2.nodes.evaluate import (
    AnswerDraftChannel,
    build_coverage,
)
from app.services.agents.v2.nodes.evaluate import evaluate_node
from app.services.agents.v2.nodes.finalizer import finalizer_node
from app.services.agents.v2.nodes.grounding import ground_node
from app.services.agents.v2.nodes.synthesize import synthesize_node
from app.services.agents.v2.persistence.evidence import EvidenceRepository

KEY_1 = bytes([1]) * 32
RUN_ID = "run-hydrator-1"
USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _runtime(
    *,
    run_id: str = RUN_ID,
    workspace_ids: tuple[UUID, ...] = (),
    can_read_people: bool = True,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=workspace_ids,
        can_read_people=can_read_people,
        allowed_capabilities=frozenset({"document.read"}),
        deadline_at=_now() + timedelta(minutes=5),
    )


def _graph_runtime(
    adapter: GovernorEvidenceHydrator,
    capability_runtime: CapabilityRuntimeContext,
    *,
    leases: Any = None,
    channel: AnswerDraftChannel | None = None,
    registry: Any = None,
) -> GraphRuntimeContext:
    from app.services.agent.runtime_selector import PlanBindingResolver

    return GraphRuntimeContext(
        capability_runtime=capability_runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=adapter,
            answer_draft_channel=channel,
            # Production ingress wires the request-scoped resolver the
            # shared scheduler feeds before dispatch (round 2 N1:
            # targeted read plans raise without one).
            pinned_target_resolver=PlanBindingResolver(),
        ),
    )


def _governor(session: AsyncSession) -> EvidenceGovernor:
    return EvidenceGovernor(session, keyring=EvidenceKeyring({"k1": KEY_1}, "k1"))


def _provenance(fetcher: str = "document.read") -> Provenance:
    return Provenance(
        acquisition_id=uuid.uuid4(), fetcher=fetcher, fetched_at=_now()
    )


def _semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query="Điều 5 của A nói gì?",
        normalized_query="Điều 5 của A nói gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _plan(task_id: str = "T-doc") -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="p-live",
        goal="Đọc Điều 5 của A",
        target_units=(
            TargetUnit(
                target_id="t1",
                binding_id="b1",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            ),
        ),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability="document.read",
                task_objective="Đọc Điều 5 của A",
                input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def _bindings(document_id: UUID, revision: str) -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b1",
                document_id=document_id,
                document_revision=revision,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def _budget() -> SynthesisRuntimeContext:
    return SynthesisRuntimeContext(
        max_evidence_items=10, max_total_chars=100_000, max_total_tokens=25_000
    )


async def _seed_document(
    session: AsyncSession, document_factory: Any
) -> tuple[UUID, UUID, UUID]:
    """Legacy document + one published revision; returns (doc, revision, ws)."""
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
    assert workspace_id is not None
    return document_id, revision_id, workspace_id


async def _append_use(
    session: AsyncSession,
    evidence_id: UUID,
    *,
    run_id: str = RUN_ID,
    task_id: str = "T-doc",
    purpose: str = "coverage",
    target_id: str | None = "t1",
) -> EvidenceUse:
    use = EvidenceUse(
        use_id=uuid.uuid4(),
        evidence_id=evidence_id,
        task_id=task_id,
        purpose=purpose,  # type: ignore[arg-type]
        target_id=target_id,
    )
    await EvidenceRepository(session).append_use(
        EvidenceUseEnvelope(contract_version="2.0", run_id=run_id, use=use)
    )
    return use


async def _persist_doc(
    governor: EvidenceGovernor,
    document_id: UUID,
    revision_id: UUID,
    content: str = "Điều 5 quy định mức phạt.",
) -> UUID:
    return await governor.persist_record(
        source=DocumentSourceIdentity(
            kind="document",
            document_id=document_id,
            document_revision=str(revision_id),
            locator=DocumentLocator(kind="document"),
        ),
        content=content,
        provenance=_provenance(),
        revision_id=revision_id,
    )


class TestGovernorHydratorAdmission:
    @pytest.mark.asyncio
    async def test_admits_document_use_with_locator_role_revision(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await _persist_doc(governor, document_id, revision_id)
        use = await _append_use(async_db, evidence_id)
        plan, bindings = _plan(), _bindings(document_id, str(revision_id))
        runtime = _graph_runtime(
            adapter, _runtime(workspace_ids=(workspace_id,))
        )
        admitted = await adapter.hydrate_for_evaluation(
            (EvidenceUseRef(use_id=use.use_id),),
            runtime=runtime,
            plan=plan,
            bindings=bindings,
        )
        assert len(admitted) == 1
        item = admitted[0]
        assert item.content == "Điều 5 quy định mức phạt."
        assert item.locator == DocumentLocator(kind="document")
        assert item.role == "target"
        assert item.target_id == "t1"
        assert item.document_revision == str(revision_id)
        assert item.classification == "normal"

    @pytest.mark.asyncio
    async def test_denies_revision_mismatch(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await _persist_doc(governor, document_id, revision_id)
        use = await _append_use(async_db, evidence_id)
        plan = _plan()
        bindings = _bindings(document_id, str(uuid.uuid4()))
        runtime = _graph_runtime(
            adapter, _runtime(workspace_ids=(workspace_id,))
        )
        assert (
            await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=use.use_id),),
                runtime=runtime,
                plan=plan,
                bindings=bindings,
            )
        ) == ()

    @pytest.mark.asyncio
    async def test_denies_foreign_workspace(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, _ = await _seed_document(async_db, document_factory)
        evidence_id = await _persist_doc(governor, document_id, revision_id)
        use = await _append_use(async_db, evidence_id)
        runtime = _graph_runtime(
            adapter, _runtime(workspace_ids=(uuid.uuid4(),))
        )
        assert (
            await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=use.use_id),),
                runtime=runtime,
                plan=_plan(),
                bindings=_bindings(document_id, str(revision_id)),
            )
        ) == ()

    @pytest.mark.asyncio
    async def test_denies_expired_and_tombstoned(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        expired_id = await governor.persist_record(
            source=DocumentSourceIdentity(
                kind="document",
                document_id=document_id,
                document_revision=str(revision_id),
                locator=DocumentLocator(kind="document"),
            ),
            content="hết hạn",
            provenance=_provenance(),
            revision_id=revision_id,
            expires_at=_now() - timedelta(seconds=1),
        )
        live_id = await _persist_doc(
            governor, document_id, revision_id, content="còn hạn"
        )
        expired_use = await _append_use(async_db, expired_id)
        live_use = await _append_use(async_db, live_id)
        plan, bindings = _plan(), _bindings(document_id, str(revision_id))
        runtime = _graph_runtime(
            adapter, _runtime(workspace_ids=(workspace_id,))
        )
        assert (
            await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=expired_use.use_id),),
                runtime=runtime,
                plan=plan,
                bindings=bindings,
            )
        ) == ()
        assert (
            await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=live_use.use_id),),
                runtime=runtime,
                plan=plan,
                bindings=bindings,
            )
        ) != ()

        await async_db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(source_deleted_at=_now())
        )
        assert (
            await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=live_use.use_id),),
                runtime=runtime,
                plan=plan,
                bindings=bindings,
            )
        ) == ()

    @pytest.mark.asyncio
    async def test_excludes_discovery_from_synthesis(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        evidence_id = await _persist_doc(governor, document_id, revision_id)
        discovery = await _append_use(
            async_db, evidence_id, purpose="discovery", target_id=None
        )
        runtime = _graph_runtime(
            adapter, _runtime(workspace_ids=(workspace_id,))
        )
        assert (
            await adapter.hydrate_for_synthesis(
                (EvidenceUseRef(use_id=discovery.use_id),),
                runtime=runtime,
                plan=_plan(),
                bindings=_bindings(document_id, str(revision_id)),
                budget=_budget(),
            )
        ) == ()

    @pytest.mark.asyncio
    async def test_people_content_is_minimized(
        self, async_db: AsyncSession
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        evidence_id = await governor.persist_people_evidence(
            record_id="rec-1",
            raw_record={"name": "Nguyễn Văn A", "national_id": "000", "extra": 1},
            required_fields=("name",),
            provenance=_provenance("people.lookup"),
        )
        use = await _append_use(
            async_db,
            evidence_id,
            task_id="T-people",
            purpose="supporting",
            target_id=None,
        )
        from app.services.agents.v2.contracts.capability import PeopleLookupInput
        from app.services.agents.v2.contracts.planning import TaskSpec as _TaskSpec

        plan = TaskPlan(
            contract_version="2.0",
            plan_id="p-people",
            goal="Ai?",
            target_units=(),
            tasks=(
                _TaskSpec(
                    task_id="T-people",
                    capability="people.lookup",
                    task_objective="Ai?",
                    input=PeopleLookupInput(kind="people.lookup", query="A"),
                    depends_on=(),
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            ),
        )
        runtime = _graph_runtime(adapter, _runtime())
        admitted = await adapter.hydrate_for_evaluation(
            (EvidenceUseRef(use_id=use.use_id),),
            runtime=runtime,
            plan=plan,
            bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        )
        assert len(admitted) == 1
        import json

        assert set(json.loads(admitted[0].content)) == {"name"}


class TestGovernorHydratorDerived:
    @pytest.mark.asyncio
    async def test_validated_derived_admitted_unvalidated_denied(
        self, async_db: AsyncSession
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        from app.services.agents.v2.contracts.evidence import MemorySourceIdentity

        source_id = await governor.persist_record(
            source=MemorySourceIdentity(kind="memory", memory_id="m-1"),
            content="ghi nhớ",
            provenance=_provenance("memory.lookup"),
        )
        good_id = await governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(source_id,)
            ),
            content="tóm tắt",
            provenance=_provenance("synthesis.overflow"),
            validation_state="validated",
        )
        bad_id = await governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(source_id,)
            ),
            content="tóm tắt khác",
            provenance=_provenance("synthesis.overflow"),
            validation_state="unvalidated",
        )
        # Derived uses hydrate targetless: the governor denies a
        # target-bound non-document use at the revision gate, so lineage
        # summaries resolve through a targetless (search) task.
        from app.services.agents.v2.contracts.capability import (
            DocumentSearchInput,
        )
        from app.services.agents.v2.contracts.planning import (
            TaskSpec as _TaskSpec2,
        )

        plan = TaskPlan(
            contract_version="2.0",
            plan_id="p-derived",
            goal="Tìm và tóm tắt.",
            target_units=(),
            tasks=(
                _TaskSpec2(
                    task_id="T-search",
                    capability="document.search",
                    task_objective="Tìm.",
                    input=DocumentSearchInput(
                        kind="document.search", query="phạt"
                    ),
                    depends_on=(),
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            ),
        )
        good_use = await _append_use(
            async_db,
            good_id,
            task_id="T-search",
            purpose="supporting",
            target_id=None,
        )
        bad_use = await _append_use(
            async_db,
            bad_id,
            task_id="T-search",
            purpose="supporting",
            target_id=None,
        )
        bindings = _bindings(uuid.uuid4(), "rev-x")
        runtime = _graph_runtime(adapter, _runtime())
        assert [
            h.use_id
            for h in await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=good_use.use_id),),
                runtime=runtime,
                plan=plan,
                bindings=bindings,
            )
        ] == [good_use.use_id]
        assert (
            await adapter.hydrate_for_evaluation(
                (EvidenceUseRef(use_id=bad_use.use_id),),
                runtime=runtime,
                plan=plan,
                bindings=bindings,
            )
        ) == ()

    @pytest.mark.asyncio
    async def test_overflow_persists_derived_with_lineage_idempotently(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        e1 = await _persist_doc(governor, document_id, revision_id, content="A" * 10)
        e2 = await _persist_doc(governor, document_id, revision_id, content="B" * 10)
        u1 = await _append_use(async_db, e1)
        u2 = await _append_use(async_db, e2)
        plan, bindings = _plan(), _bindings(document_id, str(revision_id))
        runtime = _graph_runtime(
            adapter, _runtime(workspace_ids=(workspace_id,))
        )
        tight = SynthesisRuntimeContext(
            max_evidence_items=10, max_total_chars=10, max_total_tokens=10_000
        )
        refs = (EvidenceUseRef(use_id=u1.use_id), EvidenceUseRef(use_id=u2.use_id))
        first = await adapter.hydrate_for_synthesis(
            refs, runtime=runtime, plan=plan, bindings=bindings, budget=tight
        )
        assert len(first) == 2
        assert [h.use_id for h in first[:1]] == [u1.use_id]
        derived = first[1]
        assert derived.purpose == "supporting"
        row = await EvidenceRepository(async_db).load_record(derived.evidence_id)
        assert row is not None
        assert isinstance(row.source, DerivedSourceIdentity)
        assert row.source.source_evidence_ids == (e2,)
        assert row.validation_state == "validated"

        second = await adapter.hydrate_for_synthesis(
            refs, runtime=runtime, plan=plan, bindings=bindings, budget=tight
        )
        assert [h.use_id for h in second] == [h.use_id for h in first]

        # The derived supporting use never creates read coverage.
        results = (
            AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id="T-doc",
                status="success",
                data=None,
                evidence_uses=(EvidenceUseRef(use_id=derived.use_id),),
                coverage_observations=(
                    CoverageObservation(
                        target_id="t1",
                        observed_locators=(DocumentLocator(kind="document"),),
                        outcome="read",
                    ),
                ),
                error=None,
            ),
        )
        coverage = build_coverage(plan, bindings, results, (derived,))
        assert coverage.items[0].status == "missing"


class DbEvidenceBuilder:
    """Test EvidenceBuilder that persists real governed records + uses."""

    def __init__(
        self,
        governor: EvidenceGovernor,
        session: AsyncSession,
        *,
        run_id: str,
        revision_id: UUID,
    ) -> None:
        self._governor = governor
        self._session = session
        self._run_id = run_id
        self._revision_id = revision_id

    async def persist_use(
        self,
        *,
        source: EvidenceSourceIdentity,
        content: str,
        provenance: Provenance,
        task_id: str,
        purpose: Any,
        target_id: str | None,
    ) -> EvidenceUseRef:
        evidence_id = await self._governor.persist_record(
            source=source,
            content=content,
            provenance=provenance,
            revision_id=self._revision_id,
        )
        use = EvidenceUse(
            use_id=uuid.uuid4(),
            evidence_id=evidence_id,
            task_id=task_id,
            purpose=purpose,
            target_id=target_id,
        )
        stored = await EvidenceRepository(self._session).append_use(
            EvidenceUseEnvelope(
                contract_version="2.0", run_id=self._run_id, use=use
            )
        )
        return EvidenceUseRef(use_id=stored)


class _FakeLeaseSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class _FakeLeaseRepo:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.calls: list[tuple[Any, Any, Any]] = []
        self.session = _FakeLeaseSession(self.events)

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Any = None,
        evidence_use_id: Any = None,
        *,
        now: Any = None,
    ) -> Any:
        self.calls.append((run_id, revision_id, evidence_use_id))
        return None


class _DocReadCapability:
    """document.read stand-in that persists REAL governed evidence."""

    def __init__(
        self,
        builder: DbEvidenceBuilder,
        document_id: UUID,
        revision_id: UUID,
        content: str,
    ) -> None:
        from app.services.agents.v2.contracts.capability import CapabilityDescriptor

        self.descriptor = CapabilityDescriptor(
            name="document.read",
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=False,
        )
        self._builder = builder
        self._document_id = document_id
        self._revision_id = revision_id
        self._content = content

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        assert isinstance(request, AgentRequest)
        assert isinstance(runtime, CapabilityRuntimeContext)
        target_id = request.input.target_ids[0]  # type: ignore[union-attr]
        # The plan requests the whole document, so the persisted locator
        # matches it exactly (a partial read would fail closed here).
        ref = await self._builder.persist_use(
            source=DocumentSourceIdentity(
                kind="document",
                document_id=self._document_id,
                document_revision=str(self._revision_id),
                locator=DocumentLocator(kind="document"),
            ),
            content=self._content,
            provenance=Provenance(
                acquisition_id=uuid.uuid4(),
                fetcher="document.read",
                fetched_at=_now(),
            ),
            task_id=request.task_id,
            purpose="coverage",
            target_id=target_id,
        )
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=DocumentReadOutput(kind="document.read", read_unit_count=1),
            evidence_uses=(ref,),
            coverage_observations=(
                CoverageObservation(
                    target_id=target_id,
                    observed_locators=(DocumentLocator(kind="document"),),
                    outcome="read",
                ),
            ),
            error=None,
        )


class TestGovernedNodeChain:
    @pytest.mark.asyncio
    async def test_nodes_run_on_governed_evidence_end_to_end(
        self, async_db: AsyncSession, document_factory: Any
    ) -> None:
        """Production paths (nodes + scheduler + real adapter), no hydration fakes."""
        from app.services.agents.v2.capabilities import CapabilityRegistration
        from app.services.agents.v2.capabilities import (
            build_capability_registry as _build_registry,
        )

        governor = _governor(async_db)
        adapter = GovernorEvidenceHydrator(governor)
        document_id, revision_id, workspace_id = await _seed_document(
            async_db, document_factory
        )
        content = "Điều 5 quy định mức phạt hành chính."
        builder = DbEvidenceBuilder(
            governor, async_db, run_id=RUN_ID, revision_id=revision_id
        )
        capability = _DocReadCapability(
            builder, document_id, revision_id, content
        )
        capability_runtime = _runtime(workspace_ids=(workspace_id,))
        registry = _build_registry(
            [CapabilityRegistration(capability=capability)],  # type: ignore[arg-type]
            capability_runtime,
        )
        leases = _FakeLeaseRepo()
        channel = AnswerDraftChannel()
        context = _graph_runtime(
            adapter,
            capability_runtime,
            leases=leases,
            channel=channel,
            registry=registry,
        )
        plan, bindings = _plan(), _bindings(document_id, str(revision_id))
        report = await TaskScheduler(registry).execute(
            plan=plan, runtime=context, prior_results=(), bindings=bindings
        )
        assert len(report.results) == 1
        assert leases.calls, "dispatch-created uses must be leased"

        state = SupervisorV2State(
            contract_version="2.0",
            request=RequestContext(
                contract_version="2.0",
                request_id="req-1",
                thread_id="thread-1",
                original_query="Điều 5 của A nói gì?",
                known_documents=(),
            ),
            conversation=ConversationContext(
                summary="", active_entities=(), last_focus=None, recent_turns=()
            ),
            semantic=_semantic(),
            bindings=bindings,
            query_analysis=QueryAnalysis(
                work_type="retrieve", domains=("document",)
            ),
            route_decision=RouteDecision(
                route="fast_domain", reason_code="exact_document_metadata"
            ),
            execution=ExecutionState(
                plan=plan, task_results=report.results, evidence_evaluation=None
            ),
            clarification=None,
            final_response=None,
        )
        evaluated = await evaluate_node(state, context)
        assert evaluated["execution"].evidence_evaluation.status == "sufficient"
        state["execution"] = evaluated["execution"]
        assert await synthesize_node(state, context) == {}
        assert await ground_node(state, context) == {}
        final = await finalizer_node(state, context)
        assert final["final_response"].status == "success"
        assert content in final["final_response"].content
        assert [c.citation_id for c in final["final_response"].citations] == [
            "cite-1"
        ]
