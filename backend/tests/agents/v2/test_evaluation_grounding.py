"""Task 4 — evaluator, synthesis hydration, grounding, and overflow tests (TDD).

Covers the brief Step 1 scenarios: search-not-read coverage; wrong/partial
section; revision mismatch; targetless supporting use; discovery exclusion;
tombstone; expiry; People minimization; same record/two uses; derived
faithfulness; all four evaluation statuses plus precedence; revise-once
grounding; deterministic citations; budget-overflow derived persistence;
synthesis-only reuse revalidation. Round 1 adds: targetless sufficiency
coupled to evidence (denied/error/not_found/zero-task), contradiction
admission, judge-invoked self-certification refusal, single-sided
contradictions, and the runtime-only draft channel.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityRuntimeContext,
    DocumentSearchOutput,
    PeopleLookupOutput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evaluation import (
    Contradiction,
    Coverage,
    EvidenceEvaluation,
)
from app.services.agents.v2.contracts.evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceClassification,
    EvidenceRecord,
    EvidenceUse,
    EvidenceUseRef,
    PeopleSourceIdentity,
    Provenance,
)
from app.services.agents.v2.contracts.execution import AgentError, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator, SectionLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    SemanticCriterion,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import (
    BlockingAmbiguity,
    SemanticContext,
)
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.synthesis import (
    AnswerClaim,
    AnswerDraft,
    SynthesisEvidence,
    SynthesisInput,
    SynthesisRuntimeContext,
)
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.nodes.evaluate import (
    AnswerDraftChannel,
    EvaluationError,
    EvidenceHydrator,
    HydratedEvidence,
    SemanticJudge,
    build_coverage,
    every_expecting_task_has_evidence,
    evaluate_evidence,
    evaluate_node,
    find_missing_requirements,
)
from app.services.agents.v2.nodes.execute import MissingCheckpointedPlan
from app.services.agents.v2.nodes.finalizer import finalizer_node
from app.services.agents.v2.nodes.grounding import (
    GroundingInsufficient,
    ground_answer,
    ground_node,
    render_citations,
)
from app.services.agents.v2.nodes.synthesize import (
    SynthesisError,
    apply_budget_split,
    estimate_tokens_for_chars,
    hydrate_for_synthesis,
    synthesize_answer,
    synthesize_node,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
REVISION_UUID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
REVISION = str(REVISION_UUID)
OTHER_REVISION = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
FETCHED_AT = datetime(2026, 9, 11, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fake evidence store + hydrator (mirrors EvidenceGovernor admission semantics)
# ---------------------------------------------------------------------------


@dataclass
class StoredUse:
    use: EvidenceUse
    source_kind: str  # "document" | "people" | "derived"
    document_id: UUID | None = None
    document_revision: str | None = None
    locator: Any = None
    content: str = ""
    classification: EvidenceClassification = "normal"
    expired: bool = False
    tombstoned: bool = False
    derived_validated: bool = False
    lineage: tuple[UUID, ...] = ()


def _role_for(binding: ScopedDocument | None) -> Any:
    return binding.role if binding is not None else None


class FakeHydrator:
    """In-memory EvidenceHydrator: admission mirrors the governed gate.

    Denials are SKIPPED (never raised) and recorded on ``denied`` so tests can
    prove no silent drop: every exclusion has an audited reason. The current
    plan/bindings/budget arrive explicitly per call (execute_node precedent).
    """

    def __init__(
        self,
        store: dict[UUID, StoredUse],
    ) -> None:
        self.store = dict(store)
        self.denied: list[tuple[UUID, str]] = []
        self.persisted_records: list[EvidenceRecord] = []
        self.minted_uses: list[EvidenceUse] = []
        self.hydrate_calls: list[tuple[str, tuple[UUID, ...]]] = []

    # -- admission ------------------------------------------------------
    def _binding_for(
        self, plan: TaskPlan, bindings: DocumentBindingSet, target_id: str | None
    ) -> ScopedDocument | None:
        if target_id is None:
            return None
        unit = next(
            (u for u in plan.target_units if u.target_id == target_id), None
        )
        if unit is None:
            return None
        return next(
            (b for b in bindings.bindings if b.binding_id == unit.binding_id),
            None,
        )

    def _admit(
        self,
        ref: EvidenceUseRef,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> HydratedEvidence | None:
        item = self.store.get(ref.use_id)
        if item is None:
            self.denied.append((ref.use_id, "unknown_use"))
            return None
        use = item.use
        if use.purpose == "discovery":
            self.denied.append((use.use_id, "purpose_not_hydratable"))
            return None
        if use.purpose == "coverage" and use.target_id is None:
            self.denied.append((use.use_id, "purpose_not_hydratable"))
            return None
        if item.expired:
            self.denied.append((use.use_id, "expired"))
            return None
        if item.tombstoned:
            self.denied.append((use.use_id, "document_tombstoned"))
            return None
        binding = self._binding_for(plan, bindings, use.target_id)
        if item.source_kind == "derived" and use.target_id is not None:
            # Mirrors the governor's revision gate: a target-bound
            # non-document use hydrates denied. Overflow-derived items are
            # returned directly at persist time and never re-hydrated.
            self.denied.append((use.use_id, "revision_mismatch"))
            return None
        if item.source_kind == "document":
            if (
                binding is None
                or item.document_id != binding.document_id
                or item.document_revision != binding.document_revision
            ):
                self.denied.append((use.use_id, "revision_mismatch"))
                return None
        elif item.source_kind == "people":
            if not runtime.capability_runtime.can_read_people:
                self.denied.append((use.use_id, "people_not_authorized"))
                return None
        elif item.source_kind == "derived":
            if not item.derived_validated:
                self.denied.append((use.use_id, "derived_not_validated"))
                return None
            for ancestor in item.lineage:
                source = self.store_by_evidence(ancestor)
                if source is None or source.expired or source.tombstoned:
                    self.denied.append((use.use_id, "derived_source_unresolved"))
                    return None
        if use.target_id is not None and binding is None:
            self.denied.append((use.use_id, "target_unknown"))
            return None
        label = (
            f"doc:{str(item.document_id)}"[:12]
            if item.source_kind == "document"
            else item.source_kind
        )
        return HydratedEvidence(
            use_id=use.use_id,
            evidence_id=use.evidence_id,
            task_id=use.task_id,
            purpose=use.purpose,
            target_id=use.target_id,
            content=item.content,
            role=_role_for(binding),
            source_label=label,
            classification=item.classification,
            locator=item.locator,
            document_revision=item.document_revision,
        )

    def store_by_evidence(self, evidence_id: UUID) -> StoredUse | None:
        for item in self.store.values():
            if item.use.evidence_id == evidence_id:
                return item
        return None

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> tuple[HydratedEvidence, ...]:
        self.hydrate_calls.append(
            ("evaluation", tuple(r.use_id for r in use_refs))
        )
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            item = self._admit(ref, runtime, plan, bindings)
            if item is not None:
                admitted.append(item)
        return tuple(admitted)

    async def hydrate_for_synthesis(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
        budget: SynthesisRuntimeContext,
    ) -> tuple[HydratedEvidence, ...]:
        self.hydrate_calls.append(
            ("synthesis", tuple(r.use_id for r in use_refs))
        )
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            item = self._admit(ref, runtime, plan, bindings)
            if item is not None:
                admitted.append(item)
        head, tail = apply_budget_split(tuple(admitted), budget)
        if not tail:
            return tuple(head)
        derived = await self.persist_derived_summary(
            content="\n\n".join(t.content for t in tail),
            source_evidence_ids=tuple(t.evidence_id for t in tail),
            task_id=tail[0].task_id,
            target_id=tail[0].target_id,
            provenance=Provenance(
                acquisition_id=uuid4(),
                fetcher="synthesis.overflow",
                fetched_at=datetime.now(UTC),
            ),
            classification=_max_classification(t.classification for t in tail),
            run_id=runtime.capability_runtime.run_id,
            plan=plan,
            bindings=bindings,
        )
        return tuple(head) + (derived,)

    async def persist_derived_summary(
        self,
        *,
        content: str,
        source_evidence_ids: tuple[UUID, ...],
        task_id: str,
        target_id: str | None,
        provenance: Provenance,
        classification: EvidenceClassification,
        run_id: str,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> HydratedEvidence:
        from app.services.agents.v2.contracts.validation import (
            validate_derived_evidence_faithfulness,
            validate_evidence_record,
            validate_evidence_use,
        )

        del run_id, plan, bindings
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        for record in self.persisted_records:
            if (
                record.content_hash == content_hash
                and isinstance(record.source, DerivedSourceIdentity)
                and record.source.source_evidence_ids == source_evidence_ids
            ):
                existing_use = next(
                    u for u in self.minted_uses if u.evidence_id == record.evidence_id
                )
                return self._hydrated_of(existing_use)
        source_records = [
            self._record_of(eid)
            for eid in source_evidence_ids
            if self._record_of(eid) is not None
        ]
        record = EvidenceRecord(
            evidence_id=uuid4(),
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=source_evidence_ids
            ),
            content=content,
            content_hash=content_hash,
            provenance=provenance,
        )
        validate_evidence_record(record)
        validate_derived_evidence_faithfulness(
            record,
            tuple(source_records),
            faithfulness_validated=True,
        )
        use = EvidenceUse(
            use_id=uuid4(),
            evidence_id=record.evidence_id,
            task_id=task_id,
            purpose="supporting",
            target_id=target_id,
        )
        validate_evidence_use(use)
        self.persisted_records.append(record)
        self.minted_uses.append(use)
        self.store[use.use_id] = StoredUse(
            use=use,
            source_kind="derived",
            content=content,
            classification=classification,
            derived_validated=True,
            lineage=source_evidence_ids,
        )
        return self._hydrated_of(use)

    def _record_of(self, evidence_id: UUID) -> EvidenceRecord | None:
        for record in self.persisted_records:
            if record.evidence_id == evidence_id:
                return record
        item = self.store_by_evidence(evidence_id)
        if item is None:
            return None
        if item.source_kind == "document":
            source: Any = DocumentSourceIdentity(
                kind="document",
                document_id=item.document_id or DOCUMENT_ID,
                document_revision=item.document_revision or REVISION,
                locator=item.locator or DocumentLocator(kind="document"),
            )
        elif item.source_kind == "people":
            source = PeopleSourceIdentity(kind="people", record_id="rec-1")
        else:
            source = DerivedSourceIdentity(
                kind="derived", source_evidence_ids=item.lineage
            )
        return EvidenceRecord(
            evidence_id=evidence_id,
            source=source,
            content=item.content,
            content_hash=hashlib.sha256(item.content.encode()).hexdigest(),
            provenance=Provenance(
                acquisition_id=uuid4(), fetcher="test", fetched_at=FETCHED_AT
            ),
        )

    def _hydrated_of(self, use: EvidenceUse) -> HydratedEvidence:
        item = self.store[use.use_id]
        return HydratedEvidence(
            use_id=use.use_id,
            evidence_id=use.evidence_id,
            task_id=use.task_id,
            purpose=use.purpose,
            target_id=use.target_id,
            content=item.content,
            role=None,
            source_label="derived",
            classification=item.classification,
            locator=None,
            document_revision=None,
        )


def _max_classification(
    values: Any,
) -> EvidenceClassification:
    rank = {"normal": 0, "personal": 1, "sensitive_personal": 2}
    best = "normal"
    for value in values:
        if rank[value] > rank[best]:
            best = value
    return best  # type: ignore[return-value]


class StubJudge:
    """Bounded-model stand-in: verdicts over governed evidence only."""

    def __init__(
        self,
        verdicts: dict[str, bool] | None = None,
        contradictions: tuple[Contradiction, ...] = (),
    ) -> None:
        self.verdicts = verdicts or {}
        self._contradictions = contradictions
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def assess_criterion(self, *, criterion: Any, evidence: Any) -> bool:
        self.calls.append(
            (criterion.criterion_id, tuple(h.content for h in evidence))
        )
        return self.verdicts.get(criterion.criterion_id, False)

    async def detect_contradictions(self, *, evidence: Any) -> Any:
        return self._contradictions


class FakeLeaseSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class FakeLeaseRepo:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[Any, Any, Any]] = []
        self.session = FakeLeaseSession(events)

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Any = None,
        evidence_use_id: Any = None,
        *,
        now: Any = None,
    ) -> Any:
        self.calls.append((run_id, revision_id, evidence_use_id))
        self.events.append(f"acquire:{evidence_use_id}")
        return None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def capability_runtime(
    *,
    allowed: frozenset[str] = frozenset({"document.read", "document.search"}),
    can_read_people: bool = True,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=can_read_people,
        allowed_capabilities=allowed,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def graph_runtime(
    hydrator: FakeHydrator | None = None,
    leases: FakeLeaseRepo | None = None,
    channel: AnswerDraftChannel | None = None,
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=capability_runtime(),
        services=RuntimeServices(
            retention_leases=leases,
            evidence_hydrator=hydrator,
            answer_draft_channel=channel,
        ),
    )


def semantic(
    *, ambiguities: tuple[BlockingAmbiguity, ...] = ()
) -> SemanticContext:
    return SemanticContext(
        contextualized_query="Điều 5 của A nói gì?",
        normalized_query="Điều 5 của A nói gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=ambiguities,
    )


def binding_set(
    *,
    binding_id: str = "b_r1",
    document_revision: str = REVISION,
    role: str = "target",
) -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id=binding_id,
                document_id=DOCUMENT_ID,
                document_revision=document_revision,
                role=role,  # type: ignore[arg-type]
            ),
        ),
        revision_requirement_refs=(),
    )


def read_plan(
    *,
    target_id: str = "t1",
    binding_id: str = "b_r1",
    locator: Any = None,
    criteria: Any = None,
    task_id: str = "T1",
    capability: str = "document.read",
) -> TaskPlan:
    from app.services.agents.v2.contracts.capability import DocumentReadInput

    return TaskPlan(
        contract_version="2.0",
        plan_id="p1",
        goal="Đọc Điều 5 của A",
        target_units=(
            TargetUnit(
                target_id=target_id,
                binding_id=binding_id,
                requested_locator=locator or DocumentLocator(kind="document"),
                completion_criteria=criteria
                if criteria is not None
                else (CoverageCriterion(kind="coverage"),),
            ),
        ),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability=capability,
                task_objective="Đọc Điều 5 của A",
                input=DocumentReadInput(
                    kind="document.read", target_ids=(target_id,)
                ),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def people_plan(*, task_id: str = "T1") -> TaskPlan:
    from app.services.agents.v2.contracts.capability import PeopleLookupInput

    return TaskPlan(
        contract_version="2.0",
        plan_id="p-people",
        goal="CCCD của A là gì?",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability="people.lookup",
                task_objective="Tra CCCD của A",
                input=PeopleLookupInput(kind="people.lookup", query="A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def read_result(
    task_id: str = "T1",
    target_id: str = "t1",
    *,
    use_id: UUID | None = None,
    observed: Any = None,
    outcome: str = "read",
    status: str = "success",
) -> AgentResult:
    from app.services.agents.v2.contracts.capability import DocumentReadOutput
    from app.services.agents.v2.contracts.evaluation import CoverageObservation

    locator = observed or DocumentLocator(kind="document")
    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        data=DocumentReadOutput(kind="document.read", read_unit_count=1)
        if status in ("success", "partial")
        else None,
        evidence_uses=(EvidenceUseRef(use_id=use_id),) if use_id else (),
        coverage_observations=(
            CoverageObservation(
                target_id=target_id,
                observed_locators=(locator,),
                outcome=outcome,  # type: ignore[arg-type]
            ),
        ),
        error=None,
    )


def people_result(
    task_id: str = "T1", *, status: str = "success", use_id: UUID | None = None
) -> AgentResult:
    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        data=PeopleLookupOutput(kind="people.lookup", matched=True)
        if status == "success"
        else None,
        evidence_uses=(EvidenceUseRef(use_id=use_id),) if use_id else (),
        coverage_observations=(),
        error=AgentError(code="PERMISSION_DENIED", message=status, retryable=False)
        if status in ("denied", "error")
        else None,
    )


def store_coverage_use(
    *,
    use_id: UUID,
    evidence_id: UUID,
    task_id: str = "T1",
    target_id: str = "t1",
    revision: str = REVISION,
    locator: Any = None,
    content: str = "Điều 5 quy định mức phạt.",
) -> StoredUse:
    return StoredUse(
        use=EvidenceUse(
            use_id=use_id,
            evidence_id=evidence_id,
            task_id=task_id,
            purpose="coverage",
            target_id=target_id,
        ),
        source_kind="document",
        document_id=DOCUMENT_ID,
        document_revision=revision,
        locator=locator or DocumentLocator(kind="document"),
        content=content,
    )


def generous_budget() -> SynthesisRuntimeContext:
    return SynthesisRuntimeContext(
        max_evidence_items=10, max_total_chars=100_000, max_total_tokens=25_000
    )


def make_state(
    *,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    results: tuple[AgentResult, ...] = (),
    evaluation: EvidenceEvaluation | None = None,
    route: RouteDecision | None = None,
) -> SupervisorV2State:
    return SupervisorV2State(
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
        semantic=semantic(),
        bindings=bindings,
        query_analysis=QueryAnalysis(work_type="retrieve", domains=("document",)),
        route_decision=route,
        execution=ExecutionState(
            plan=plan, task_results=results, evidence_evaluation=evaluation
        ),
        clarification=None,
        final_response=None,
    )


EVIDENCE_ID = UUID("33333333-3333-3333-3333-333333333333")
USE_ID = UUID("44444444-4444-4444-4444-444444444444")


# ---------------------------------------------------------------------------
# Evaluator: coverage rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_results_never_complete_coverage() -> None:
    """Discovery candidates are not read coverage: search success + read miss."""
    from app.services.agents.v2.contracts.binding import DocumentDiscoveryCandidate

    plan = read_plan()
    bindings = binding_set()
    search_result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="success",
        data=DocumentSearchOutput(
            kind="document.search",
            candidates=(
                DocumentDiscoveryCandidate(
                    candidate_id=uuid4(),
                    document_id=DOCUMENT_ID,
                    document_revision=REVISION,
                ),
            ),
        ),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    hydrator = FakeHydrator({})
    runtime = graph_runtime(hydrator)
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=bindings,
        results=(search_result,),
        semantic=semantic(),
        runtime=runtime,
    )
    assert evaluation.status == "insufficient"
    assert len(evaluation.missing) == 1
    assert evaluation.missing[0].target_id == "t1"
    assert evaluation.missing[0].criterion_kind == "coverage"


@pytest.mark.asyncio
async def test_wrong_section_read_is_missing() -> None:
    """A read of section S2 never covers requested section S1."""
    requested = SectionLocator(kind="section", structure_node_id="sec-1")
    observed = SectionLocator(kind="section", structure_node_id="sec-2")
    plan = read_plan(locator=requested)
    bindings = binding_set()
    use_id, evidence_id = uuid4(), uuid4()
    hydrator = FakeHydrator(
        {
            use_id: store_coverage_use(
                use_id=use_id,
                evidence_id=evidence_id,
                locator=observed,
                content="Nội dung mục 2.",
            )
        },
    )
    runtime = graph_runtime(hydrator)
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=bindings,
        results=(read_result(use_id=use_id, observed=observed),),
        semantic=semantic(),
        runtime=runtime,
    )
    assert evaluation.status == "insufficient"
    assert evaluation.coverage.items[0].status == "missing"


@pytest.mark.asyncio
async def test_partial_document_read_needs_explicit_allowance() -> None:
    """Section read of a whole-document target: partial without/with allowance."""
    observed = SectionLocator(kind="section", structure_node_id="sec-1")
    use_id, evidence_id = uuid4(), uuid4()

    def setup(criteria: Any) -> tuple[TaskPlan, FakeHydrator]:
        plan = read_plan(locator=DocumentLocator(kind="document"), criteria=criteria)
        hydrator = FakeHydrator(
            {
                use_id: store_coverage_use(
                    use_id=use_id,
                    evidence_id=evidence_id,
                    locator=observed,
                    content="Nội dung mục 1.",
                )
            },
        )
        return plan, hydrator

    plan, hydrator = setup((CoverageCriterion(kind="coverage"),))
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=binding_set(),
        results=(read_result(use_id=use_id, observed=observed),),
        semantic=semantic(),
        runtime=graph_runtime(hydrator),
    )
    assert evaluation.coverage.items[0].status == "read_partial"
    assert evaluation.status == "insufficient"

    plan2, hydrator2 = setup(
        (
            CoverageCriterion(
                kind="coverage",
                minimum_status="read_partial",
                allow_partial_reason="mục 1 là đủ",
            ),
        )
    )
    evaluation2 = await evaluate_evidence(
        plan=plan2,
        bindings=binding_set(),
        results=(read_result(use_id=use_id, observed=observed),),
        semantic=semantic(),
        runtime=graph_runtime(hydrator2),
    )
    assert evaluation2.status == "sufficient"


@pytest.mark.asyncio
async def test_revision_mismatch_excludes_use_from_coverage() -> None:
    """Observed read + stale-revision use: hydrator denies, coverage missing."""
    plan = read_plan()
    bindings = binding_set()
    use_id, evidence_id = uuid4(), uuid4()
    hydrator = FakeHydrator(
        {
            use_id: store_coverage_use(
                use_id=use_id, evidence_id=evidence_id, revision=OTHER_REVISION
            )
        },
    )
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=bindings,
        results=(read_result(use_id=use_id),),
        semantic=semantic(),
        runtime=graph_runtime(hydrator),
    )
    assert evaluation.status == "insufficient"
    assert hydrator.denied == [(use_id, "revision_mismatch")]
    assert evaluation.coverage.items[0].status == "missing"


@pytest.mark.asyncio
async def test_evaluator_rechecks_revision_even_when_hydration_admits() -> None:
    """A use admitted with a drifted revision still cannot confirm coverage."""
    plan = read_plan()
    bindings = binding_set()
    use_id = uuid4()
    hydrated = (
        HydratedEvidence(
            use_id=use_id,
            evidence_id=uuid4(),
            task_id="T1",
            purpose="coverage",
            target_id="t1",
            content="Điều 5.",
            role="target",
            source_label="doc",
            classification="normal",
            locator=DocumentLocator(kind="document"),
            document_revision=OTHER_REVISION,
        ),
    )
    coverage = build_coverage(plan, bindings, (read_result(use_id=use_id),), hydrated)
    assert coverage.items[0].status == "missing"


@pytest.mark.asyncio
async def test_targetless_supporting_use_succeeds_for_people_plan() -> None:
    """People lookup: zero targets, one admitted supporting use → sufficient."""
    plan = people_plan()
    use_id, evidence_id = uuid4(), uuid4()
    store = {
        use_id: StoredUse(
            use=EvidenceUse(
                use_id=use_id,
                evidence_id=evidence_id,
                task_id="T1",
                purpose="supporting",
                target_id=None,
            ),
            source_kind="people",
            content='{"name":"Nguyễn Văn A"}',
            classification="personal",
        )
    }
    hydrator = FakeHydrator(store)
    result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(),
        error=None,
    )
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=binding_set(),
        results=(result,),
        semantic=semantic(),
        runtime=graph_runtime(hydrator),
    )
    assert evaluation.status == "sufficient"
    assert evaluation.coverage.items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["denied", "error", "not_found"])
async def test_targetless_task_without_evidence_is_insufficient(status: str) -> None:
    """C1: a People task that produced nothing never certifies sufficient."""
    plan = people_plan()
    hydrator = FakeHydrator({})
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=binding_set(),
        results=(people_result(status=status),),
        semantic=semantic(),
        runtime=graph_runtime(hydrator),
    )
    assert evaluation.status == "insufficient"
    assert not every_expecting_task_has_evidence(plan, ())


@pytest.mark.asyncio
async def test_zero_task_plan_is_insufficient() -> None:
    """C1: a plan with no tasks (and no direct path) is never sufficient."""
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="p-empty",
        goal="nothing planned",
        target_units=(),
        tasks=(),
    )
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=binding_set(),
        results=(),
        semantic=semantic(),
        runtime=graph_runtime(FakeHydrator({})),
    )
    assert evaluation.status == "insufficient"


@pytest.mark.asyncio
async def test_denied_read_task_drives_insufficient() -> None:
    """C1: a denied document.read names its target instead of certifying."""
    plan = read_plan()
    denied = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="denied",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(
            code="PERMISSION_DENIED", message="no pin", retryable=False
        ),
    )
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=binding_set(),
        results=(denied,),
        semantic=semantic(),
        runtime=graph_runtime(FakeHydrator({})),
    )
    assert evaluation.status == "insufficient"
    assert any(m.target_id == "t1" for m in evaluation.missing)


@pytest.mark.asyncio
async def test_tombstoned_and_expired_uses_are_excluded() -> None:
    plan = read_plan()
    bindings = binding_set()
    tomb_id, tomb_ev, exp_id, exp_ev = uuid4(), uuid4(), uuid4(), uuid4()
    hydrator = FakeHydrator(
        {
            tomb_id: StoredUse(
                use=EvidenceUse(
                    use_id=tomb_id,
                    evidence_id=tomb_ev,
                    task_id="T1",
                    purpose="coverage",
                    target_id="t1",
                ),
                source_kind="document",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                locator=DocumentLocator(kind="document"),
                content="Đã xóa.",
                tombstoned=True,
            ),
            exp_id: StoredUse(
                use=EvidenceUse(
                    use_id=exp_id,
                    evidence_id=exp_ev,
                    task_id="T1",
                    purpose="coverage",
                    target_id="t1",
                ),
                source_kind="document",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                locator=DocumentLocator(kind="document"),
                content="Hết hạn.",
                expired=True,
            ),
        },
    )
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=bindings,
        results=(read_result(use_id=tomb_id), read_result(use_id=exp_id)),
        semantic=semantic(),
        runtime=graph_runtime(hydrator),
    )
    assert evaluation.status == "insufficient"
    assert (tomb_id, "document_tombstoned") in hydrator.denied
    assert (exp_id, "expired") in hydrator.denied


@pytest.mark.asyncio
async def test_people_content_stays_minimized_through_hydration() -> None:
    """Only task-required People fields reach synthesis evidence."""
    plan = people_plan()
    use_id = uuid4()
    minimized = '{"name":"Nguyễn Văn A"}'
    hydrator = FakeHydrator(
        {
            use_id: StoredUse(
                use=EvidenceUse(
                    use_id=use_id,
                    evidence_id=uuid4(),
                    task_id="T1",
                    purpose="supporting",
                    target_id=None,
                ),
                source_kind="people",
                content=minimized,
                classification="personal",
            )
        },
    )
    runtime = graph_runtime(hydrator)
    evidence = await hydrate_for_synthesis(
        (EvidenceUseRef(use_id=use_id),),
        runtime,
        plan=plan,
        bindings=binding_set(),
        budget=generous_budget(),
    )
    assert len(evidence) == 1
    assert evidence[0].content == minimized
    import json

    assert set(json.loads(evidence[0].content)) == {"name"}


@pytest.mark.asyncio
async def test_same_record_two_uses_preserve_context() -> None:
    """Coverage + supporting uses of one record hydrate with own purpose/target."""
    plan = read_plan()
    evidence_id = uuid4()
    coverage_id, supporting_id = uuid4(), uuid4()
    hydrator = FakeHydrator(
        {
            coverage_id: store_coverage_use(
                use_id=coverage_id,
                evidence_id=evidence_id,
                content="Điều 5 quy định mức phạt.",
            ),
            supporting_id: StoredUse(
                use=EvidenceUse(
                    use_id=supporting_id,
                    evidence_id=evidence_id,
                    task_id="T1",
                    purpose="supporting",
                    target_id="t1",
                ),
                source_kind="document",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                locator=DocumentLocator(kind="document"),
                content="Điều 5 quy định mức phạt.",
            ),
        },
    )
    runtime = graph_runtime(hydrator)
    admitted = await hydrator.hydrate_for_evaluation(
        (EvidenceUseRef(use_id=coverage_id), EvidenceUseRef(use_id=supporting_id)),
        runtime=runtime,
        plan=plan,
        bindings=binding_set(),
    )
    assert len(admitted) == 2
    by_id = {h.use_id: h for h in admitted}
    assert by_id[coverage_id].purpose == "coverage"
    assert by_id[supporting_id].purpose == "supporting"
    assert by_id[coverage_id].evidence_id == by_id[supporting_id].evidence_id


@pytest.mark.asyncio
async def test_derived_use_requires_recursive_validation() -> None:
    """Unvalidated derived evidence is denied; validated lineage is admitted."""
    plan = people_plan()
    source_evidence = uuid4()
    source_use = uuid4()
    derived_evidence = uuid4()
    derived_use = uuid4()
    store = {
        source_use: StoredUse(
            use=EvidenceUse(
                use_id=source_use,
                evidence_id=source_evidence,
                task_id="T1",
                purpose="supporting",
                target_id=None,
            ),
            source_kind="people",
            content='{"name":"A"}',
            classification="personal",
        ),
        derived_use: StoredUse(
            use=EvidenceUse(
                use_id=derived_use,
                evidence_id=derived_evidence,
                task_id="T1",
                purpose="supporting",
                target_id=None,
            ),
            source_kind="derived",
            content="Tóm tắt: A.",
            derived_validated=False,
            lineage=(source_evidence,),
        ),
    }
    hydrator = FakeHydrator(store)
    runtime = graph_runtime(hydrator)
    assert (
        await hydrator.hydrate_for_evaluation(
            (EvidenceUseRef(use_id=derived_use),),
            runtime=runtime,
            plan=plan,
            bindings=binding_set(),
        )
    ) == ()
    assert hydrator.denied == [(derived_use, "derived_not_validated")]

    store[derived_use].derived_validated = True
    admitted = await hydrator.hydrate_for_evaluation(
        (EvidenceUseRef(use_id=derived_use),),
        runtime=runtime,
        plan=plan,
        bindings=binding_set(),
    )
    assert [h.use_id for h in admitted] == [derived_use]


# ---------------------------------------------------------------------------
# Evaluator: statuses and precedence
# ---------------------------------------------------------------------------


def _contradiction(use_ids: tuple[UUID, ...]) -> Contradiction:
    return Contradiction(
        contradiction_id="k1",
        claim_a="Mức phạt là 5 triệu.",
        claim_b="Mức phạt là 10 triệu.",
        evidence_use_ids=use_ids,
    )


@pytest.mark.asyncio
async def test_all_four_statuses_and_precedence() -> None:
    plan = read_plan()
    bindings = binding_set()
    other_use = uuid4()
    hydrator = FakeHydrator(
        {
            USE_ID: store_coverage_use(use_id=USE_ID, evidence_id=EVIDENCE_ID),
            other_use: store_coverage_use(
                use_id=other_use, evidence_id=uuid4(), content="Mức khác."
            ),
        },
    )
    runtime = graph_runtime(hydrator)
    good = (read_result(use_id=USE_ID), read_result(use_id=other_use))

    sufficient = await evaluate_evidence(
        plan=plan, bindings=bindings, results=good, semantic=semantic(),
        runtime=runtime,
    )
    assert sufficient.status == "sufficient"

    insufficient = await evaluate_evidence(
        plan=plan, bindings=bindings, results=(read_result(),), semantic=semantic(),
        runtime=runtime,
    )
    assert insufficient.status == "insufficient"

    clash = _contradiction((USE_ID, other_use))
    contradictory = await evaluate_evidence(
        plan=plan, bindings=bindings, results=good, semantic=semantic(),
        runtime=runtime,
        semantic_judge=StubJudge(contradictions=(clash,)),
    )
    assert contradictory.status == "contradictory"

    needs_input = await evaluate_evidence(
        plan=plan, bindings=bindings, results=good,
        semantic=semantic(
            ambiguities=(
                BlockingAmbiguity(ambiguity_id="a1", description="A nào?"),
            )
        ),
        runtime=runtime,
        semantic_judge=StubJudge(contradictions=(clash,)),
    )
    assert needs_input.status == "needs_input"

    # Precedence: contradiction beats missing; needs_input beats contradiction.
    # Missing coverage AND a valid two-sided contradiction → contradictory.
    contra_over_missing = await evaluate_evidence(
        plan=plan,
        bindings=bindings,
        results=(
            read_result(use_id=USE_ID, outcome="missing"),
            read_result(use_id=other_use, outcome="missing"),
        ),
        semantic=semantic(),
        runtime=runtime,
        semantic_judge=StubJudge(
            contradictions=(_contradiction((other_use, USE_ID)),)
        ),
    )
    assert contra_over_missing.status == "contradictory"

    needs_input_result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="needs_input",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    assert (
        await evaluate_evidence(
            plan=plan, bindings=bindings, results=(needs_input_result,),
            semantic=semantic(), runtime=runtime,
        )
    ).status == "needs_input"


@pytest.mark.asyncio
async def test_contradiction_outside_admitted_set_is_rejected() -> None:
    """M2: a conflict citing non-admitted uses cannot be certified."""
    plan = read_plan()
    bindings = binding_set()
    hydrator = FakeHydrator(
        {USE_ID: store_coverage_use(use_id=USE_ID, evidence_id=EVIDENCE_ID)},
    )
    with pytest.raises(EvaluationError, match="outside the admitted set"):
        await evaluate_evidence(
            plan=plan, bindings=bindings, results=(read_result(use_id=USE_ID),),
            semantic=semantic(), runtime=graph_runtime(hydrator),
            semantic_judge=StubJudge(
                contradictions=(_contradiction((USE_ID, uuid4())),)
            ),
        )


@pytest.mark.asyncio
async def test_single_sided_contradiction_does_not_block() -> None:
    """M6: a supported single-use contradiction may stand beside sufficient."""
    plan = read_plan()
    bindings = binding_set()
    hydrator = FakeHydrator(
        {USE_ID: store_coverage_use(use_id=USE_ID, evidence_id=EVIDENCE_ID)},
    )
    evaluation = await evaluate_evidence(
        plan=plan, bindings=bindings, results=(read_result(use_id=USE_ID),),
        semantic=semantic(), runtime=graph_runtime(hydrator),
        semantic_judge=StubJudge(contradictions=(_contradiction((USE_ID,)),)),
    )
    assert evaluation.status == "sufficient"
    assert len(evaluation.contradictions) == 1


@pytest.mark.asyncio
async def test_judge_cannot_self_certify_factual_success() -> None:
    """A cooperative judge never overrides missing deterministic coverage."""
    criteria = (
        CoverageCriterion(kind="coverage"),
        SemanticCriterion(
            kind="semantic", criterion_id="c-rel", description="Trọng tâm."
        ),
    )
    plan = read_plan(criteria=criteria)
    bindings = binding_set()
    hydrator = FakeHydrator({})
    judge = StubJudge(verdicts={"c-rel": True}, contradictions=())
    evaluation = await evaluate_evidence(
        plan=plan, bindings=bindings, results=(read_result(),), semantic=semantic(),
        runtime=graph_runtime(hydrator), semantic_judge=judge,
    )
    assert evaluation.status == "insufficient"
    # M4: the judge really ran — the verdict was refused, not skipped.
    assert judge.calls != []
    assert judge.calls[0][0] == "c-rel"


@pytest.mark.asyncio
async def test_semantic_criterion_needs_a_judge_verdict() -> None:
    criteria = (
        CoverageCriterion(kind="coverage"),
        SemanticCriterion(
            kind="semantic", criterion_id="c-rel", description="Trả lời đúng trọng tâm."
        ),
    )
    plan = read_plan(criteria=criteria)
    bindings = binding_set()
    hydrator = FakeHydrator(
        {USE_ID: store_coverage_use(use_id=USE_ID, evidence_id=EVIDENCE_ID)},
    )
    runtime = graph_runtime(hydrator)
    without_judge = await evaluate_evidence(
        plan=plan, bindings=bindings, results=(read_result(use_id=USE_ID),),
        semantic=semantic(), runtime=runtime,
    )
    assert without_judge.status == "insufficient"
    assert without_judge.missing[0].criterion_kind == "semantic"
    assert without_judge.missing[0].semantic_criterion_id == "c-rel"

    with_judge = await evaluate_evidence(
        plan=plan, bindings=bindings, results=(read_result(use_id=USE_ID),),
        semantic=semantic(), runtime=runtime,
        semantic_judge=StubJudge(verdicts={"c-rel": True}),
    )
    assert with_judge.status == "sufficient"


@pytest.mark.asyncio
async def test_build_coverage_and_missing_helpers_agree() -> None:
    plan = read_plan()
    bindings = binding_set()
    hydrator = FakeHydrator({})
    hydrated = await hydrator.hydrate_for_evaluation(
        (), runtime=graph_runtime(hydrator), plan=plan, bindings=bindings
    )
    coverage = build_coverage(plan, bindings, (read_result(),), hydrated)
    assert coverage.items[0].target_id == "t1"
    missing = find_missing_requirements(plan, coverage, hydrated, {})
    # One entry: the task-level rule does not duplicate a flagged target.
    assert len(missing) == 1
    assert missing[0].target_id == "t1"


@pytest.mark.asyncio
async def test_evaluate_node_checkpoints_evaluation() -> None:
    plan = read_plan()
    bindings = binding_set()
    hydrator = FakeHydrator(
        {USE_ID: store_coverage_use(use_id=USE_ID, evidence_id=EVIDENCE_ID)},
    )
    state = make_state(
        plan=plan, bindings=bindings, results=(read_result(use_id=USE_ID),)
    )
    update = await evaluate_node(state, graph_runtime(hydrator))
    assert set(update) == {"execution"}
    assert update["execution"].evidence_evaluation is not None
    assert update["execution"].evidence_evaluation.status == "sufficient"
    assert update["execution"].plan == plan
    assert update["execution"].task_results == (read_result(use_id=USE_ID),)


@pytest.mark.asyncio
async def test_evaluate_node_requires_a_checkpointed_plan() -> None:
    state = make_state(plan=read_plan(), bindings=binding_set())
    state["execution"] = ExecutionState(
        plan=None, task_results=(), evidence_evaluation=None
    )
    with pytest.raises(MissingCheckpointedPlan):
        await evaluate_node(state, graph_runtime())


@pytest.mark.asyncio
async def test_evaluate_fails_closed_without_hydrator() -> None:
    from app.services.agents.v2.nodes.evaluate import EvaluationError

    plan = read_plan()
    with pytest.raises(EvaluationError, match="evidence_hydrator"):
        await evaluate_evidence(
            plan=plan, bindings=binding_set(), results=(read_result(),),
            semantic=semantic(), runtime=graph_runtime(None),
        )


def test_answer_draft_channel_stores_per_run() -> None:
    """C2: the runtime-only handoff is keyed by run id and reports misses."""
    from app.services.agents.v2.contracts.synthesis import SynthesisEvidence

    channel = AnswerDraftChannel()
    assert channel.get("run-1") is None
    draft = AnswerDraft(
        content="Điều 5.",
        claims=(
            AnswerClaim(
                claim_id="claim-1",
                text="Điều 5.",
                evidence_use_ids=(USE_ID,),
            ),
        ),
    )
    channel.store_draft("run-1", draft=draft, evidence=())
    assert channel.get("run-1") is not None
    assert channel.get("run-1").draft == draft
    assert channel.get("run-1").grounded_draft is None
    assert channel.get("other-run") is None


# ---------------------------------------------------------------------------
# Synthesis hydration + budget overflow
# ---------------------------------------------------------------------------


def _sufficient_input(
    *use_ids: UUID, plan: TaskPlan | None = None
) -> SynthesisInput:
    return SynthesisInput(
        semantic=semantic(),
        evaluation=EvidenceEvaluation(
            status="sufficient",
            coverage=Coverage(items=()),
            missing=(),
            contradictions=(),
        ),
        evidence_uses=tuple(EvidenceUseRef(use_id=u) for u in use_ids),
    )


@pytest.mark.asyncio
async def test_discovery_uses_excluded_from_synthesis() -> None:
    plan = read_plan()
    coverage_id, discovery_id = uuid4(), uuid4()
    hydrator = FakeHydrator(
        {
            coverage_id: store_coverage_use(
                use_id=coverage_id, evidence_id=uuid4()
            ),
            discovery_id: StoredUse(
                use=EvidenceUse(
                    use_id=discovery_id,
                    evidence_id=uuid4(),
                    task_id="T1",
                    purpose="discovery",
                    target_id=None,
                ),
                source_kind="document",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                content="candidate",
            ),
        },
    )
    evidence = await hydrate_for_synthesis(
        (EvidenceUseRef(use_id=coverage_id), EvidenceUseRef(use_id=discovery_id)),
        graph_runtime(hydrator),
        plan=plan,
        bindings=binding_set(),
        budget=generous_budget(),
    )
    assert [e.use_id for e in evidence] == [coverage_id]
    assert (discovery_id, "purpose_not_hydratable") in hydrator.denied
    assert isinstance(evidence[0], SynthesisEvidence)
    assert set(SynthesisEvidence.model_fields) == {
        "use_id",
        "content",
        "role",
        "target_id",
        "source_label",
    }


@pytest.mark.asyncio
async def test_synthesis_requires_sufficient_evaluation() -> None:
    plan = read_plan()
    hydrator = FakeHydrator({})
    bad = SynthesisInput(
        semantic=semantic(),
        evaluation=EvidenceEvaluation(
            status="insufficient",
            coverage=Coverage(items=()),
            missing=(),
            contradictions=(),
        ),
        evidence_uses=(),
    )
    with pytest.raises(ContractValidationError, match="sufficient"):
        await synthesize_answer(
            synthesis_input=bad,
            runtime=graph_runtime(hydrator),
            plan=plan,
            bindings=binding_set(),
            budget=generous_budget(),
        )


@pytest.mark.asyncio
async def test_budget_overflow_persisted_as_validated_derived_evidence() -> None:
    """Overflow tail is persisted (never silently truncated) with lineage."""
    plan = read_plan()
    u1, u2 = uuid4(), uuid4()
    e1, e2 = uuid4(), uuid4()
    content1 = "Điều 5 quy định mức phạt."
    content2 = "Điều 6 quy định thẩm quyền."
    hydrator = FakeHydrator(
        {
            u1: store_coverage_use(
                use_id=u1, evidence_id=e1, content=content1
            ),
            u2: store_coverage_use(
                use_id=u2, evidence_id=e2, content=content2
            ),
        },
    )
    runtime = graph_runtime(hydrator)
    result = await synthesize_answer(
        synthesis_input=_sufficient_input(u1, u2),
        runtime=runtime,
        plan=plan,
        bindings=binding_set(),
        budget=SynthesisRuntimeContext(
            max_evidence_items=10,
            max_total_chars=len(content1),
            max_total_tokens=10_000,
        ),
    )
    assert len(hydrator.persisted_records) == 1
    record = hydrator.persisted_records[0]
    assert isinstance(record.source, DerivedSourceIdentity)
    assert record.source.source_evidence_ids == (e2,)
    assert record.content == content2
    assert len(hydrator.minted_uses) == 1
    overflow_use = hydrator.minted_uses[0]
    assert overflow_use.purpose == "supporting"
    assert overflow_use.evidence_id == record.evidence_id
    assert overflow_use.use_id in result.admitted_use_ids
    assert result.draft.content == content1
    assert result.draft.claims[0].evidence_use_ids == (u1,)
    # The overflow-derived use is admitted for governance, not drafted:
    # citing it would re-violate the budget the persist just honored.
    for claim in result.draft.claims:
        assert overflow_use.use_id not in claim.evidence_use_ids


@pytest.mark.asyncio
async def test_overflow_lease_acquired_and_committed_by_synthesize_node() -> None:
    # The node passes the generous DEFAULT_SYNTHESIS_BUDGET, so the tail
    # must exceed it for overflow to trigger on the node path.
    from app.services.agents.v2.nodes.synthesize import DEFAULT_SYNTHESIS_BUDGET

    plan = read_plan()
    u1, u2 = uuid4(), uuid4()
    content1 = "Điều 5 quy định mức phạt."
    content2 = "Điều 6 quy định. " * (
        DEFAULT_SYNTHESIS_BUDGET.max_total_chars // len("Điều 6 quy định. ") + 1
    )
    hydrator = FakeHydrator(
        {
            u1: store_coverage_use(use_id=u1, evidence_id=uuid4(), content=content1),
            u2: store_coverage_use(use_id=u2, evidence_id=uuid4(), content=content2),
        },
    )
    events: list[str] = []
    leases = FakeLeaseRepo(events)
    evaluation = EvidenceEvaluation(
        status="sufficient", coverage=Coverage(items=()), missing=(), contradictions=()
    )
    state = make_state(
        plan=plan,
        bindings=binding_set(),
        results=(read_result(use_id=u1), read_result(use_id=u2)),
        evaluation=evaluation,
    )
    update = await synthesize_node(state, graph_runtime(hydrator, leases))
    assert update == {}
    assert len(hydrator.persisted_records) == 1
    assert "commit" in events
    assert any(
        call[0] == "run-1" and call[1] is None and call[2] is not None
        for call in leases.calls
    )


@pytest.mark.asyncio
async def test_shared_synthesize_and_lease_leases_overflow() -> None:
    """I3: every path through the shared helper leaves fresh uses leased."""
    from app.services.agents.v2.nodes.synthesize import synthesize_and_lease

    plan = read_plan()
    u1, u2 = uuid4(), uuid4()
    content1 = "Điều 5 quy định mức phạt."
    hydrator = FakeHydrator(
        {
            u1: store_coverage_use(use_id=u1, evidence_id=uuid4(), content=content1),
            u2: store_coverage_use(
                use_id=u2, evidence_id=uuid4(), content="Điều 6 quy định."
            ),
        },
    )
    events: list[str] = []
    leases = FakeLeaseRepo(events)
    runtime = graph_runtime(hydrator, leases)
    result = await synthesize_and_lease(
        synthesis_input=_sufficient_input(u1, u2),
        runtime=runtime,
        plan=plan,
        bindings=binding_set(),
        budget=SynthesisRuntimeContext(
            max_evidence_items=10,
            max_total_chars=len(content1),
            max_total_tokens=10_000,
        ),
    )
    assert result.draft.content == content1
    assert "commit" in events
    assert any(
        call[0] == "run-1" and call[1] is None and call[2] is not None
        for call in leases.calls
    )


@pytest.mark.asyncio
async def test_synthesis_reuse_revalidates_current_uses() -> None:
    """A use admitted once is excluded once expired: no stale citation."""
    plan = read_plan()
    u1, u2 = uuid4(), uuid4()
    hydrator = FakeHydrator(
        {
            u1: store_coverage_use(
                use_id=u1, evidence_id=uuid4(), content="Điều 5 quy định."
            ),
            u2: store_coverage_use(
                use_id=u2, evidence_id=uuid4(), content="Điều 6 quy định."
            ),
        },
    )
    runtime = graph_runtime(hydrator)
    first = await synthesize_answer(
        synthesis_input=_sufficient_input(u1, u2),
        runtime=runtime,
        plan=plan,
        bindings=binding_set(),
        budget=generous_budget(),
    )
    assert first.admitted_use_ids == frozenset({u1, u2})

    hydrator.store[u2].expired = True
    second = await synthesize_answer(
        synthesis_input=_sufficient_input(u1, u2),
        runtime=runtime,
        plan=plan,
        bindings=binding_set(),
        budget=generous_budget(),
    )
    assert second.admitted_use_ids == frozenset({u1})
    assert (u2, "expired") in hydrator.denied
    for claim in second.draft.claims:
        assert u2 not in claim.evidence_use_ids


@pytest.mark.asyncio
async def test_synthesize_node_rejects_missing_evaluation() -> None:
    state = make_state(plan=read_plan(), bindings=binding_set())
    with pytest.raises(SynthesisError, match="sufficient"):
        await synthesize_node(state, graph_runtime())


@pytest.mark.asyncio
async def test_apply_budget_split_is_deterministic() -> None:
    assert estimate_tokens_for_chars(0) == 0
    assert estimate_tokens_for_chars(4) == 1
    assert estimate_tokens_for_chars(5) == 2
    items = (
        HydratedEvidence(
            use_id=uuid4(), evidence_id=uuid4(), task_id="T1",
            purpose="coverage", target_id="t1", content="a" * 10,
            role="target", source_label="s",
            classification="normal", locator=None,
        ),
        HydratedEvidence(
            use_id=uuid4(), evidence_id=uuid4(), task_id="T1",
            purpose="coverage", target_id="t1", content="b" * 10,
            role="target", source_label="s",
            classification="normal", locator=None,
        ),
    )
    budget = SynthesisRuntimeContext(
        max_evidence_items=1, max_total_chars=1000, max_total_tokens=1000
    )
    head, tail = apply_budget_split(items, budget)
    assert len(head) == 1 and len(tail) == 1
    head2, tail2 = apply_budget_split(items, budget)
    assert (head, tail) == (head2, tail2)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def _grounded_fixture() -> tuple[AnswerDraft, list[HydratedEvidence]]:
    u1, u2 = uuid4(), uuid4()
    e1, e2 = uuid4(), uuid4()
    draft = AnswerDraft(
        content="Điều 5 quy định mức phạt. Điều 6 quy định thẩm quyền.",
        claims=(
            AnswerClaim(
                claim_id="claim-1",
                text="Điều 5 quy định mức phạt.",
                evidence_use_ids=(u1,),
            ),
            AnswerClaim(
                claim_id="claim-2",
                text="Điều 6 quy định thẩm quyền.",
                evidence_use_ids=(u2,),
            ),
        ),
    )
    evidence = [
        HydratedEvidence(
            use_id=u1, evidence_id=e1, task_id="T1", purpose="coverage",
            target_id="t1", content="Điều 5 quy định mức phạt.",
            role="target", source_label="doc-A",
            classification="normal",
            locator=DocumentLocator(kind="document"),
            document_revision=REVISION,
        ),
        HydratedEvidence(
            use_id=u2, evidence_id=e2, task_id="T1", purpose="coverage",
            target_id="t1", content="Điều 6 quy định thẩm quyền.",
            role="target", source_label="doc-A",
            classification="normal",
            locator=DocumentLocator(kind="document"),
            document_revision=REVISION,
        ),
    ]
    return draft, evidence


@pytest.mark.asyncio
async def test_grounded_answer_maps_every_assertion() -> None:
    draft, evidence = _grounded_fixture()
    grounded = await ground_answer(draft=draft, evidence=tuple(evidence))
    assert grounded.draft == draft
    assert grounded.admitted_use_ids == frozenset(
        {evidence[0].use_id, evidence[1].use_id}
    )
    assert [c.citation_id for c in grounded.citations] == ["cite-1", "cite-2"]
    assert [c.evidence_id for c in grounded.citations] == [
        evidence[0].evidence_id,
        evidence[1].evidence_id,
    ]


@pytest.mark.asyncio
async def test_unmapped_assertion_revises_once_then_insufficient() -> None:
    draft, evidence = _grounded_fixture()
    bad = AnswerDraft(
        content=draft.content + " Câu này không có căn cứ.",
        claims=draft.claims,
    )
    calls: list[AnswerDraft] = []

    async def revise(draft_in: AnswerDraft, report: Any) -> AnswerDraft:
        calls.append(draft_in)
        assert len(report.unmapped_assertions) == 1
        return draft_in  # still unmapped

    with pytest.raises(GroundingInsufficient) as exc_info:
        await ground_answer(draft=bad, evidence=tuple(evidence), revise=revise)
    assert len(calls) == 1
    assert len(exc_info.value.report.unmapped_assertions) == 1


@pytest.mark.asyncio
async def test_successful_revision_grounds_answer() -> None:
    draft, evidence = _grounded_fixture()
    bad = AnswerDraft(
        content=draft.content + " Câu này không có căn cứ.",
        claims=draft.claims,
    )
    fixed_claim = AnswerClaim(
        claim_id="claim-3",
        text="Câu này không có căn cứ.",
        evidence_use_ids=(evidence[0].use_id,),
    )

    async def revise(draft_in: AnswerDraft, report: Any) -> AnswerDraft:
        return AnswerDraft(
            content=draft_in.content, claims=draft_in.claims + (fixed_claim,)
        )

    grounded = await ground_answer(
        draft=bad, evidence=tuple(evidence), revise=revise
    )
    # Citations are per-evidence: the repaired claim reuses cite-1.
    assert [c.citation_id for c in grounded.citations] == ["cite-1", "cite-2"]
    assert grounded.draft.claims[-1].claim_id == "claim-3"


@pytest.mark.asyncio
async def test_ground_rejects_claim_outside_admitted_set() -> None:
    draft, evidence = _grounded_fixture()
    outsider = AnswerDraft(
        content="Điều 5 quy định mức phạt.",
        claims=(
            AnswerClaim(
                claim_id="claim-1",
                text="Điều 5 quy định mức phạt.",
                evidence_use_ids=(uuid4(),),
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="admitted"):
        await ground_answer(draft=outsider, evidence=tuple(evidence))


@pytest.mark.asyncio
async def test_ambiguous_assertion_needs_revision() -> None:
    u1, u2 = uuid4(), uuid4()
    draft = AnswerDraft(
        content="Mức phạt áp dụng.",
        claims=(
            AnswerClaim(
                claim_id="claim-1",
                text="Mức phạt áp dụng cho điều 5.",
                evidence_use_ids=(u1,),
            ),
            AnswerClaim(
                claim_id="claim-2",
                text="Mức phạt áp dụng cho điều 6.",
                evidence_use_ids=(u2,),
            ),
        ),
    )
    evidence = [
        HydratedEvidence(
            use_id=u1, evidence_id=uuid4(), task_id="T1", purpose="supporting",
            target_id=None, content="x", role=None,
            source_label="s", classification="normal", locator=None,
        ),
        HydratedEvidence(
            use_id=u2, evidence_id=uuid4(), task_id="T1", purpose="supporting",
            target_id=None, content="y", role=None,
            source_label="s", classification="normal", locator=None,
        ),
    ]
    with pytest.raises(GroundingInsufficient) as exc_info:
        await ground_answer(draft=draft, evidence=tuple(evidence))
    assert len(exc_info.value.report.ambiguous_assertions) == 1


@pytest.mark.asyncio
async def test_citations_are_deterministic_and_claim_free() -> None:
    draft, evidence = _grounded_fixture()
    first = render_citations(draft, tuple(evidence))
    second = render_citations(draft, tuple(evidence))
    assert first == second
    assert [c.citation_id for c in first] == ["cite-1", "cite-2"]
    from app.services.agents.v2.contracts.response import RenderedCitation

    assert set(RenderedCitation.model_fields) == {
        "citation_id",
        "evidence_id",
        "label",
    }
    assert [c.label for c in first] == ["doc-A", "doc-A"]


@pytest.mark.asyncio
async def test_empty_budget_head_drafts_from_derived_only() -> None:
    """Nothing fits the budget: the draft falls back to the derived item."""
    plan = read_plan()
    u1 = uuid4()
    hydrator = FakeHydrator(
        {u1: store_coverage_use(use_id=u1, evidence_id=uuid4())},
    )
    result = await synthesize_answer(
        synthesis_input=_sufficient_input(u1),
        runtime=graph_runtime(hydrator),
        plan=plan,
        bindings=binding_set(),
        budget=SynthesisRuntimeContext(
            max_evidence_items=0, max_total_chars=0, max_total_tokens=0
        ),
    )
    assert len(hydrator.persisted_records) == 1
    derived_use = hydrator.minted_uses[0].use_id
    assert result.draft.claims[0].evidence_use_ids == (derived_use,)


@pytest.mark.asyncio
async def test_ground_node_consumes_channel_without_resynthesis() -> None:
    """C2: ground consumes the stored draft; the hydrator runs synthesis once."""
    plan = read_plan()
    u1 = uuid4()
    hydrator = FakeHydrator(
        {
            u1: store_coverage_use(
                use_id=u1, evidence_id=uuid4(), content="Điều 5 quy định."
            )
        },
    )
    channel = AnswerDraftChannel()
    evaluation = EvidenceEvaluation(
        status="sufficient", coverage=Coverage(items=()), missing=(), contradictions=()
    )
    state = make_state(
        plan=plan,
        bindings=binding_set(),
        results=(read_result(use_id=u1),),
        evaluation=evaluation,
    )
    runtime = graph_runtime(hydrator, FakeLeaseRepo([]), channel)
    assert await synthesize_node(state, runtime) == {}
    synth_calls = [
        call for call in hydrator.hydrate_calls if call[0] == "synthesis"
    ]
    assert len(synth_calls) == 1
    assert await ground_node(state, runtime) == {}
    assert [
        call for call in hydrator.hydrate_calls if call[0] == "synthesis"
    ] == synth_calls
    assert channel.get("run-1") is not None
    assert channel.get("run-1").grounded_draft is not None


@pytest.mark.asyncio
async def test_ground_node_miss_rederives_once() -> None:
    """C2: an empty channel re-derives exactly once (no per-node re-run)."""
    plan = read_plan()
    u1 = uuid4()
    hydrator = FakeHydrator(
        {
            u1: store_coverage_use(
                use_id=u1, evidence_id=uuid4(), content="Điều 5 quy định."
            ),
        },
    )
    evaluation = EvidenceEvaluation(
        status="sufficient", coverage=Coverage(items=()), missing=(), contradictions=()
    )
    state = make_state(
        plan=plan,
        bindings=binding_set(),
        results=(read_result(use_id=u1),),
        evaluation=evaluation,
    )
    runtime = graph_runtime(hydrator, FakeLeaseRepo([]), AnswerDraftChannel())
    assert await ground_node(state, runtime) == {}
    assert len([c for c in hydrator.hydrate_calls if c[0] == "synthesis"]) == 1
    assert channel_grounded(runtime) is not None


def channel_grounded(runtime: GraphRuntimeContext) -> Any:
    channel = runtime.services.answer_draft_channel
    assert channel is not None
    return channel.get("run-1").grounded_draft  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_finalizer_turns_synthesis_failure_into_typed_response() -> None:
    """C1: sufficient verdict + empty admission → typed insufficient, no raise."""
    plan = people_plan()
    evaluation = EvidenceEvaluation(
        status="sufficient", coverage=Coverage(items=()), missing=(), contradictions=()
    )
    route = RouteDecision(route="fast_domain", reason_code="simple_people_lookup")
    state = make_state(
        plan=plan,
        bindings=binding_set(),
        results=(people_result(status="success"),),
        evaluation=evaluation,
        route=route,
    )
    final = await finalizer_node(
        state, graph_runtime(FakeHydrator({}), FakeLeaseRepo([]), AnswerDraftChannel())
    )
    assert final["final_response"].status == "insufficient"
    assert "t1" not in final["final_response"].content
