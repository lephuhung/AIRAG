"""Bounded append-only replan + discovery + research loop (Phase 3, Task 5).

Proves the canonical loop ``plan -> validate_checkpoint -> execute ->
evaluate -> decide -> (replan -> validate_checkpoint)...`` with every replan
checkpointed before the next execute, completed tasks never rerun, budgets
enforced, discovery UUID/pin/promotion semantics, the framework-neutral
summarize skill (no summary agent), and the subagent advisory boundary.
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.binding import (
    BindingPromotionRequest,
    DocumentBindingSet,
    DocumentDiscoveryCandidate,
    ScopedDocument,
)
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
    DocumentSearchInput,
    DocumentSearchOutput,
    KnowledgeGraphInput,
    PeopleLookupInput,
    PeopleLookupOutput,
    SectionReadInput,
    SectionReadOutput,
)
from app.services.agents.v2.contracts.evaluation import (
    Contradiction,
    Coverage,
    CoverageObservation,
    EvidenceEvaluation,
)
from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity, EvidenceUseRef
from app.services.agents.v2.contracts.execution import (
    AgentRequest,
    AgentResult,
    TaskExecutionSummary,
)
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    DiscoveryPolicy,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    ResearchBudgetView,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_replan,
)
from app.services.agents.v2.dependencies.people_document import (
    PeopleDocumentMaterialization,
)
from app.services.agents.v2.tools.gateway import (
    AgentToolGateway,
    CapabilityInvocationProposal,
    UnplannedCapabilityDispatch,
    require_planned_dispatch,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
COMPARE_QUERY = "So sánh Chương II A và Chương III B"


class FakeReadCapability:
    """Atomic stub: one bounded read per planned target."""

    def __init__(
        self,
        name: str = "document.read",
        *,
        missing_targets: frozenset[str] = frozenset(),
        fail_first_for: frozenset[str] = frozenset(),
        truncate_first_for: frozenset[str] = frozenset(),
        locator_for: dict[str, object] | None = None,
    ) -> None:
        domain = "section" if name == "section.read" else "document"
        self.descriptor = CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []
        self.missing_targets = missing_targets
        self.fail_first_for = fail_first_for
        self.truncate_first_for = truncate_first_for
        self._locator_for = locator_for or {}
        self._failed_once: set[str] = set()
        self._truncated_once: set[str] = set()
        self.uses: dict[UUID, tuple[str, str]] = {}

    def _locator(self, target_id: str):  # type: ignore[no-untyped-def]
        return self._locator_for.get(
            target_id, DocumentLocator(kind="document")
        )

    def _output(self, read_unit_count: int):  # type: ignore[no-untyped-def]
        if self.descriptor.name == "section.read":
            return SectionReadOutput(
                kind="section.read", read_unit_count=read_unit_count
            )
        return DocumentReadOutput(
            kind="document.read", read_unit_count=read_unit_count
        )

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        target_id = request.input.target_ids[0]  # type: ignore[union-attr]
        if target_id in self.fail_first_for and target_id not in self._failed_once:
            self._failed_once.add(target_id)
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=0),
                evidence_uses=(),
                coverage_observations=(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    ),
                ),
                error=None,
            )
        if (
            target_id in self.truncate_first_for
            and target_id not in self._truncated_once
        ):
            self._truncated_once.add(target_id)
            use_id = uuid4()
            self.uses[use_id] = (request.task_id, target_id)
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=1),
                evidence_uses=(EvidenceUseRef(use_id=use_id),),
                coverage_observations=(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="truncated",
                    ),
                ),
                error=None,
            )
        if target_id in self.missing_targets:
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=self._output(0),
                evidence_uses=(),
                coverage_observations=(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    ),
                ),
                error=None,
            )
        use_id = uuid4()
        self.uses[use_id] = (request.task_id, target_id)
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="success",
            data=self._output(1),
            evidence_uses=(EvidenceUseRef(use_id=use_id),),
            coverage_observations=(
                CoverageObservation(
                    target_id=target_id,
                    observed_locators=(self._locator(target_id),),
                    outcome="read",
                ),
            ),
            error=None,
        )


class FakeLeases:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, object, object]] = []
        self.session = self._Session()

    class _Session:
        def __init__(self) -> None:
            self.commits = 0

        async def commit(self) -> None:
            self.commits += 1

    async def acquire_or_refresh(
        self, run_id: str, revision_id: object = None, evidence_use_id: object = None
    ) -> object:
        self.acquired.append((run_id, revision_id, evidence_use_id))
        return {"run_id": run_id, "revision": revision_id, "use": evidence_use_id}


class FakeHydrator:
    def __init__(self, capabilities: tuple[FakeReadCapability, ...]) -> None:
        self._capabilities = capabilities

    def _use_owners(self) -> dict[UUID, tuple[str, str]]:
        owners: dict[UUID, tuple[str, str]] = {}
        for capability in self._capabilities:
            owners.update(capability.uses)
        return owners

    async def hydrate_for_evaluation(
        self, use_refs, *, runtime, plan, bindings
    ):  # type: ignore[no-untyped-def]
        from app.services.agents.v2.nodes.evaluate import HydratedEvidence

        binding_by_target = {
            unit.target_id: next(
                binding
                for binding in bindings.bindings
                if binding.binding_id == unit.binding_id
            )
            for unit in plan.target_units
        }
        locator_by_target = {
            unit.target_id: unit.requested_locator for unit in plan.target_units
        }
        use_owners = self._use_owners()
        admitted = []
        for ref in use_refs:
            owner = use_owners.get(ref.use_id)
            if owner is None:
                continue
            task_id, target_id = owner
            binding = binding_by_target[target_id]
            admitted.append(
                HydratedEvidence(
                    use_id=ref.use_id,
                    evidence_id=uuid4(),
                    task_id=task_id,
                    purpose="coverage",
                    target_id=target_id,
                    content=f"content of {target_id}",
                    role=binding.role,
                    source_label=target_id,
                    source_identity=DocumentSourceIdentity(
                        kind="document",
                        document_id=binding.document_id,
                        document_revision=binding.document_revision,
                        locator=locator_by_target[target_id],
                    ),
                    classification="normal",
                    locator=locator_by_target[target_id],
                    document_revision=binding.document_revision,
                )
            )
        return tuple(admitted)

    async def hydrate_for_synthesis(
        self, use_refs, *, runtime, plan, bindings, budget
    ):  # type: ignore[no-untyped-def]
        return await self.hydrate_for_evaluation(
            use_refs, runtime=runtime, plan=plan, bindings=bindings
        )


def _semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query=COMPARE_QUERY,
        normalized_query=COMPARE_QUERY.lower(),
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _bindings() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
            ScopedDocument(
                binding_id="b2",
                document_id=DOC_B,
                document_revision=str(REV_B),
                role="reference",
            ),
        ),
        revision_requirement_refs=(),
    )


def _analysis(work_type: str = "compare") -> QueryAnalysis:
    return QueryAnalysis(work_type=work_type, domains=("document",))  # type: ignore[arg-type]


def _read_task(
    task_id: str = "T1",
    target: str = "t1",
    capability: str = "document.read",
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability=capability,
        task_objective=f"Read {target}",
        input=DocumentReadInput(kind="document.read", target_ids=(target,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _two_target_plan() -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="p1",
        goal=COMPARE_QUERY,
        target_units=(
            TargetUnit(
                target_id="t1",
                binding_id="b1",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            ),
            TargetUnit(
                target_id="t2",
                binding_id="b2",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            ),
        ),
        tasks=(_read_task("T1", "t1"), _read_task("T2", "t2")),
    )


def _harness(
    run_id: str = "run-replan-1",
    missing: frozenset[str] = frozenset(),
    allowed: frozenset[str] | None = None,
    fail_first_for: frozenset[str] = frozenset(),
    truncate_first_for: frozenset[str] = frozenset(),
    locator_for: dict[str, object] | None = None,
) -> tuple[tuple[FakeReadCapability, ...], FakeLeases, GraphRuntimeContext]:
    capabilities = (
        FakeReadCapability(
            "document.read",
            missing_targets=missing,
            fail_first_for=fail_first_for,
            truncate_first_for=truncate_first_for,
            locator_for=locator_for,
        ),
        FakeReadCapability(
            "section.read",
            missing_targets=missing,
            fail_first_for=fail_first_for,
            truncate_first_for=truncate_first_for,
            locator_for=locator_for,
        ),
    )
    leases = FakeLeases()
    runtime = CapabilityRuntimeContext(
        request_id="req-replan-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=(
            allowed
            if allowed is not None
            else frozenset({"document.read", "section.read"})
        ),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities],
        runtime,
    )
    return (
        capabilities,
        leases,
        GraphRuntimeContext(
            capability_runtime=runtime,
            services=RuntimeServices(
                capability_registry=registry,
                retention_leases=leases,
                evidence_hydrator=FakeHydrator(capabilities),
                pinned_target_resolver=_fresh_pinned_resolver(),
            ),
        ),
    )


def _fresh_pinned_resolver():
    """Request-scoped resolver mirror: production ingress wires one for the
    shared scheduler feed (round 2 N1: targeted plans raise without one)."""
    from app.services.agent.runtime_selector import PlanBindingResolver

    return PlanBindingResolver()


DOC_C = UUID("33333333-3333-3333-3333-333333333333")
DOC_D = UUID("44444444-4444-4444-4444-444444444444")
REV_C = UUID("c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3")
REV_D = UUID("d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4")
CANDIDATE_C = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
CANDIDATE_D = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


class FakeSearchCapability:
    """Atomic stub: discovery search returning fixed UUID candidates."""

    def __init__(self) -> None:
        self.descriptor = CapabilityDescriptor(
            name="document.search",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="search",
            supports_parallel=False,
        )
        self.calls: list[tuple[AgentRequest, object]] = []

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        assert isinstance(request.input, DocumentSearchInput)
        assert request.input.person_identifier is None
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="success",
            data=DocumentSearchOutput(
                kind="document.search",
                candidates=(
                    DocumentDiscoveryCandidate(
                        candidate_id=CANDIDATE_C,
                        document_id=DOC_C,
                        document_revision=str(REV_C),
                    ),
                    DocumentDiscoveryCandidate(
                        candidate_id=CANDIDATE_D,
                        document_id=DOC_D,
                        document_revision=str(REV_D),
                    ),
                ),
            ),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )


class FakePeopleCapability:
    """Atomic stub: people lookup returning a scripted terminal outcome."""

    def __init__(
        self, *, status: str = "not_found", error_code: str | None = None
    ) -> None:
        self.descriptor = CapabilityDescriptor(
            name="people.lookup",  # type: ignore[arg-type]
            domain="people",  # type: ignore[arg-type]
            operation_type="lookup",
            supports_parallel=False,
        )
        self._status = status
        self._error_code = error_code
        self.calls: list[tuple[AgentRequest, object]] = []

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        assert isinstance(request.input, PeopleLookupInput)
        if self._status == "not_found":
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="not_found",
                data=PeopleLookupOutput(kind="people.lookup", matched=False),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        assert self._error_code is not None
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="error",
            data=None,
            evidence_uses=(),
            coverage_observations=(),
            error={"code": self._error_code, "message": "upstream slow", "retryable": True},  # type: ignore[arg-type]
        )


class FakeDraftChannel:
    """Framework handoff double: records the deterministic reduce draft."""

    def __init__(self) -> None:
        self.stored: list[tuple[str, object, tuple]] = []

    def store_draft(self, run_id: str, *, draft, evidence) -> None:  # type: ignore[no-untyped-def]
        self.stored.append((run_id, draft, tuple(evidence)))


def _people_harness(
    run_id: str,
    people: FakePeopleCapability,
    *,
    with_resolver: bool = True,
) -> tuple[FakePeopleCapability, FakeSearchCapability, FakeLeases, FakeBindingResolver | None, FakeDraftChannel, GraphRuntimeContext]:
    document_capability = FakeReadCapability("document.read")
    search_capability = FakeSearchCapability()
    channel = FakeDraftChannel()
    leases = FakeLeases()
    resolver = FakeBindingResolver(frozenset({DOC_C, DOC_D})) if with_resolver else None
    runtime = CapabilityRuntimeContext(
        request_id="req-people-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=True,
        allowed_capabilities=frozenset(
            {"people.lookup", "document.search", "document.read"}
        ),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [
            CapabilityRegistration(capability=people),
            CapabilityRegistration(capability=document_capability),
            CapabilityRegistration(capability=search_capability),
        ],
        runtime,
    )
    return (
        people,
        search_capability,
        leases,
        resolver,
        channel,
        GraphRuntimeContext(
            capability_runtime=runtime,
            services=RuntimeServices(
                capability_registry=registry,
                retention_leases=leases,
                evidence_hydrator=FakeHydrator((document_capability,)),
                binding_resolver=resolver,
                answer_draft_channel=channel,
                pinned_target_resolver=_fresh_pinned_resolver(),
            ),
        ),
    )


def _person_semantic() -> SemanticContext:
    from app.services.agents.v2.contracts.conversation import EntityReference

    return SemanticContext(
        contextualized_query="CCCD cua A xuat hien trong nghi dinh nao",
        normalized_query="cccd cua a xuat hien trong nghi dinh nao",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(
            EntityReference(ref_id="p1", kind="person", label="A"),
        ),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _cross_domain_analysis() -> QueryAnalysis:
    return QueryAnalysis(
        work_type="cross_domain",  # type: ignore[arg-type]
        domains=("people", "document"),  # type: ignore[arg-type]
    )


class FakeBindingResolver:
    """Binding Resolver double: it alone creates/pins discovered bindings.

    Revalidates current visibility (ACL) against its server-side set and pins
    the candidate's exact discovered revision. Raises ``DiscoveryDenied``
    for invisible documents instead of leaking metadata.
    """

    def __init__(self, visible: frozenset[UUID]) -> None:
        from app.services.agents.v2.discovery import DiscoveryDenied

        self._visible = visible
        self._denied = DiscoveryDenied
        self.created: list[tuple[object, object, ScopedDocument]] = []

    async def add_discovered_binding(
        self, request, candidate, capability_runtime
    ):  # type: ignore[no-untyped-def]
        if candidate.document_id not in self._visible:
            raise self._denied(
                "the discovered document is not visible under current authorization"
            )
        binding = ScopedDocument(
            binding_id=f"d_{candidate.candidate_id.hex[:8]}",
            document_id=candidate.document_id,
            document_revision=candidate.document_revision,
            role=request.requested_role,
        )
        self.created.append((request, candidate, binding))
        return binding


def _discovery_harness(
    run_id: str = "run-discovery-1",
    visible: frozenset[UUID] | None = None,
    truncate_first_for: frozenset[str] = frozenset(),
) -> tuple[tuple[object, ...], FakeLeases, FakeBindingResolver, GraphRuntimeContext]:
    document_capability = FakeReadCapability(
        "document.read", truncate_first_for=truncate_first_for
    )
    search_capability = FakeSearchCapability()
    capabilities = (document_capability, search_capability)
    leases = FakeLeases()
    resolver = FakeBindingResolver(
        visible if visible is not None else frozenset({DOC_C, DOC_D})
    )
    runtime = CapabilityRuntimeContext(
        request_id="req-discovery-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=True,
        allowed_capabilities=frozenset(
            {"people.lookup", "document.search", "document.read"}
        ),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities],
        runtime,
    )
    return (
        capabilities,
        leases,
        resolver,
        GraphRuntimeContext(
            capability_runtime=runtime,
            services=RuntimeServices(
                capability_registry=registry,
                retention_leases=leases,
                evidence_hydrator=FakeHydrator((document_capability,)),
                binding_resolver=resolver,
                pinned_target_resolver=_fresh_pinned_resolver(),
            ),
        ),
    )


def _people_plan() -> TaskPlan:
    """Targetless people plan: T1 people.lookup with no TargetUnits."""
    return TaskPlan(
        contract_version="2.0",
        plan_id="p-people",
        goal="CCCD cua A xuat hien trong nghi dinh nao",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="T1",
                capability="people.lookup",
                task_objective="find A",
                input=PeopleLookupInput(kind="people.lookup", query="A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def _insufficient_evaluation() -> EvidenceEvaluation:
    return EvidenceEvaluation(
        status="insufficient",
        coverage=Coverage(items=()),
        missing=(),
        contradictions=(),
    )


def _not_found_people_result() -> AgentResult:
    return AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="not_found",
        data=PeopleLookupOutput(kind="people.lookup", matched=False),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )


def _timeout_people_result() -> AgentResult:
    return AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error={"code": "TIMEOUT", "message": "upstream slow", "retryable": True},  # type: ignore[arg-type]
    )


def _parent_state() -> dict:
    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.conversation import ConversationContext
    from app.services.agents.v2.contracts.request import RequestContext
    from app.services.agents.v2.contracts.routing import RouteDecision
    from app.services.agents.v2.contracts.state import ExecutionState

    return {
        "contract_version": CONTRACT_VERSION,
        "request": RequestContext(
            contract_version=CONTRACT_VERSION,
            request_id="req-replan-1",
            thread_id="thread-replan-1",
            original_query=COMPARE_QUERY,
            known_documents=(),
        ),
        "conversation": ConversationContext(
            summary="",
            active_entities=(),
            last_focus=None,
            recent_turns=(),
        ),
        "semantic": _semantic(),
        "bindings": _bindings(),
        "query_analysis": _analysis(),
        "route_decision": RouteDecision(
            route="complex_research", reason_code="comparison"
        ),
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None
        ),
        "clarification": None,
        "final_response": None,
    }


def _policy(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "allow_reference_discovery": False,
        "allow_supporting_discovery": False,
        "max_discovered_documents": 0,
    }
    values.update(overrides)
    return DiscoveryPolicy(**values)  # type: ignore[arg-type]


def _budget(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "max_tasks_remaining": 8,
        "max_replans_remaining": 2,
        "max_parallel_branches": 2,
    }
    values.update(overrides)
    return ResearchBudgetView(**values)  # type: ignore[arg-type]


@pytest.fixture
def replan_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the bounded replan loop: deployment settings allow 2 replans."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8, V2_MAX_PARALLEL_BRANCHES=2, V2_MAX_REPLANS=2
        ),
    )


@pytest.fixture
def discovery_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy-enabled discovery plus a bounded replan budget."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8,
            V2_MAX_PARALLEL_BRANCHES=2,
            V2_MAX_REPLANS=2,
            V2_ALLOW_REFERENCE_DISCOVERY=True,
            V2_ALLOW_SUPPORTING_DISCOVERY=True,
            V2_MAX_DISCOVERED_DOCUMENTS=4,
        ),
    )


def _child_input(**overrides):  # type: ignore[no-untyped-def]
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    values: dict = {
        "contract_version": "2.0",
        "semantic": _semantic(),
        "bindings": _bindings(),
        "query_analysis": _analysis(),
        # Task 7B fix (R73): the execute node derives the post-router
        # v1-fallback guard from the resolved routing — child inputs carry
        # the same supported verdict production threads from the router.
        "route_decision": RouteDecision(
            route="complex_research", reason_code="comparison"
        ),
        "plan": None,
        "task_results": (),
        "evaluation": None,
        "replans_remaining": 0,
    }
    values.update(overrides)
    return ComplexResearchState(**values)  # type: ignore[arg-type]


def _appended(
    current: TaskPlan,
    task_id: str = "T3",
    capability: str = "document.read",
    target: str = "t2",
    reason: str = "coverage gap for target t2",
) -> TaskPlan:
    return current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id=task_id,
                    capability=capability,
                    task_objective=f"Re-read {target}",
                    input=DocumentReadInput(
                        kind="document.read", target_ids=(target,)
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason=reason,
                        task_ids=("T2",),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )


# ---------------------------------------------------------------------------
# Replan validation: thin runtime wrapper over the frozen validator (R31)
# ---------------------------------------------------------------------------


def test_replan_can_add_tasks_but_cannot_widen_authorization() -> None:
    from app.services.agents.v2.replanning import ReplanRejected, validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    from app.services.agents.v2.complex_research_graph import (
        build_task_execution_summaries,
    )

    outcomes = build_task_execution_summaries(())

    accepted = validate_runtime_replan(
        current, _appended(current), outcomes, _policy(), _budget(), context
    )
    assert [task.task_id for task in accepted.tasks] == ["T1", "T2", "T3"]

    # Widening authorization: rewriting a completed task's capability fails.
    widened = current.model_copy(
        update={
            "tasks": (
                current.tasks[0].model_copy(update={"capability": "people.lookup"}),
                current.tasks[1],
            )
            + current.tasks[2:]
        }
    )
    # (no new task appended either: prefix itself is rewritten)
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current, widened, outcomes, _policy(), _budget(), context
        )

    # A capability outside the CURRENT runtime catalog cannot be appended,
    # even though the frozen shape rules would otherwise accept it.
    kg_append = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T3",
                    capability="knowledge_graph.query",
                    task_objective="KG lookup",
                    input=KnowledgeGraphInput(
                        kind="knowledge_graph.query", query="A"
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason="extra",
                        task_ids=(),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ReplanRejected):
        validate_runtime_replan(
            current, kg_append, outcomes, _policy(), _budget(), context
        )


def test_replan_task_budget_exhausted() -> None:
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current,
            _appended(current),
            (),
            _policy(),
            _budget(max_tasks_remaining=0),
            context,
        )


def test_replan_budget_zero_replans_remaining() -> None:
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current,
            _appended(current),
            (),
            _policy(),
            _budget(max_replans_remaining=0),
            context,
        )


def test_replan_parallel_branch_bound() -> None:
    from app.services.agents.v2.replanning import ReplanRejected, validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    wide = current
    for index, target in (("T3", "t1"), ("T4", "t2"), ("T5", "t1")):
        wide = wide.model_copy(
            update={
                "tasks": wide.tasks
                + (
                    TaskSpec(
                        task_id=index,
                        capability="document.read",
                        task_objective=f"Re-read {target}",
                        input=DocumentReadInput(
                            kind="document.read", target_ids=(target,)
                        ),
                        depends_on=(),
                        origin=ReplanTaskOrigin(
                            kind="replan",
                            reason="fan out",
                            task_ids=(),
                            evidence_use_ids=(),
                        ),
                    ),
                )
            }
        )
    with pytest.raises(ReplanRejected):
        validate_runtime_replan(
            current, wide, (), _policy(), _budget(max_parallel_branches=2), context
        )


def test_completed_task_immutability() -> None:
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()

    rewritten_objective = current.model_copy(
        update={
            "tasks": (
                current.tasks[0].model_copy(
                    update={"task_objective": "changed after completion"}
                ),
                current.tasks[1],
                _appended(current).tasks[2],
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current, rewritten_objective, (), _policy(), _budget(), context
        )

    dropped = _appended(current).model_copy(
        update={"tasks": (_appended(current).tasks[1], _appended(current).tasks[2])}
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(current, dropped, (), _policy(), _budget(), context)

    retargeted = _appended(current).model_copy(
        update={"target_units": _appended(current).target_units[:1]}
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(current, retargeted, (), _policy(), _budget(), context)

    renamed = _appended(current).model_copy(
        update={"plan_id": "other", "goal": "other goal"}
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(current, renamed, (), _policy(), _budget(), context)


@pytest.mark.asyncio
async def test_replan_origin_carries_evidence_use_trigger_lineage(replan_settings: None) -> None:
    """Evidence-triggered replan records evidence_use_ids (spec §26)."""
    from app.services.agents.v2.complex_research_graph import (
        build_replan_proposal,
        complex_evaluate_node,
        complex_execute_node,
        validate_checkpoint_node,
    )

    _, _, context = _harness(
        run_id="run-lineage-1", missing=frozenset({"t2"})
    )

    # First pass only: drive validate/execute/evaluate manually through nodes.
    child = _child_input(replans_remaining=2)
    child = {**child, **await validate_checkpoint_node(child, context)}
    child = {**child, **await complex_execute_node(child, context)}
    child = {**child, **await complex_evaluate_node(child, context)}
    assert child["evaluation"].status == "insufficient"

    proposal = build_replan_proposal(child, context)
    assert proposal is not None
    # R50: build_replan_proposal returns a PROPOSAL (new tasks), never an
    # authoritative appended plan.
    new_tasks = proposal.new_tasks
    assert len(new_tasks) >= 1
    prior_use_ids = {
        ref.use_id for result in child["task_results"] for ref in result.evidence_uses
    }
    assert prior_use_ids, "the successful T1 read produced an admitted use"
    for task in new_tasks:
        assert isinstance(task.origin, ReplanTaskOrigin)
        assert set(task.origin.evidence_use_ids) <= prior_use_ids
        assert set(task.origin.task_ids) <= {
            task.task_id for task in child["plan"].tasks
        }
    assert prior_use_ids & {
        use_id
        for task in new_tasks
        for use_id in task.origin.evidence_use_ids
    }, "the replan preserves the triggering-use context"


def test_no_evidence_not_found_replan_stays_distinguishable() -> None:
    """T1=not_found with no uses stays a replan input, never a rerun ( §26)."""
    from app.services.agents.v2.complex_research_graph import (
        build_task_execution_summaries,
    )
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    not_found_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="not_found",
        data=PeopleLookupOutput(kind="people.lookup", matched=False),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    outcomes = build_task_execution_summaries((not_found_result,))
    assert outcomes[0].status == "not_found"
    assert outcomes[0].error_code is None
    # The replan appends only new tasks; T1 is never rerun and no use exists.
    accepted = validate_runtime_replan(
        current, _appended(current), outcomes, _policy(), _budget(), context
    )
    assert [task.task_id for task in accepted.tasks] == ["T1", "T2", "T3"]


@pytest.mark.asyncio
async def test_timeout_stays_distinguishable_from_not_found() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_task_execution_summaries,
    )
    from app.services.agents.v2.dependencies.people_document import (
        materialize_person_dependency,
    )

    timeout_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error={"code": "TIMEOUT", "message": "upstream slow", "retryable": True},  # type: ignore[arg-type]
    )
    outcomes = build_task_execution_summaries((timeout_result,))
    assert outcomes[0].status == "error"
    assert outcomes[0].error_code == "TIMEOUT"

    _, _, context = _harness(run_id="run-timeout-1")
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=timeout_result,
        runtime=context,
        plan=TaskPlan(
            contract_version="2.0",
            plan_id="pp",
            goal="q",
            target_units=(),
            tasks=(
                TaskSpec(
                    task_id="T1",
                    capability="people.lookup",
                    task_objective="find A",
                    input=PeopleLookupInput(kind="people.lookup", query="A"),
                    depends_on=(),
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            ),
        ),
        bindings=_bindings(),
        query="nghi dinh nao neu CCCD cua A",
    )
    assert outcome.kind == "failed"
    assert outcome.error_code == "TIMEOUT"
    assert outcome.kind != "not_found"


# ---------------------------------------------------------------------------
# Canonical loop: checkpointed replan, completed tasks never rerun (R32)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_task_is_not_rerun(replan_settings: None) -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    # T2's first read is truncated (it owns admitted evidence, so the task
    # gate passes) while t2 coverage stays incomplete: the replan re-reads
    # t2 once and must never redispatch the completed T1/T2 tasks.
    (document_capability, _), _, context = _harness(
        run_id="run-norerun-1", truncate_first_for=frozenset({"t2"})
    )
    child = _child_input(replans_remaining=2)
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert output["evaluation"].status == "sufficient"
    assert len(output["plan"].tasks) == 3
    served = [call[0].task_id for call in document_capability.calls]
    assert served.count("T1") == 1
    assert served.count("T2") == 1
    assert served.count("T3") == 1


@pytest.mark.asyncio
async def test_replan_loop_terminates_on_terminal_or_typed_failure() -> None:
    from app.services.agents.v2.complex_research_graph import (
        _decide_branch,
        build_complex_research_subgraph,
    )

    # No budget left: decide finalizes even when insufficient.
    _, _, context = _harness(run_id="run-term-1", missing=frozenset({"t2"}))
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(replans_remaining=0), context=context
    )
    assert output["evaluation"].status == "insufficient"
    assert len(output["plan"].tasks) == 2
    assert _decide_branch(output) == "finalize"

    # Contradictory evaluations never replan: the evaluator owns conflicts.
    contradictory = _child_input(
        plan=_two_target_plan(),
        evaluation=EvidenceEvaluation(
            status="contradictory",
            coverage=Coverage(items=()),
            missing=(),
            contradictions=(
                Contradiction(
                    contradiction_id="c1",
                    claim_a="A",
                    claim_b="not A",
                    evidence_use_ids=(uuid4(),),
                ),
            ),
        ),
        replans_remaining=2,
    )
    assert _decide_branch(contradictory) == "finalize"

    # Unsupported work stays the typed unavailable boundary.
    unsupported = _child_input(query_analysis=_analysis("evaluate"))
    assert _decide_branch(unsupported) == "finalize"


def test_production_entry_budget_comes_from_settings(replan_settings: None) -> None:
    """R36: supervisor entry uses the settings-driven limits, not a constant.

    Routes through ``build_complex_research_state`` (the production
    supervisor-entry mapping) and proves a non-zero ``V2_MAX_REPLANS``
    yields a non-zero remaining replan budget. There is no ``MAX_REPLANS``
    constant left to pin: the single source of truth is the settings.
    """
    import app.services.agents.v2.complex_research_graph as graph_module

    assert not hasattr(graph_module, "MAX_REPLANS")
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
        build_research_budget_view,
    )

    child = build_complex_research_state(_parent_state())  # type: ignore[arg-type]
    assert child["replans_remaining"] == 2

    _, _, context = _harness(run_id="run-prod-entry")
    budget = build_research_budget_view(child, context)
    assert budget.max_replans_remaining == 2


@pytest.mark.asyncio
async def test_production_entry_allows_a_replan(replan_settings: None) -> None:
    """R36 end to end: production-mapped state actually replans once."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
        build_complex_research_subgraph,
    )

    (_, _), _, context = _harness(
        run_id="run-prod-entry-2", truncate_first_for=frozenset({"t2"})
    )
    child = build_complex_research_state(_parent_state())  # type: ignore[arg-type]
    assert child["replans_remaining"] == 2
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert [task.task_id for task in output["plan"].tasks] == ["T1", "T2", "T3"]
    assert output["evaluation"].status == "sufficient"


@pytest.mark.asyncio
async def test_not_found_recovery_replan_on_real_graph(
    discovery_settings: None,
) -> None:
    """R37/§26: no-evidence not_found stays actionable on the real graph.

    T1 people.lookup returns not_found with no EvidenceUse: the run must
    propose a recovery replan (fallback discovery search, no rerun of T1,
    no fabricated scalar) instead of finalizing as a silent success.
    """
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        replan_advisable,
    )

    capabilities, _, _, context = _discovery_harness(run_id="run-notfound-1")
    search_capability = capabilities[1]
    child = _child_input(
        plan=_people_plan(),
        task_results=(_not_found_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    assert replan_advisable(child) is True
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    tasks = output["plan"].tasks
    assert [task.task_id for task in tasks] == ["T1", "T2"]
    recovery = tasks[1]
    assert recovery.capability == "document.search"
    assert recovery.depends_on == ("T1",)
    assert recovery.input.person_identifier is None  # type: ignore[union-attr]
    assert recovery.input.query == _people_plan().goal  # type: ignore[union-attr]
    assert isinstance(recovery.origin, ReplanTaskOrigin)
    assert "not_found" in recovery.origin.reason
    assert "TIMEOUT" not in recovery.origin.reason
    assert recovery.origin.task_ids == ("T1",)
    # T1 was never redispatched: only the fallback search ran.
    assert [call[0].task_id for call in search_capability.calls] == ["T2"]
    # Not a silent success: the people task still owns no admitted use.
    assert output["evaluation"].status == "insufficient"


@pytest.mark.asyncio
async def test_timeout_recovery_is_distinct_and_fabricates_nothing(
    discovery_settings: None,
) -> None:
    """R37/§26: TIMEOUT is typed distinctly from not_found, same guarantees."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        replan_advisable,
    )

    capabilities, _, _, context = _discovery_harness(run_id="run-timeout-1")
    search_capability = capabilities[1]
    child = _child_input(
        plan=_people_plan(),
        task_results=(_timeout_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    assert replan_advisable(child) is True
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    tasks = output["plan"].tasks
    assert [task.task_id for task in tasks] == ["T1", "T2"]
    recovery = tasks[1]
    assert recovery.capability == "document.search"
    assert recovery.input.person_identifier is None  # type: ignore[union-attr]
    assert isinstance(recovery.origin, ReplanTaskOrigin)
    assert "TIMEOUT" in recovery.origin.reason
    assert "not_found" not in recovery.origin.reason
    assert [call[0].task_id for call in search_capability.calls] == ["T2"]
    assert output["evaluation"].status == "insufficient"


def test_denied_people_task_never_recovers() -> None:
    """R37: denial is terminal — no fallback work is proposed for it."""
    from app.services.agents.v2.complex_research_graph import replan_advisable

    denied = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="denied",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error={"code": "PERMISSION_DENIED", "message": "no", "retryable": False},  # type: ignore[arg-type]
    )
    child = _child_input(
        plan=_people_plan(),
        task_results=(denied,),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    assert replan_advisable(child) is False


@pytest.mark.asyncio
async def test_real_t1_not_found_drives_recovery(discovery_settings: None) -> None:
    """R46: production graph routes T1 not_found to a distinct recovery replan.

    Starts from the real initial entry with a stub registry, lets the
    compiled graph route plan → validate → execute → evaluate → decide →
    replan → validate naturally (asserted on the stream), and proves: T1
    dispatched exactly once, the real evaluate went insufficient, decide
    admitted the replan, T2 was appended and dispatched, no scalar was
    fabricated, and not_found stays distinct from TIMEOUT.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
        normalize_complex_state,
    )

    people, search, _, _, _, context = _people_harness(
        "run-real-nf-1", FakePeopleCapability(status="not_found")
    )
    child = _child_input(
        semantic=_person_semantic(),
        query_analysis=_cross_domain_analysis(),
        replans_remaining=2,
    )
    compiled = _build_complex_research_graph().compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "thread-real-nf-1"}}
    seen: list[str] = []
    node_updates: dict[str, list[dict]] = {}
    async for chunk in compiled.astream(child, config=config, context=context):
        for node_name, update in chunk.items():
            seen.append(node_name)
            node_updates.setdefault(node_name, []).append(update)
    assert seen[0] == "plan"
    assert "replan" in seen
    first_replan = seen.index("replan")
    assert "validate_checkpoint" in seen[first_replan:]

    first_execute = node_updates["execute"][0]
    assert [(r.task_id, r.status) for r in first_execute["task_results"]] == [
        ("T1", "not_found")
    ]
    first_evaluate = node_updates["evaluate"][0]
    assert first_evaluate["evaluation"].status == "insufficient"
    assert first_evaluate["evaluation"].missing == ()

    snapshot = await compiled.aget_state(config)
    # aget_state returns checkpoint-serde values: coerce exactly like a
    # checkpoint load before walking the aggregate.
    terminal = normalize_complex_state(dict(snapshot.values))
    assert [task.task_id for task in terminal["plan"].tasks] == ["T1", "T2"]
    recovery = terminal["plan"].tasks[1]
    assert recovery.capability == "document.search"
    assert recovery.input.person_identifier is None  # type: ignore[union-attr]
    assert recovery.input.query == terminal["plan"].goal  # type: ignore[union-attr]
    assert "not_found" in recovery.origin.reason  # type: ignore[union-attr]
    assert "TIMEOUT" not in recovery.origin.reason  # type: ignore[union-attr]
    assert recovery.origin.task_ids == ("T1",)  # type: ignore[union-attr]
    assert [call[0].task_id for call in people.calls] == ["T1"]
    assert [call[0].task_id for call in search.calls] == ["T2"]
    assert terminal["evaluation"].status == "insufficient"


@pytest.mark.asyncio
async def test_real_t1_timeout_drives_distinct_recovery(
    discovery_settings: None,
) -> None:
    """R46: production graph routes T1 TIMEOUT to a distinct recovery replan."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
        build_task_execution_summaries,
        normalize_complex_state,
    )

    people, search, _, _, _, context = _people_harness(
        "run-real-to-1", FakePeopleCapability(status="error", error_code="TIMEOUT")
    )
    child = _child_input(
        semantic=_person_semantic(),
        query_analysis=_cross_domain_analysis(),
        replans_remaining=2,
    )
    compiled = _build_complex_research_graph().compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "thread-real-to-1"}}
    seen: list[str] = []
    node_updates: dict[str, list[dict]] = {}
    async for chunk in compiled.astream(child, config=config, context=context):
        for node_name, update in chunk.items():
            seen.append(node_name)
            node_updates.setdefault(node_name, []).append(update)
    assert seen[0] == "plan"
    assert "replan" in seen

    first_execute = node_updates["execute"][0]
    assert [(r.task_id, r.status) for r in first_execute["task_results"]] == [
        ("T1", "error")
    ]
    outcomes = build_task_execution_summaries(first_execute["task_results"])
    assert outcomes[0].error_code == "TIMEOUT"
    first_evaluate = node_updates["evaluate"][0]
    assert first_evaluate["evaluation"].status == "insufficient"

    snapshot = await compiled.aget_state(config)
    # aget_state returns checkpoint-serde values: coerce exactly like a
    # checkpoint load before walking the aggregate.
    terminal = normalize_complex_state(dict(snapshot.values))
    assert [task.task_id for task in terminal["plan"].tasks] == ["T1", "T2"]
    recovery = terminal["plan"].tasks[1]
    assert recovery.capability == "document.search"
    assert recovery.input.person_identifier is None  # type: ignore[union-attr]
    assert "TIMEOUT" in recovery.origin.reason  # type: ignore[union-attr]
    assert "not_found" not in recovery.origin.reason  # type: ignore[union-attr]
    assert [call[0].task_id for call in people.calls] == ["T1"]
    assert [call[0].task_id for call in search.calls] == ["T2"]
    assert terminal["evaluation"].status == "insufficient"


def test_canonical_loop_edges_checkpoint_replan_before_execute() -> None:
    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
    )

    graph = _build_complex_research_graph().compile()
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("plan", "validate_checkpoint") in edges
    assert ("validate_checkpoint", "execute") in edges
    assert ("execute", "materialize") in edges
    assert ("settle", "evaluate") in edges
    assert ("evaluate", "reduce") in edges
    assert ("reduce", "decide") in edges
    assert ("replan", "validate_checkpoint") in edges
    assert ("finalize", "__end__") in edges


@pytest.mark.asyncio
async def test_tool_gateway_checkpoints_replan_before_dispatch(
    replan_settings: None,
) -> None:
    """R40: gateway proposes only; a REAL saver proves replan-before-dispatch.

    First the gateway contract holds (accepted plan, zero dispatches, no
    checkpoint/execute surface). Then the subgraph runs under a real
    ``InMemorySaver`` and the audit asserts the checkpoint that first
    carries the 3-task replan strictly precedes the checkpoint that first
    carries the new T3 result — the Task-3 reviewer saver-audit method.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
        complex_evaluate_node,
        complex_execute_node,
        validate_checkpoint_node,
    )

    capabilities, leases, context = _harness(
        run_id="run-gw-audit-1", truncate_first_for=frozenset({"t2"})
    )
    document_capability = capabilities[0]
    gateway = AgentToolGateway()
    current = _two_target_plan()
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="document.read",
            objective="Re-read t2",
            input=DocumentReadInput(kind="document.read", target_ids=("t2",)),
            depends_on=("T2",),
        ),
        current,
        context,
    )
    assert outcome.accepted is True
    assert all(capability.calls == [] for capability in capabilities)
    assert not hasattr(gateway, "checkpoint")
    assert not hasattr(gateway, "execute")
    assert require_planned_dispatch(outcome.plan, "T3", context).task_id == "T3"

    # Drive the first pass through the real nodes, then resume the loop
    # under a real saver from that checkpointed first pass.
    child = _child_input(replans_remaining=2)
    child = {**child, **await validate_checkpoint_node(child, context)}
    child = {**child, **await complex_execute_node(child, context)}
    child = {**child, **await complex_evaluate_node(child, context)}
    assert child["evaluation"].status == "insufficient"

    saver = InMemorySaver()
    compiled = _build_complex_research_graph().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "thread-gw-audit-1"}}
    terminal = await compiled.ainvoke(child, config=config, context=context)
    assert terminal["evaluation"].status == "sufficient"
    assert [task.task_id for task in terminal["plan"].tasks] == ["T1", "T2", "T3"]

    history: list[dict] = []
    async for tup in saver.alist(config):
        history.append(dict(tup.checkpoint["channel_values"]))
    history.reverse()  # oldest first
    assert history, "the subgraph wrote checkpoints under the saver"

    def plan_task_count(values: dict) -> int:
        plan = values.get("plan")
        if plan is None:
            return 0
        tasks = plan.get("tasks") if isinstance(plan, dict) else plan.tasks
        return len(tasks or ())

    def result_ids(values: dict) -> set[str]:
        results = values.get("task_results") or ()
        ids = set()
        for result in results:
            task_id = result.get("task_id") if isinstance(result, dict) else result.task_id
            ids.add(task_id)
        return ids

    first_replan_at = next(
        (index for index, values in enumerate(history) if plan_task_count(values) == 3),
        None,
    )
    first_t3_result_at = next(
        (index for index, values in enumerate(history) if "T3" in result_ids(values)),
        None,
    )
    assert first_replan_at is not None, "a checkpoint carries the replan"
    assert first_t3_result_at is not None, "a checkpoint carries the T3 result"
    assert first_replan_at < first_t3_result_at, (
        "the replan-bearing checkpoint must precede the new-task dispatch"
    )
    assert leases.session.commits >= 1
    served = [call[0].task_id for call in document_capability.calls]
    assert served.count("T1") == 1 and served.count("T3") == 1


@pytest.mark.asyncio
async def test_gateway_rejects_when_planned_task_leaves_current_catalog() -> None:
    capabilities = (FakeReadCapability("document.read"),)
    full_runtime = CapabilityRuntimeContext(
        request_id="req-gw-2",
        run_id="run-gw-2",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.read", "section.read"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    full_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities]
        + [
            CapabilityRegistration(
                capability=FakeReadCapability("section.read"),
            )
        ],
        full_runtime,
    )
    gateway = AgentToolGateway()
    current = _two_target_plan()
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="section.read",
            objective="Re-read t1 by section",
            input=SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        current,
        GraphRuntimeContext(
            capability_runtime=full_runtime,
            services=RuntimeServices(capability_registry=full_registry),
        ),
    )
    assert outcome.accepted is True

    narrowed_runtime = CapabilityRuntimeContext(
        request_id="req-gw-2",
        run_id="run-gw-2",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.read"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    narrowed_registry = build_capability_registry(
        [CapabilityRegistration(capability=capabilities[0])],
        narrowed_runtime,
    )
    narrowed = GraphRuntimeContext(
        capability_runtime=narrowed_runtime,
        services=RuntimeServices(capability_registry=narrowed_registry),
    )
    # The previously accepted section task left the current catalog: a new
    # proposal against the narrowed runtime cannot widen it back.
    retry = await gateway.propose(
        CapabilityInvocationProposal(
            capability="section.read",
            objective="Re-read t1 by section",
            input=SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        current,
        narrowed,
    )
    assert retry.accepted is False
    assert retry.plan == current


# ---------------------------------------------------------------------------
# Discovery semantics: UUID identity, pin/promotion, no autonomous targets (R33)
# ---------------------------------------------------------------------------


def test_discovery_candidate_uuid_uniqueness() -> None:
    from app.services.agents.v2.discovery import mint_candidate

    first = mint_candidate(DOC_A, str(REV_A))
    second = mint_candidate(DOC_A, str(REV_A))
    assert isinstance(first.candidate_id, UUID)
    assert first.candidate_id != second.candidate_id
    assert isinstance(first, DocumentDiscoveryCandidate)


def test_parallel_discovery_tasks_cannot_collide() -> None:
    from app.services.agents.v2.discovery import mint_candidate
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    candidates = tuple(mint_candidate(DOC_A, str(REV_A)) for _ in range(8))
    ids = [candidate.candidate_id for candidate in candidates]
    assert len(set(ids)) == len(ids)
    results = tuple(
        AgentResult(
            contract_version="2.0",
            task_id=f"T{index}",
            status="success",
            data=DocumentSearchOutput(
                kind="document.search", candidates=(candidate,)
            ),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )
        for index, candidate in enumerate(candidates)
    )
    registry = DiscoveryCandidateRegistry.from_results(results)
    assert len(registry) == len(candidates)


def test_no_autonomous_target_creation() -> None:
    from app.services.agents.v2.discovery import (
        DiscoveryDisabled,
        request_addition,
    )
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=DocumentSearchOutput(
            kind="document.search",
            candidates=(
                DocumentDiscoveryCandidate(
                    candidate_id=uuid4(),
                    document_id=DOC_A,
                    document_revision=str(REV_A),
                ),
            ),
        ),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    registry = DiscoveryCandidateRegistry.from_results([result])
    candidate_id = registry.candidates()[0].candidate_id
    # The planner cannot create a new user target from discovery.
    with pytest.raises((DiscoveryDisabled, ValueError, ContractValidationError)):
        request_addition(
            registry, candidate_id, "target", policy=_policy()  # type: ignore[arg-type]
        )
    # Additions create supporting/discovered bindings only.
    addition = request_addition(
        registry,
        candidate_id,
        "discovered",
        policy=_policy(allow_reference_discovery=True, max_discovered_documents=1),
    )
    assert addition.requested_role == "discovered"
    supporting = request_addition(
        registry,
        candidate_id,
        "supporting",
        policy=_policy(allow_supporting_discovery=True, max_discovered_documents=1),
    )
    assert supporting.requested_role == "supporting"
    # The frozen replan validator rejects autonomous target-unit additions.
    current = _two_target_plan()
    proposed = current.model_copy(
        update={
            "target_units": current.target_units
            + (
                TargetUnit(
                    target_id="t3",
                    binding_id="b1",
                    requested_locator=DocumentLocator(kind="document"),
                    completion_criteria=(CoverageCriterion(kind="coverage"),),
                ),
            ),
            "tasks": current.tasks + (_appended(current).tasks[2],),
        }
    )
    with pytest.raises(ContractValidationError):
        validate_replan(current, proposed, (), _policy(), _budget())


def test_policy_disabled_discovery() -> None:
    from app.services.agents.v2.discovery import DiscoveryDisabled, request_addition
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    candidate_id = uuid4()
    registry = DiscoveryCandidateRegistry.from_results(
        [
            AgentResult(
                contract_version="2.0",
                task_id="T1",
                status="success",
                data=DocumentSearchOutput(
                    kind="document.search",
                    candidates=(
                        DocumentDiscoveryCandidate(
                            candidate_id=candidate_id,
                            document_id=DOC_A,
                            document_revision=str(REV_A),
                        ),
                    ),
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        ]
    )
    with pytest.raises(DiscoveryDisabled):
        request_addition(registry, candidate_id, "discovered", policy=_policy())
    # A document.search replan is rejected while discovery is disabled.
    current = _two_target_plan()
    search_append = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T3",
                    capability="document.search",
                    task_objective="discover more",
                    input=DocumentSearchInput(
                        kind="document.search", query="Dieu 5"
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason="discover",
                        task_ids=(),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_replan(current, search_append, (), _policy(), _budget())


def test_discovery_acl_denial() -> None:
    from app.services.agents.v2.discovery import (
        DiscoveryDenied,
        require_candidate_visible,
    )

    candidate = DocumentDiscoveryCandidate(
        candidate_id=uuid4(),
        document_id=DOC_A,
        document_revision=str(REV_A),
    )
    assert require_candidate_visible(candidate, frozenset({DOC_A})) is candidate
    with pytest.raises(DiscoveryDenied):
        require_candidate_visible(candidate, frozenset({DOC_B}))


def test_exact_revision_pin_and_promotion() -> None:
    from app.services.agents.v2.discovery import resolve_pin, validate_promotion
    from app.services.agents.v2.discovery import PromotionRequiresApproval
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    candidate_id = uuid4()
    registry = DiscoveryCandidateRegistry.from_results(
        [
            AgentResult(
                contract_version="2.0",
                task_id="T1",
                status="success",
                data=DocumentSearchOutput(
                    kind="document.search",
                    candidates=(
                        DocumentDiscoveryCandidate(
                            candidate_id=candidate_id,
                            document_id=DOC_A,
                            document_revision=str(REV_A),
                        ),
                    ),
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        ]
    )
    document_id, revision = resolve_pin(registry, candidate_id)
    assert document_id == DOC_A
    assert revision == str(REV_A)

    request = BindingPromotionRequest(
        source_binding_id="b1", requested_role="reference"
    )
    # Promotion requires an explicit validated policy/user action.
    with pytest.raises(PromotionRequiresApproval):
        validate_promotion(_bindings(), request, approved=False, reason="compare needs it")
    with pytest.raises(PromotionRequiresApproval):
        validate_promotion(_bindings(), request, approved=True, reason="  ")
    approved = validate_promotion(
        _bindings(), request, approved=True, reason="user confirmed reference"
    )
    assert approved == request
    with pytest.raises((PromotionRequiresApproval, ContractValidationError)):
        validate_promotion(
            _bindings(),
            BindingPromotionRequest(
                source_binding_id="missing", requested_role="reference"
            ),
            approved=True,
            reason="user confirmed",
        )


def test_current_latest_rebinding() -> None:
    from app.services.agents.v2.discovery import pin_for_rebinding

    # An ordinary pin stays pinned when a newer revision appears.
    assert (
        pin_for_rebinding(
            pinned_revision=str(REV_A),
            latest_revision="rev-new",
            is_current_required=False,
            user_authorized_refresh=True,
        )
        == str(REV_A)
    )
    # Explicit current/latest semantics rebind only with user authorization.
    assert (
        pin_for_rebinding(
            pinned_revision=str(REV_A),
            latest_revision="rev-new",
            is_current_required=True,
            user_authorized_refresh=False,
        )
        == str(REV_A)
    )
    assert (
        pin_for_rebinding(
            pinned_revision=str(REV_A),
            latest_revision="rev-new",
            is_current_required=True,
            user_authorized_refresh=True,
        )
        == "rev-new"
    )


@pytest.mark.asyncio
async def test_discovery_end_to_end_through_binding_resolver(
    discovery_settings: None,
) -> None:
    """R39: policy-enabled search → UUID candidate → resolver pins exact revision.

    The recovery fallback search runs on the real graph, its checkpointed
    candidates settle through the injected Binding Resolver (the tools layer
    creates nothing), and each new binding pins the candidate's exact
    discovered revision.
    """
    from pathlib import Path as _Path

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capabilities, leases, resolver, context = _discovery_harness(
        run_id="run-disc-e2e-1"
    )
    search_capability = capabilities[1]
    child = _child_input(
        plan=_people_plan(),
        task_results=(_not_found_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert [call[0].task_id for call in search_capability.calls] == ["T2"]

    # The resolver — not the tools layer — performed creation and received
    # the server-side candidates (UUID identity, exact revision).
    assert len(resolver.created) == 2
    for request, candidate, binding in resolver.created:
        assert candidate.candidate_id in (CANDIDATE_C, CANDIDATE_D)
        assert binding.document_id == candidate.document_id
        assert binding.document_revision == candidate.document_revision
        assert binding.role == request.requested_role == "discovered"
    pinned = {
        (binding.document_id, binding.document_revision)
        for binding in output["bindings"].bindings
    }
    assert (DOC_C, str(REV_C)) in pinned
    assert (DOC_D, str(REV_D)) in pinned
    assert leases.session.commits >= 1

    # The tools layer creates/pins no binding: no binding contracts there.
    tools_dir = (
        _Path(__file__).resolve().parents[5]
        / "backend/app/services/agents/v2/tools"
    )
    for module_path in sorted(tools_dir.glob("*.py")):
        source = module_path.read_text()
        assert "ScopedDocument" not in source, module_path.name
        assert "DocumentBindingSet" not in source, module_path.name


@pytest.mark.asyncio
async def test_discovery_policy_disabled_settles_nothing(replan_settings: None) -> None:
    """R39: with discovery disabled, no search is proposed and nothing settles."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capabilities, _, resolver, context = _discovery_harness(
        run_id="run-disc-off-1"
    )
    search_capability = capabilities[1]
    child = _child_input(
        plan=_people_plan(),
        task_results=(_not_found_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert [task.task_id for task in output["plan"].tasks] == ["T1"]
    assert search_capability.calls == []
    assert resolver.created == []
    assert output["evaluation"].status == "insufficient"


@pytest.mark.asyncio
async def test_discovery_acl_denied_candidate_settles_nothing(
    discovery_settings: None,
) -> None:
    """R39: an ACL-denied candidate is skipped without metadata leakage."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capabilities, _, resolver, context = _discovery_harness(
        run_id="run-disc-deny-1", visible=frozenset({DOC_C})
    )
    child = _child_input(
        plan=_people_plan(),
        task_results=(_not_found_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert len(resolver.created) == 1
    _, denied_candidate, _ = resolver.created[0]
    assert denied_candidate.document_id == DOC_C
    pinned = {
        (binding.document_id, binding.document_revision)
        for binding in output["bindings"].bindings
    }
    assert (DOC_C, str(REV_C)) in pinned
    assert (DOC_D, str(REV_D)) not in pinned
    # The run terminates (finalize) rather than looping on the denial.
    assert output["evaluation"].status == "insufficient"


@pytest.mark.asyncio
async def test_discovery_cap_counts_existing_plus_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R44: max_discovered_documents=1 with 2 candidates settles exactly one."""
    from types import SimpleNamespace

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8,
            V2_MAX_PARALLEL_BRANCHES=2,
            V2_MAX_REPLANS=2,
            V2_ALLOW_REFERENCE_DISCOVERY=True,
            V2_ALLOW_SUPPORTING_DISCOVERY=True,
            V2_MAX_DISCOVERED_DOCUMENTS=1,
        ),
    )
    capabilities, _, resolver, context = _discovery_harness(
        run_id="run-disc-cap-1"
    )
    child = _child_input(
        plan=_people_plan(),
        task_results=(_not_found_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    output = await build_complex_research_subgraph().ainvoke(
        child, context=context
    )
    assert len(resolver.created) == 1
    _, settled_candidate, _ = resolver.created[0]
    assert settled_candidate.candidate_id == CANDIDATE_C
    # The extra candidate is recorded, never added.
    assert output["discovery_deferred"] == (str(CANDIDATE_D),)
    pinned = {
        (binding.document_id, binding.document_revision)
        for binding in output["bindings"].bindings
    }
    assert (DOC_C, str(REV_C)) in pinned
    assert (DOC_D, str(REV_D)) not in pinned


@pytest.mark.asyncio
async def test_discovery_cap_honours_previously_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R44: one pre-existing discovered binding consumes a max=1 budget."""
    from types import SimpleNamespace

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8,
            V2_MAX_PARALLEL_BRANCHES=2,
            V2_MAX_REPLANS=2,
            V2_ALLOW_REFERENCE_DISCOVERY=True,
            V2_ALLOW_SUPPORTING_DISCOVERY=True,
            V2_MAX_DISCOVERED_DOCUMENTS=1,
        ),
    )
    capabilities, _, resolver, context = _discovery_harness(
        run_id="run-disc-cap-2"
    )
    seeded = DocumentBindingSet(
        bindings=_bindings().bindings
        + (
            ScopedDocument(
                binding_id="d_prior",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="discovered",
            ),
        ),
        revision_requirement_refs=(),
    )
    child = _child_input(
        plan=_people_plan(),
        bindings=seeded,
        task_results=(_not_found_people_result(),),
        evaluation=_insufficient_evaluation(),
        replans_remaining=2,
    )
    output = await build_complex_research_subgraph().ainvoke(
        child, context=context
    )
    assert resolver.created == []
    assert set(output["discovery_deferred"]) == {
        str(CANDIDATE_C),
        str(CANDIDATE_D),
    }


# ---------------------------------------------------------------------------
# Model-facing projections carry no People scalar (R34)
# ---------------------------------------------------------------------------


def test_replan_input_carries_no_people_scalar() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_model_observations,
        build_model_replan_input,
    )
    from app.services.agents.v2.dependencies.people_document import (
        append_materialized_dependent,
    )
    from app.services.agents.v2.replanning import append_replan_tasks

    base = TaskPlan(
        contract_version="2.0",
        plan_id="pp",
        goal="nghi dinh nao neu CCCD cua A",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="T1",
                capability="people.lookup",
                task_objective="find A",
                input=PeopleLookupInput(kind="people.lookup", query="A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    use_id = uuid4()
    # R52: append_materialized_dependent returns the concrete T2 PROPOSAL;
    # the governed append (append_replan_tasks) builds AND validates the
    # authoritative plan.
    dependent = append_materialized_dependent(
        current=base,
        outcome=PeopleDocumentMaterialization(
            kind="materialized",
            scalar="012345678901",
            input=DocumentSearchInput(
                kind="document.search",
                query="nghi dinh",
                person_identifier="012345678901",
            ),
            error_code=None,
            reason="governed scalar materialized server-side",
            evidence_use_ids=(use_id,),
        ),
        query="nghi dinh",
        next_task_id="T2",
    )
    _, _, _, context = _discovery_harness(run_id="run-redact-1")
    materialized = append_replan_tasks(
        base,
        (dependent,),
        (TaskExecutionSummary(task_id="T1", status="success"),),
        DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=True,
            max_discovered_documents=1,
        ),
        ResearchBudgetView(
            max_tasks_remaining=8,
            max_replans_remaining=1,
            max_parallel_branches=2,
        ),
        context,
    )
    assert materialized.tasks[1].input.person_identifier == "012345678901"  # type: ignore[union-attr]

    child = _child_input(
        plan=materialized,
        task_results=(),
        evaluation=None,
        replans_remaining=1,
        people_scalar_available={"T1": True},
    )
    model_input = build_model_replan_input(child, context)
    payload = model_input.model_dump_json()
    assert "012345678901" not in payload
    assert model_input.current_plan is not None
    for task in model_input.current_plan.tasks:
        assert getattr(task.input, "person_identifier", None) is None

    people_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(),
        error=None,
    )
    observations = build_model_observations(
        _child_input(
            plan=materialized,
            task_results=(people_result,),
            people_scalar_available={"T1": True},
        )
    )
    assert len(observations) == 1
    assert observations[0].projection.dependency_scalar_available is True  # type: ignore[union-attr]
    assert "012345678901" not in observations[0].model_dump_json()


# ---------------------------------------------------------------------------
# Summarize is a skill, not an agent route (R35/R38)
# ---------------------------------------------------------------------------


def _single_target_bindings() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def _section_semantic() -> SemanticContext:
    from app.services.agents.v2.contracts.semantic import SectionReference

    return SemanticContext(
        contextualized_query="Tom tat Chuong II va Chuong III",
        normalized_query="tom tat chuong ii va chuong iii",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(
            SectionReference(
                ref_id="r1", label="Chuong II", structure_node_id="chap-II"
            ),
            SectionReference(
                ref_id="r2", label="Chuong III", structure_node_id="chap-III"
            ),
        ),
        blocking_ambiguities=(),
    )


@pytest.mark.asyncio
async def test_summary_is_skill_not_agent_route() -> None:
    """R38: the REAL selection path serves summarize; no summary agent."""
    from app.services.agents.v2.complex_research_graph import (
        validate_checkpoint_node,
    )
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    assert summarize_policy.supports_work_type("summarize") is True
    assert summarize_policy.supports_work_type("compare") is False
    assert summarize_policy.supports_work_type("evaluate") is False

    _, leases, context = _harness(run_id="run-summarize-1")
    child = _child_input(
        query_analysis=_analysis("summarize"), bindings=_single_target_bindings()
    )
    # The real initial-validation path selects the summarize skill by work
    # type — the test never calls the policy directly.
    decided = await validate_checkpoint_node(child, context)
    plan = decided["plan"]
    assert plan.plan_id.startswith("summarize-")
    assert len(plan.target_units) == 1
    assert len(plan.tasks) == 1
    assert plan.tasks[0].capability == "document.read"
    assert isinstance(plan.tasks[0].origin, InitialTaskOrigin)
    assert leases.session.commits >= 1

    v2_root = (
        Path(__file__).resolve().parents[5]
        / "backend/app/services/agents/v2"
    )
    assert v2_root.is_dir()
    assert list(v2_root.glob("*summary_agent*")) == []
    import app.services.agents.v2.complex_research_graph as subgraph_module

    source = inspect.getsource(subgraph_module)
    assert "SummaryAgent" not in source
    assert "summary_agent" not in source
    import app.services.agents.v2.skills.summarize.policy as policy_module

    policy_source = inspect.getsource(policy_module)
    assert "capability.execute(" not in policy_source
    assert "TaskScheduler" not in policy_source
    assert "FinalResponse" not in policy_source
    assert "EvidenceUse" not in policy_source


def test_summarize_workflow_descriptor_is_explicit() -> None:
    """R43(a)(b): the policy returns ordered MAP tasks plus an explicit REDUCE."""
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.summarize import policy as summarize_policy
    from app.services.agents.v2.skills.summarize.policy import (
        ReduceSpec,
        SummarizeWorkflow,
    )

    _, _, context = _harness(run_id="run-summarize-wf-1")
    child = _child_input(
        query_analysis=_analysis("summarize"),
        bindings=_single_target_bindings(),
        semantic=_section_semantic(),
    )
    first = summarize_policy.build_summarize_workflow(
        build_planning_input(child, context)
    )
    second = summarize_policy.build_summarize_workflow(
        build_planning_input(child, context)
    )
    assert isinstance(first, SummarizeWorkflow)
    assert isinstance(first.reduce, ReduceSpec)
    assert first.reduce.mode == "extractive"
    # Deterministic order: repeated proposals agree exactly.
    assert first == second
    assert first.reduce.map_task_ids == ("T1", "T2")
    assert [task.task_id for task in first.plan.tasks] == ["T1", "T2"]
    assert [unit.target_id for unit in first.plan.target_units] == ["s1", "s2"]
    # The descriptor lives outside the frozen contracts.
    assert SummarizeWorkflow.__module__.startswith(
        "app.services.agents.v2.skills.summarize"
    )


@pytest.mark.asyncio
async def test_large_summarize_map_reduces_over_sections() -> None:
    """R38+R43(c): map covers, then the framework reduce stage executes."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
        validate_checkpoint_node,
    )
    from app.services.agents.v2.contracts.locators import SectionLocator
    from app.services.agents.v2.contracts.validation import validate_answer_draft

    locator_for = {
        "s1": SectionLocator(kind="section", structure_node_id="chap-II"),
        "s2": SectionLocator(kind="section", structure_node_id="chap-III"),
    }
    capabilities, _, context = _harness(
        run_id="run-summarize-map-1", locator_for=locator_for
    )
    section_capability = capabilities[1]
    channel = FakeDraftChannel()
    context.services.answer_draft_channel = channel
    child = _child_input(
        query_analysis=_analysis("summarize"),
        bindings=_single_target_bindings(),
        semantic=_section_semantic(),
    )
    decided = await validate_checkpoint_node(child, context)
    plan = decided["plan"]
    assert plan.plan_id.startswith("summarize-")
    assert [unit.target_id for unit in plan.target_units] == ["s1", "s2"]
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]
    assert {task.capability for task in plan.tasks} == {"section.read"}
    assert decided["reduce_spec"].map_task_ids == ("T1", "T2")

    compiled = _build_complex_research_graph().compile(
        checkpointer=InMemorySaver()
    )
    output = await compiled.ainvoke(
        child,
        config={"configurable": {"thread_id": "thread-summarize-map-1"}},
        context=context,
    )
    assert output["evaluation"].status == "sufficient"
    assert len(section_capability.calls) == 2

    # The REDUCE stage executed through the framework handoff: one ordered
    # extractive draft (s1 before s2), validated against admitted uses.
    assert len(channel.stored) == 1
    run_id, draft, evidence = channel.stored[0]
    assert run_id == "run-summarize-map-1"
    assert [claim.text for claim in draft.claims] == [
        "content of s1",
        "content of s2",
    ]
    admitted_ids = frozenset(item.use_id for item in evidence)
    assert len(admitted_ids) == 2
    validate_answer_draft(draft, admitted_ids)


def test_reduce_uses_framework_boundaries_not_agent_or_capability() -> None:
    """R43(c)(d)/R47: reduce runs the synthesis boundary; no new capability."""
    import app.services.agents.v2.complex_research_graph as graph_module
    import app.services.agents.v2.skills.summarize.policy as policy_module
    from app.services.agents.v2.contracts.capability import CapabilityInput

    reduce_source = inspect.getsource(graph_module.summarize_reduce_node)
    assert "synthesize_answer" in reduce_source
    assert "store_draft" in reduce_source
    assert "hydrate_for_evaluation" not in reduce_source
    assert "build_extractive_draft" not in reduce_source
    assert "capability.execute(" not in reduce_source
    assert "TaskScheduler" not in reduce_source
    policy_source = inspect.getsource(policy_module)
    assert "capability.execute(" not in policy_source

    # No new frozen capability beyond the P0 additive `document.retrieve`:
    # the input union is exactly the frozen set plus the authorized addition.
    from typing import get_args

    members = get_args(get_args(CapabilityInput)[0])
    kinds: set[str] = set()
    for member in members:
        kinds.update(get_args(member.model_fields["kind"].annotation))
    assert kinds == {
        "people.lookup",
        "document.search",
        "document.retrieve",
        "document.read",
        "section.read",
        "write",
        "knowledge_graph.query",
        "memory.lookup",
        "abbreviation.resolve",
    }
    # The workflow descriptor is not a frozen contract.
    import app.services.agents.v2.contracts as contracts_package

    assert not hasattr(contracts_package, "SummarizeWorkflow")
    assert not hasattr(contracts_package, "ReduceSpec")


class BudgetSpyHydrator(FakeHydrator):
    """Synthesis-entry spy: records the budget and enforces its item head."""

    def __init__(self, capabilities: tuple[FakeReadCapability, ...]) -> None:
        super().__init__(capabilities)
        self.synthesis_calls: list[dict] = []

    async def hydrate_for_synthesis(  # type: ignore[no-untyped-def]
        self, use_refs, *, runtime, plan, bindings, budget
    ):
        self.synthesis_calls.append(
            {"budget": budget, "ref_ids": [ref.use_id for ref in use_refs]}
        )
        admitted = await super().hydrate_for_synthesis(
            use_refs, runtime=runtime, plan=plan, bindings=bindings, budget=budget
        )
        return tuple(admitted)[: budget.max_evidence_items]


@pytest.mark.asyncio
async def test_reduce_applies_synthesis_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R47: the reduce exercises the synthesis entry and honors its budget."""
    from langgraph.checkpoint.memory import InMemorySaver

    import app.services.agents.v2.complex_research_graph as graph_module
    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
    )
    from app.services.agents.v2.contracts.locators import SectionLocator
    from app.services.agents.v2.contracts.synthesis import SynthesisRuntimeContext

    tiny_budget = SynthesisRuntimeContext(
        max_evidence_items=1, max_total_chars=100000, max_total_tokens=10000
    )
    monkeypatch.setattr(
        graph_module, "DEFAULT_SYNTHESIS_BUDGET", tiny_budget
    )
    locator_for = {
        "s1": SectionLocator(kind="section", structure_node_id="chap-II"),
        "s2": SectionLocator(kind="section", structure_node_id="chap-III"),
    }
    (document_capability, section_capability), leases, context = _harness(
        run_id="run-reduce-budget-1", locator_for=locator_for
    )
    spy = BudgetSpyHydrator((document_capability, section_capability))
    context.services.evidence_hydrator = spy
    channel = FakeDraftChannel()
    context.services.answer_draft_channel = channel
    child = _child_input(
        query_analysis=_analysis("summarize"),
        bindings=_single_target_bindings(),
        semantic=_section_semantic(),
    )
    compiled = _build_complex_research_graph().compile(
        checkpointer=InMemorySaver()
    )
    output = await compiled.ainvoke(
        child,
        config={"configurable": {"thread_id": "thread-reduce-budget-1"}},
        context=context,
    )
    assert output["evaluation"].status == "sufficient"
    # The synthesis entry ran exactly once (the reduce), with the budget,
    # and the map uses arrived in spec order.
    assert len(spy.synthesis_calls) == 1
    call = spy.synthesis_calls[0]
    assert call["budget"] is tiny_budget
    s1_uses = {
        use_id
        for use_id, (_, target_id) in section_capability.uses.items()
        if target_id == "s1"
    }
    s2_uses = {
        use_id
        for use_id, (_, target_id) in section_capability.uses.items()
        if target_id == "s2"
    }
    assert len(s1_uses) == 1 and len(s2_uses) == 1
    assert call["ref_ids"] == [next(iter(s1_uses)), next(iter(s2_uses))]
    # The passed budget bound the draft: one head claim, not two.
    assert len(channel.stored) == 1
    _, draft, _ = channel.stored[0]
    assert [claim.text for claim in draft.claims] == ["content of s1"]


@pytest.mark.asyncio
async def test_reduce_missing_seam_fails_closed() -> None:
    """R47: a missing channel or hydrator fails the mandatory reduce closed."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.services.agents.v2.complex_research_graph import (
        ComplexResearchError,
        _build_complex_research_graph,
        summarize_reduce_node,
    )
    from app.services.agents.v2.contracts.locators import SectionLocator
    from app.services.agents.v2.nodes.synthesize import SynthesisError

    locator_for = {
        "s1": SectionLocator(kind="section", structure_node_id="chap-II"),
        "s2": SectionLocator(kind="section", structure_node_id="chap-III"),
    }
    _, _, context = _harness(
        run_id="run-reduce-closed-1", locator_for=locator_for
    )
    channel = FakeDraftChannel()
    context.services.answer_draft_channel = channel
    child = _child_input(
        query_analysis=_analysis("summarize"),
        bindings=_single_target_bindings(),
        semantic=_section_semantic(),
    )
    compiled = _build_complex_research_graph().compile(
        checkpointer=InMemorySaver()
    )
    sufficient = await compiled.ainvoke(
        child,
        config={"configurable": {"thread_id": "thread-reduce-closed-1"}},
        context=context,
    )
    assert sufficient["evaluation"].status == "sufficient"
    assert len(channel.stored) == 1

    context.services.answer_draft_channel = None
    with pytest.raises(ComplexResearchError):
        await summarize_reduce_node(sufficient, context)
    context.services.answer_draft_channel = channel
    context.services.evidence_hydrator = None
    with pytest.raises(SynthesisError):
        await summarize_reduce_node(sufficient, context)


def test_bounded_summarize_stays_fast() -> None:
    """R38: a single bound document summarizes on the fast path, not complex."""
    from app.services.agents.v2.contracts.semantic import DocumentReference
    from app.services.agents.v2.nodes.routing import decide_route

    semantic = SemanticContext(
        contextualized_query="Tom tat A",
        normalized_query="tom tat a",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="tai lieu A",
                normalized_reference="tai lieu a",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOC_A,
                candidate_document_ids=(),
            ),
        ),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    bindings = DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    decision = decide_route(
        _analysis("summarize"),
        semantic,
        bindings,
        allowed_capabilities=frozenset({"document.read"}),
    )
    assert decision.route == "fast_domain"


# ---------------------------------------------------------------------------
# Subagents are advisory only: no plan/task/execution authority (§9)
# ---------------------------------------------------------------------------


# R41 uses option (b): no advisory/subagent module exists (the amendment
# authorizes advisory subagents but requires none), so the guards pin the
# actual invariants structurally — a new module would add surface the
# amendment does not require. Advisory input can only be the typed
# observation/projection payloads, whose schemas are pinned below, and no
# advisory output can become an authoritative task except through the
# validated gateway/replan path.


def _scan_append_ownership(
    root: Path,
) -> tuple[
    set[tuple[str, str]],
    set[tuple[str, str]],
    set[tuple[str, str]],
    set[tuple[str, str]],
    set[tuple[str, str]],
]:
    """Scan a v2 tree for append-ownership invariants (R52).

    Returns ``(constructors, append_sites, in_place_sites,
    append_replan_tasks_callers, plan_returns)`` where each element is the set
    of ``(module_rel, enclosing_function)`` locations. ``append_sites`` is the
    single governed surface: EVERY append expression -- ``model_copy(update={
    "tasks": ...})``, ``list(<*.tasks>)`` / ``tuple(<*.tasks>)``,
    ``<tasks-expr>.append(...)``, ``<tasks> + ...``, and in-place ``.tasks``
    assignment -- is collected and its ENCLOSING FUNCTION resolved. Callers
    assert that set is exactly ``append_replan_tasks``.
    """
    import ast

    constructors: set[tuple[str, str]] = set()
    append_sites: set[tuple[str, str]] = set()
    in_place_sites: set[tuple[str, str]] = set()
    append_replan_tasks_callers: set[tuple[str, str]] = set()
    plan_returns: set[tuple[str, str]] = set()

    for module_path in sorted(root.rglob("*.py")):
        if "__pycache__" in module_path.parts:
            continue
        rel = str(module_path.relative_to(root))
        tree = ast.parse(module_path.read_text())

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []
                self.tainted: list[set[str]] = []

            def _func(self) -> str:
                return self.scope[-1] if self.scope else "<module>"

            def _tainted(self) -> set[str]:
                return self.tainted[-1] if self.tainted else set()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.scope.append(node.name)
                self.tainted.append(set())
                self.generic_visit(node)
                self.tainted.pop()
                self.scope.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def reads_tasks(self, node: ast.AST) -> bool:
                if isinstance(node, ast.Attribute) and node.attr == "tasks":
                    return True
                if isinstance(node, ast.Name) and node.id in self._tainted():
                    return True
                if isinstance(node, (ast.Tuple, ast.List)):
                    return any(self.reads_tasks(element) for element in node.elts)
                if isinstance(node, ast.Starred):
                    return self.reads_tasks(node.value)
                if isinstance(node, ast.BinOp):
                    return self.reads_tasks(node.left) or self.reads_tasks(node.right)
                return False

            @staticmethod
            def _call_name(func_ref: ast.expr) -> str:
                if isinstance(func_ref, ast.Name):
                    return func_ref.id
                if isinstance(func_ref, ast.Attribute):
                    return func_ref.attr
                return ""

            @staticmethod
            def _is_model_copy_with_tasks(node: ast.Call) -> bool:
                if Visitor._call_name(node.func) != "model_copy":
                    return False
                for keyword in node.keywords:
                    if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
                        return any(
                            isinstance(key, ast.Constant) and key.value == "tasks"
                            for key in keyword.value.keys
                        )
                return False

            def _tasks_wrap(self, node: ast.Call) -> bool:
                """``list(<tasks>)`` / ``tuple(<tasks>)`` call-wrapped forms."""
                return (
                    self._call_name(node.func) in ("list", "tuple")
                    and bool(node.args)
                    and self.reads_tasks(node.args[0])
                )

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        value = node.value
                        if self.reads_tasks(value) or (
                            isinstance(value, ast.Call) and self._tasks_wrap(value)
                        ):
                            self._tainted().add(target.id)
                    if isinstance(target, ast.Attribute) and target.attr == "tasks":
                        in_place_sites.add((rel, self._func()))
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if (
                    isinstance(node.target, ast.Name)
                    and node.value is not None
                    and self.reads_tasks(node.value)
                ):
                    self._tainted().add(node.target.id)
                if isinstance(node.target, ast.Attribute) and node.target.attr == "tasks":
                    in_place_sites.add((rel, self._func()))
                self.generic_visit(node)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                if isinstance(node.target, ast.Attribute) and node.target.attr == "tasks":
                    in_place_sites.add((rel, self._func()))
                if isinstance(node.op, ast.Add) and self.reads_tasks(node.value):
                    append_sites.add((rel, self._func()))
                self.generic_visit(node)

            def visit_BinOp(self, node: ast.BinOp) -> None:
                if isinstance(node.op, ast.Add) and (
                    self.reads_tasks(node.left) or self.reads_tasks(node.right)
                ):
                    append_sites.add((rel, self._func()))
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                name = self._call_name(node.func)
                if name == "TaskPlan":
                    constructors.add((rel, self._func()))
                elif name == "append_replan_tasks":
                    append_replan_tasks_callers.add((rel, self._func()))
                elif name == "model_copy" and self._is_model_copy_with_tasks(node):
                    append_sites.add((rel, self._func()))
                elif self._tasks_wrap(node):
                    append_sites.add((rel, self._func()))
                elif (
                    name == "append"
                    and isinstance(node.func, ast.Attribute)
                    and self.reads_tasks(node.func.value)
                ):
                    append_sites.add((rel, self._func()))
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                if isinstance(node.value, ast.Dict) and any(
                    isinstance(key, ast.Constant) and key.value == "plan"
                    for key in node.value.keys
                ):
                    plan_returns.add((rel, self._func()))
                self.generic_visit(node)

        Visitor().visit(tree)

    return (
        constructors,
        append_sites,
        in_place_sites,
        append_replan_tasks_callers,
        plan_returns,
    )


def test_append_guard_flags_call_wrapped_defeat_form(tmp_path: Path) -> None:
    """R52: the guard FAILS the reviewer's call-wrapped defeat form.

    The defeat snippet wraps ``list(plan.tasks)`` + ``.append`` +
    ``model_copy(update={"tasks": tuple(...)})`` inside an otherwise ordinary
    function. The SAME scanner used for the real tree must flag that function
    as an append site -- proving the detector descends through the
    call-wrapped forms rather than only immediate ``.tasks + ...``. A second,
    minimal snippet (only the ``list`` + ``.append`` forms, no ``model_copy``)
    proves those call-wrapped forms are detected independently of the
    ``model_copy`` token.
    """
    import textwrap

    defeat = tmp_path / "defeat_snippet.py"
    defeat.write_text(
        textwrap.dedent(
            """
            def redact_scalar_for_model(plan, new_task):
                copied = list(plan.tasks)
                copied.append(new_task)
                return plan.model_copy(update={"tasks": tuple(copied)})
            """
        )
    )
    _, append_sites, _, _, _ = _scan_append_ownership(tmp_path)
    assert ("defeat_snippet.py", "redact_scalar_for_model") in append_sites

    # The call-wrapped forms are detected even WITHOUT the model_copy token,
    # so the guard does not depend on that single marker.
    minimal = tmp_path / "list_append_snippet.py"
    minimal.write_text(
        textwrap.dedent(
            """
            def rogue_builder(plan, new_task):
                copied = list(plan.tasks)
                copied.append(new_task)
                return copied
            """
        )
    )
    _, append_sites, _, _, _ = _scan_append_ownership(tmp_path)
    assert ("list_append_snippet.py", "rogue_builder") in append_sites


def test_subagent_cannot_append_authoritative_tasks() -> None:
    """R52: every authoritative plan-append is single-sourced and governed.

    Structural (AST, enclosing-function granularity, with temporary-variable
    taint tracking -- not a file whitelist and not only ``.tasks +``):

    * ``TaskPlan(`` constructors exist only in the deterministic builders,
      the same-length model-facing redaction projection, and the governed
      initial planner (which builds fresh from validated model steps and
      still passes frozen ``validate_task_plan`` before lease/checkpoint).
    * EVERY append expression -- ``model_copy(update={"tasks": ...})``,
      ``list(<*.tasks>)`` / ``tuple(<*.tasks>)``, ``<tasks-expr>.append(...)``,
      ``<tasks> + ...``, and in-place ``.tasks`` assignment -- resolves to the
      ENCLOSING FUNCTION ``replanning.append_replan_tasks``.
    * no function mutates ``.tasks`` in place.
    * ``append_replan_tasks`` is called only from the governed owners:
      ``validate_checkpoint_node``, ``people_document_materialize_node`` and
      ``AgentToolGateway.propose`` (call-graph basis).
    * checkpointed-plan updates are returned only by the two owning nodes.

    A rogue append in any other function -- via temporary variable, unpacking,
    list construction, or a helper that returns an appended plan -- fails one
    of these assertions, so an ungoverned append cannot slip through.
    """
    v2_root = Path(__file__).resolve().parents[5] / "backend/app/services/agents/v2"
    assert v2_root.is_dir()

    (
        constructors,
        append_sites,
        in_place_sites,
        append_replan_tasks_callers,
        plan_returns,
    ) = _scan_append_ownership(v2_root)

    # Initial-plan constructors: deterministic builders + the same-length
    # redaction projection (which rebuilds the plan WITHOUT a tasks model_copy)
    # + the governed initial planner (fresh construction from model steps;
    # every TaskSpec input is server-built and the plan passes frozen
    # validation before lease/checkpoint — never an append). Task 12 moved
    # the people-first construction into the cross-domain skill:
    # ``_people_first_plan`` is now a thin delegate with no ``TaskPlan(``
    # constructor, so the scanner correctly drops it and the skill builder
    # takes its place — the single-construction-site property is preserved.
    assert constructors == {
        ("nodes/fast_plan.py", "build_fast_plan"),
        ("skills/compare/policy.py", "build_compare_plan"),
        ("skills/cross_domain/policy.py", "build_cross_domain_plan"),
        ("skills/evaluate/policy.py", "build_evaluate_plan"),
        ("skills/multi_goal/policy.py", "build_multi_goal_plan"),
        ("skills/retrieve/policy.py", "build_retrieve_plan"),
        ("skills/summarize/policy.py", "build_summarize_workflow"),
        ("dependencies/people_document.py", "redact_scalar_for_model"),
        ("planning/planner.py", "_build_plan"),
    }
    # The single authoritative append expression surface (R52): every append
    # expression's ENCLOSING FUNCTION is exactly append_replan_tasks.
    assert append_sites == {("replanning.py", "append_replan_tasks")}
    assert in_place_sites == set()
    # Call-graph basis: the single append helper is reached only from the
    # governed owners.
    assert append_replan_tasks_callers == {
        ("complex_research_graph.py", "validate_checkpoint_node"),
        ("complex_research_graph.py", "people_document_materialize_node"),
        ("tools/gateway.py", "propose"),
    }
    # Checkpointed-plan updates are returned only by the two owning nodes
    # (the parent->child mapping passes the already-checkpointed plan through).
    assert plan_returns == {
        ("complex_research_graph.py", "validate_checkpoint_node"),
        ("complex_research_graph.py", "people_document_materialize_node"),
    }

    # A direct internal execute_ready_tasks call is NOT an ingress: it takes an
    # explicit plan and is only reachable through the shared scheduler, which
    # the graph's execute node feeds exclusively the checkpointed plan. The
    # governed graph path is proven separately (see
    # test_unvalidated_append_never_becomes_checkpointed_plan).
    _, _, context = _harness(run_id="run-subagent-rogue-1")
    with pytest.raises(UnplannedCapabilityDispatch):
        require_planned_dispatch(_two_target_plan(), "T-subagent-1", context)


@pytest.mark.asyncio
async def test_unvalidated_append_never_becomes_checkpointed_plan(
    discovery_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R52: the governed graph never checkpoints/dispatches an unvalidated append.

    A plan whose append bypassed ``append_replan_tasks`` never becomes the
    checkpointed plan the graph dispatches. The enforcing chain is: every append
    goes through ``append_replan_tasks`` (which calls the frozen validator), the
    subgraph checkpoints only that validated plan, and ``execute`` runs only the
    checkpointed plan. A direct internal ``execute_ready_tasks`` call is NOT an
    ingress (it takes an explicit plan and is only reachable through the shared
    scheduler the execute node feeds exclusively the checkpointed plan), so it is
    not defended here.

    This test runs the REAL compiled graph through the recovery replan and
    proves (a) the only append dispatched is the validated recovery task, and
    (b) the reviewer's call-wrapped rogue append (catalog-valid, no fabricated
    scalar) is never the checkpointed plan and never dispatched.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    import app.services.agents.v2.complex_research_graph as graph_module
    import app.services.agents.v2.replanning as replanning_module
    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
        normalize_complex_state,
    )

    people, search, _, _, _, context = _people_harness(
        "run-rogue-plan-1", FakePeopleCapability(status="not_found")
    )
    child = _child_input(
        semantic=_person_semantic(),
        query_analysis=_cross_domain_analysis(),
        replans_remaining=2,
    )

    # Spy on the single authoritative append AND the frozen validator so the
    # behavioral proof asserts (a) the checkpointed plan is EXACTLY the append
    # result and (b) that append result was actually validated.
    real_append = graph_module.append_replan_tasks
    append_results: list[TaskPlan] = []

    def spying_append(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = real_append(*args, **kwargs)
        append_results.append(result)
        return result

    monkeypatch.setattr(graph_module, "append_replan_tasks", spying_append)

    real_validate = replanning_module.validate_replan
    validation_calls: list[tuple] = []

    def spying_validate(*args, **kwargs):  # type: ignore[no-untyped-def]
        validation_calls.append(args)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(replanning_module, "validate_replan", spying_validate)

    compiled = _build_complex_research_graph().compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "thread-rogue-plan-1"}}
    async for _chunk in compiled.astream(child, config=config, context=context):
        pass

    snapshot = await compiled.aget_state(config)
    terminal = normalize_complex_state(dict(snapshot.values))

    # The append went through the single governed function, the frozen validator
    # ran on it, and its result is exactly the checkpointed plan.
    assert len(append_results) == 1
    assert len(validation_calls) == 1
    assert terminal["plan"] == append_results[0]
    assert terminal["plan"] == validation_calls[0][1]
    assert [task.task_id for task in terminal["plan"].tasks] == ["T1", "T2"]
    assert [call[0].task_id for call in search.calls] == ["T2"]

    # The reviewer's defeat form: catalog-valid document.search, no fabricated
    # scalar, but bypassing append_replan_tasks. It must never be the
    # checkpointed plan nor any dispatched task.
    base = _people_plan()
    rogue_task = TaskSpec(
        task_id="T2",
        capability="document.search",
        task_objective="rogue append bypassing append_replan_tasks",
        input=DocumentSearchInput(
            kind="document.search", query="nghi dinh", person_identifier=None
        ),
        depends_on=(),
        origin=ReplanTaskOrigin(
            kind="replan",
            reason="bypassed-append-replan-tasks",
            task_ids=(),
            evidence_use_ids=(),
        ),
    )
    copied = list(base.tasks)
    copied.append(rogue_task)
    rogue_plan = base.model_copy(update={"tasks": tuple(copied)})

    assert terminal["plan"] != rogue_plan
    assert "bypassed-append-replan-tasks" not in terminal["plan"].model_dump_json()
    dispatched = search.calls[0][0]
    assert dispatched.task_id == "T2"
    assert dispatched.objective != rogue_task.task_objective
    # The governed append carries the recovery lineage the rogue lacks.
    recovery = terminal["plan"].tasks[1]
    assert recovery.depends_on == ("T1",)
    assert recovery.origin.task_ids == ("T1",)  # type: ignore[union-attr]
    assert "not_found" in recovery.origin.reason  # type: ignore[union-attr]


def test_catalog_valid_append_bypassing_governance_is_rejected() -> None:
    """R50: governance, not capability validity, rejects an ungoverned append.

    ``document.search`` IS in the current catalog, so capability validity
    alone would not reject it. But an append that fabricates a
    ``person_identifier`` without going through the governed
    People->Document materializer (the only governed place that may produce
    a materialized scalar) is rejected by the governed validation entry
    point with a frozen contract error -- not a capability mismatch.
    """
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, _, context = _discovery_harness(run_id="run-governance-1")
    current = _people_plan()
    rogue = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T2",
                    capability="document.search",
                    task_objective="rogue discovery search with a fabricated scalar",
                    input=DocumentSearchInput(
                        kind="document.search",
                        query="nghi dinh",
                        person_identifier="smuggled-scalar",
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason="smuggled",
                        task_ids=(),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current,
            rogue,
            (),
            _policy(allow_supporting_discovery=True, max_discovered_documents=1),
            _budget(),
            context,
        )


@pytest.mark.asyncio
async def test_subagent_cannot_receive_raw_people_record_or_runtime_secrets() -> None:
    """R51/R53: KNOWN scalar + KNOWN runtime secret never reach model projections.

    Both sentinels are REAL (non-vacuous): the People scalar is present in
    the authoritative checkpointed plan (T2 ``person_identifier``) and the
    runtime secret is present in the request-scoped service seam
    (``RuntimeServices.authorization``). Both must be ABSENT from every
    advisory/subagent-facing projection (model replan input, observations,
    tool adapter inputs) AND from the model-facing synthesis boundary
    (``SynthesisInput`` / ``SynthesisEvidence``). The test fails if either
    leaks into a projection.
    """
    import json

    from app.services.agents.v2.complex_research_graph import (
        build_model_observations,
        build_model_replan_input,
    )
    from app.services.agents.v2.tools.adapters import AgentToolAdapter
    from app.services.agents.v2.tools.observations import ObservationProjector

    sentinel_scalar = "987654321098"
    sentinel_secret = "sentinel-workspace-" + WORKSPACE_ID.hex[:8]
    capabilities, leases, context = _harness(run_id="run-subagent-2")
    # R51: inject the KNOWN runtime secret into the request-scoped service
    # seam that a buggy model projection could otherwise serialize.
    context = GraphRuntimeContext(
        capability_runtime=context.capability_runtime,
        services=RuntimeServices(
            capability_registry=context.services.capability_registry,
            retention_leases=context.services.retention_leases,
            evidence_hydrator=context.services.evidence_hydrator,
            authorization={
                "runtime_secret": sentinel_secret,
                "workspace_ids": [
                    str(context.capability_runtime.workspace_ids[0])
                ],
                "can_read_people": context.capability_runtime.can_read_people,
                "allowed_capabilities": sorted(
                    context.capability_runtime.allowed_capabilities
                ),
                "deadline_at": str(context.capability_runtime.deadline_at),
            },
        ),
    )
    # Non-vacuous: the secret is genuinely present in the runtime seam.
    assert sentinel_secret in json.dumps(context.services.authorization)

    base = _two_target_plan()
    sentinel_plan = base.model_copy(
        update={
            "tasks": (
                base.tasks[0],
                base.tasks[1].model_copy(
                    update={
                        "capability": "document.search",
                        "input": DocumentSearchInput(
                            kind="document.search",
                            query="nghi dinh",
                            person_identifier=sentinel_scalar,
                        ),
                    }
                ),
            )
        }
    )
    # Non-vacuous: the People scalar is genuinely present in the plan.
    assert sentinel_scalar in sentinel_plan.model_dump_json()
    child = _child_input(
        plan=sentinel_plan,
        task_results=(
            AgentResult(
                contract_version="2.0",
                task_id="T1",
                status="success",
                data=PeopleLookupOutput(kind="people.lookup", matched=True),
                evidence_uses=(EvidenceUseRef(use_id=uuid4()),),
                coverage_observations=(),
                error=None,
            ),
        ),
        people_scalar_available={"T1": True},
    )
    advisory_payloads = [
        build_model_replan_input(child, context).model_dump_json(),
        ObservationProjector.project(
            child["task_results"][0], dependency_scalar_available=True
        ).model_dump_json(),
        "".join(
            observation.model_dump_json()
            for observation in build_model_observations(child)
        ),
    ]
    for payload in advisory_payloads:
        assert sentinel_scalar not in payload
        assert sentinel_secret not in payload
        for secret in (
            "workspace_ids",
            "can_read_people",
            "allowed_capabilities",
            "deadline_at",
            "storage_key",
            "runtime_secret",
        ):
            assert secret not in payload
    assert sentinel_scalar in child["plan"].model_dump_json()

    adapter = AgentToolAdapter(context.services.capability_registry)
    runtime_only = {
        "request_id",
        "run_id",
        "user_id",
        "workspace_ids",
        "can_read_people",
        "allowed_capabilities",
        "deadline_at",
        "runtime_secret",
    }
    for name in adapter.visible_tool_names():
        assert not (set(adapter.input_fields(name)) & runtime_only), name

    # R53: the model-facing synthesis boundary must carry neither sentinel.
    # Seed a governed evidence use for T1 so the real synthesis path can
    # hydrate a projection, then capture exactly what the DraftBuilder seam
    # receives (SynthesisInput + SynthesisEvidence).
    from app.services.agents.v2.contracts.synthesis import SynthesisInput
    from app.services.agents.v2.nodes.synthesize import (
        DEFAULT_SYNTHESIS_BUDGET,
        build_extractive_draft,
        synthesize_answer,
    )

    synthesis_use_id = uuid4()
    capabilities[0].uses[synthesis_use_id] = ("T1", "t1")
    sufficient = EvidenceEvaluation(
        status="sufficient",
        coverage=Coverage(items=()),
        missing=(),
        contradictions=(),
    )
    synthesis_input = SynthesisInput(
        semantic=_semantic(),
        evaluation=sufficient,
        evidence_uses=(EvidenceUseRef(use_id=synthesis_use_id),),
    )

    class _CapturingDraftBuilder:
        def __init__(self) -> None:
            self.input: SynthesisInput | None = None
            self.evidence: tuple = ()

        def build(self, synthesis_input, evidence):  # type: ignore[no-untyped-def]
            self.input = synthesis_input
            self.evidence = tuple(evidence)
            return build_extractive_draft(tuple(evidence))

    builder = _CapturingDraftBuilder()
    await synthesize_answer(
        synthesis_input=synthesis_input,
        runtime=context,
        plan=sentinel_plan,
        bindings=_bindings(),
        budget=DEFAULT_SYNTHESIS_BUDGET,
        draft_builder=builder,
    )
    assert builder.input is not None
    assert sentinel_scalar not in builder.input.model_dump_json()
    assert sentinel_secret not in builder.input.model_dump_json()
    for item in builder.evidence:
        assert sentinel_scalar not in item.model_dump_json()
        assert sentinel_secret not in item.model_dump_json()


def test_no_duplicate_test_function_names() -> None:
    """R49 recurrence guard: no duplicate test definitions may shadow a proof.

    Parses THIS module and asserts every ``test_`` function name is defined
    exactly once. A re-introduced duplicate (which Python silently lets
    shadow the earlier definition) fails this guard instead of producing a
    false-green run.
    """
    import ast

    source = Path(__file__).read_text()
    tree = ast.parse(source)
    seen: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                seen.setdefault(node.name, []).append(node.lineno)
    duplicates = {
        name: linenos for name, linenos in seen.items() if len(linenos) > 1
    }
    assert not duplicates, (
        f"duplicate test function definitions shadow earlier proofs: {duplicates}"
    )
