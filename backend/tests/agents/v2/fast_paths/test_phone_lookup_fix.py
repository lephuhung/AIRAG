"""Live phone-lookup vertical fix: exact phone -> Mongo ``people.lookup`` (TDD).

Live defect: authenticated ``Tra cứu số điện thoại 0989755968`` routed
``QueryAnalysis`` ``retrieve``/``document`` and dispatched a T1
``document.retrieve``. Direct ``search_by_phone`` finds 3 distinct person
groups (schemas ``bhxh``/``vnvc``). Required behavior: an exact unscoped
people-identifier query deterministically routes ``people.lookup``; the
adapter dispatches the matching Mongo ``search_by_phone`` (CCCD/BHXH/name
consistently); ALL distinct people are returned (never first-only); there
is no ``document.retrieve`` fallback on ``not_found``; explicit document
refs keep document semantics authoritative.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    PeopleLookupInput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evidence import EvidenceUseRef, PeopleSourceIdentity
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.semantic import DocumentReference, SemanticContext
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.synthesis import SynthesisRuntimeContext
from app.services.agents.v2.execution.scheduler import TaskScheduler
from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel, HydratedEvidence
from app.services.agents.v2.nodes.evaluate import evaluate_node
from app.services.agents.v2.nodes.finalizer import finalizer_node
from app.services.agents.v2.nodes.fast_plan import build_fast_plan
from app.services.agents.v2.nodes.grounding import ground_node
from app.services.agents.v2.nodes.routing import analyze_query, decide_route
from app.services.agents.v2.nodes.synthesize import (
    apply_budget_split,
    synthesize_node,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
REVISION = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

LIVE_QUERY = "Tra cứu số điện thoại 0989755968"
LIVE_PHONE = "0989755968"

FULL_FAST_CAPABILITIES = frozenset(
    {
        "people.lookup",
        "document.search",
        "document.retrieve",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
    }
)

#: Live-shaped heterogeneous Mongo persons: bhxh + vnvc field aliases, with
#: unrelated PII (DOB/address/BHXH number) that must never reach the checkpoint.
LIVE_PERSONS = [
    {
        "_id": "aaa",
        "_source_schema": "bhxh",
        "_person_group": 1,
        "hoTen": "Nguyễn Văn An",
        "soDienThoai": LIVE_PHONE,
        "maSoBhxh": "1234567890",
        "ngaySinhHienThi": "01/01/1980",
        "diaChi": "Số 1 Phố Huế, Hà Nội",
    },
    {
        "_id": "bbb",
        "_source_schema": "vnvc",
        "_person_group": 2,
        "fullName": "Trần Thị Bình",
        "mobile": LIVE_PHONE,
        "diaChi": "Số 2 Lê Lợi, Đà Nẵng",
        "fullNam": "02/02/1990",
    },
    {
        "_id": "ccc",
        "_source_schema": "bhxh",
        "_person_group": 3,
        "hoTen": "Lê Văn Cường",
        "soDienThoai": LIVE_PHONE,
        "soCmnd": "079203012345",
        "ngaySinhHienThi": "03/03/1975",
        "diaChi": "Số 3 Hùng Vương, Huế",
    },
]


def phone_search_stub(calls: list, persons: list[dict] | None = None):
    """Injectable ``search_by_phone`` async-generator stand-in (spy)."""

    async def _search(phone: str, limit: int = 10):
        calls.append((phone, limit))
        found = persons if persons is not None else LIVE_PERSONS
        if found:
            yield {
                "found": True,
                "persons": [dict(p) for p in found],
                "schemas": sorted({p["_source_schema"] for p in found}),
                "lookup_type": "phone",
            }
        yield {"found": False, "persons": [], "display": "Không tìm thấy"}

    return _search


def name_search_spy(calls: list):
    async def _search(name: str, limit: int = 10):
        calls.append((name, limit))
        yield {"found": False, "persons": [], "display": "Không tìm thấy"}

    return _search


class FakeEvidenceBuilder:
    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.calls: list[dict] = []

    async def persist_use(
        self, *, source, content, provenance, task_id, purpose, target_id
    ) -> EvidenceUseRef:
        from uuid import uuid5

        evidence_id = uuid5(
            UUID("99999999-9999-9999-9999-999999999999"),
            source.model_dump_json() + ":" + content,
        )
        use_id = uuid5(
            UUID("99999999-9999-9999-9999-999999999999"),
            f"{self.run_id}:{task_id}:{evidence_id}:{purpose}:{target_id}",
        )
        self.calls.append(
            {
                "source": source,
                "content": content,
                "provenance": provenance,
                "task_id": task_id,
                "purpose": purpose,
                "target_id": target_id,
                "evidence_id": evidence_id,
                "use_id": use_id,
            }
        )
        return EvidenceUseRef(use_id=use_id)


class FakeLeaseSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class FakeLeaseRepo:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.calls: list[tuple] = []
        self.session = FakeLeaseSession(self.events)

    async def acquire_or_refresh(self, run_id, revision_id=None, evidence_use_id=None, **kwargs):
        self.calls.append((run_id, revision_id, evidence_use_id))
        return None


class FakeHydrator:
    def __init__(self, items: dict) -> None:
        self.items = dict(items)
        self.calls: list[str] = []

    async def hydrate_for_evaluation(self, use_refs, *, runtime, plan, bindings):
        self.calls.append("evaluation")
        return tuple(self.items[r.use_id] for r in use_refs if r.use_id in self.items)

    async def hydrate_for_synthesis(self, use_refs, *, runtime, plan, bindings, budget):
        self.calls.append("synthesis")
        admitted = [self.items[r.use_id] for r in use_refs if r.use_id in self.items]
        admitted = [h for h in admitted if h.purpose != "discovery"]
        head, _ = apply_budget_split(tuple(admitted), budget)
        return tuple(head)

    async def persist_derived_summary(self, **kwargs):
        raise AssertionError("no overflow expected")


class SpyDocumentRetrieveCapability:
    """Zero-call guard: any document.retrieve dispatch fails the test."""

    descriptor = CapabilityDescriptor(
        name="document.retrieve",
        domain="document",
        operation_type="search",
        supports_parallel=False,
    )

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def execute(self, request: AgentRequest, runtime: CapabilityRuntimeContext) -> AgentResult:
        self.calls.append((request, runtime))
        raise AssertionError("document.retrieve must never dispatch for a phone lookup")


def capability_runtime(
    *,
    allowed: frozenset[str] = FULL_FAST_CAPABILITIES,
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


def semantic_for(query: str) -> SemanticContext:
    return SemanticContext(
        contextualized_query=query,
        normalized_query=query,
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def empty_bindings() -> DocumentBindingSet:
    return DocumentBindingSet(bindings=(), revision_requirement_refs=())


def make_state(
    *,
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    analysis=None,
    route=None,
    execution: ExecutionState | None = None,
) -> SupervisorV2State:
    return SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query=semantic.normalized_query,
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="", active_entities=(), last_focus=None, recent_turns=()
        ),
        semantic=semantic,
        bindings=bindings,
        query_analysis=analysis,
        route_decision=route,
        execution=execution
        if execution is not None
        else ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )


def full_context(*, registry, hydrator, allowed=FULL_FAST_CAPABILITIES):
    from app.services.agent.runtime_selector import PlanBindingResolver

    leases = FakeLeaseRepo()
    channel = AnswerDraftChannel()
    return (
        GraphRuntimeContext(
            capability_runtime=capability_runtime(allowed=allowed),
            services=RuntimeServices(
                capability_registry=registry,
                retention_leases=leases,
                evidence_hydrator=hydrator,
                answer_draft_channel=channel,
                pinned_target_resolver=PlanBindingResolver(),
            ),
        ),
        leases,
        channel,
    )


def hydrated_people(use_id, evidence_id, task_id, content, record_id) -> HydratedEvidence:
    return HydratedEvidence(
        use_id=use_id,
        evidence_id=evidence_id,
        task_id=task_id,
        purpose="supporting",
        target_id=None,
        content=content,
        role=None,
        source_label="people",
        source_identity=PeopleSourceIdentity(kind="people", record_id=record_id),
        classification="personal",
        locator=None,
    )


# ---------------------------------------------------------------------------
# Route: exact unscoped phone query -> people.lookup (never document.retrieve)
# ---------------------------------------------------------------------------


def test_live_phone_query_routes_people_lookup_not_document() -> None:
    semantic = semantic_for(LIVE_QUERY)
    bindings = empty_bindings()
    analysis = analyze_query(semantic)
    assert analysis.work_type == "lookup"
    assert tuple(analysis.domains) == ("people",)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert (route.route, route.reason_code) == ("fast_domain", "simple_people_lookup")
    plan = build_fast_plan(semantic, bindings, analysis, route)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].capability == "people.lookup"


# ---------------------------------------------------------------------------
# Vertical: Mongo search_by_phone -> ALL 3 distinct governed people
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_phone_end_to_end_returns_all_three_people() -> None:
    from app.services.agents.supervisor_v2 import (
        V1PeopleLookupService,
        V1PeopleMultiMatchAdapter,
    )
    from app.services.agents.v2.capabilities import PeopleCapability

    phone_calls: list = []
    doc_retrieve = SpyDocumentRetrieveCapability()
    stub = phone_search_stub(phone_calls)
    service = V1PeopleLookupService(phone_lookup=stub)
    adapter = V1PeopleMultiMatchAdapter(phone_lookup=stub)
    evidence = FakeEvidenceBuilder()
    people = PeopleCapability(
        service=service, evidence=evidence, multi_match=adapter
    )
    runtime = capability_runtime()
    registry = build_capability_registry(
        [
            CapabilityRegistration(capability=people),
            CapabilityRegistration(capability=doc_retrieve),
        ],
        runtime,
    )

    semantic = semantic_for(LIVE_QUERY)
    bindings = empty_bindings()
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    plan = build_fast_plan(semantic, bindings, analysis, route)
    assert plan.tasks[0].capability == "people.lookup"

    hydrator = FakeHydrator({})
    context, _, channel = full_context(registry=registry, hydrator=hydrator)
    report = await TaskScheduler(registry).execute(
        plan=plan, runtime=context, prior_results=(), bindings=bindings
    )
    assert len(report.results) == 1
    result = report.results[0]
    assert result.status == "success"
    # search_by_phone dispatched exactly once; document provider never called.
    assert len(phone_calls) == 1
    assert LIVE_PHONE in phone_calls[0][0]
    assert doc_retrieve.calls == []
    # ALL 3 distinct people become governed EvidenceUses (never first-only).
    assert len(result.evidence_uses) == 3
    assert len({ref.use_id for ref in result.evidence_uses}) == 3
    assert len(evidence.calls) == 3
    for call in evidence.calls:
        assert call["source"].kind == "people"
        assert call["purpose"] == "supporting"

    # Governed hydration + synthesis answer all 3 with sources.
    items = {}
    for call, ref in zip(evidence.calls, result.evidence_uses):
        items[ref.use_id] = hydrated_people(
            ref.use_id, call["evidence_id"], plan.tasks[0].task_id,
            call["content"], call["source"].record_id,
        )
    hydrator.items.update(items)
    state = make_state(
        semantic=semantic, bindings=bindings, analysis=analysis, route=route,
        execution=ExecutionState(
            plan=plan, task_results=report.results, evidence_evaluation=None
        ),
    )
    evaluated = await evaluate_node(state, context)
    assert evaluated["execution"].evidence_evaluation.status == "sufficient"
    state["execution"] = evaluated["execution"]
    # Task 6B: people_card presentation runs inline at the synthesize
    # boundary (no LLM, no SynthesisCheckpoint) — the existing non-LLM
    # presentation stores its grounded result in the channel and the
    # finalizer replays it with its rendered citations.
    from app.services.agents.v2.contracts.response import RenderedCitation
    from app.services.agents.v2.contracts.synthesis import AnswerClaim, AnswerDraft

    people_content = "Nguyễn Văn An\nTrần Thị Bình\nLê Văn Cường"
    channel = context.services.answer_draft_channel
    channel.store_grounded(
        "run-1",
        draft=AnswerDraft(
            content=people_content,
            claims=(
                AnswerClaim(
                    claim_id="claim-1",
                    text=people_content,
                    evidence_use_ids=tuple(
                        ref.use_id for ref in result.evidence_uses
                    ),
                ),
            ),
        ),
        citations=tuple(
            RenderedCitation(
                citation_id=f"cite-{i}",
                evidence_id=call["evidence_id"],
                label="people",
            )
            for i, call in enumerate(evidence.calls, start=1)
        ),
    )
    final = await finalizer_node(state, context)
    assert final["final_response"].status == "success"
    content = final["final_response"].content
    assert "Nguyễn Văn An" in content
    assert "Trần Thị Bình" in content
    assert "Lê Văn Cường" in content
    assert len(final["final_response"].citations) == 3

    # No raw Mongo record / unrelated PII in the checkpoint or evidence content.
    checkpoint_json = result.model_dump_json()
    for forbidden in ("01/01/1980", "Phố Huế", "1234567890", "079203012345", "_person_group", "soDienThoai"):
        assert forbidden not in checkpoint_json, forbidden
    for call in evidence.calls:
        assert "01/01/1980" not in call["content"]
        assert "Phố Huế" not in call["content"]
        assert "1234567890" not in call["content"]


@pytest.mark.asyncio
async def test_phone_capability_uses_search_by_phone_not_search_by_name() -> None:
    from app.services.agents.supervisor_v2 import (
        V1PeopleLookupService,
        V1PeopleMultiMatchAdapter,
    )
    from app.services.agents.v2.capabilities import PeopleCapability

    phone_calls: list = []
    name_calls: list = []
    phone_stub = phone_search_stub(phone_calls)
    name_stub = name_search_spy(name_calls)
    service = V1PeopleLookupService(
        phone_lookup=phone_stub,
        name_lookup=name_stub,
    )
    adapter = V1PeopleMultiMatchAdapter(
        phone_lookup=phone_stub,
        name_lookup=name_stub,
    )
    evidence = FakeEvidenceBuilder()
    capability = PeopleCapability(
        service=service, evidence=evidence, multi_match=adapter
    )
    result = await capability.execute(
        AgentRequest(
            contract_version="2.0", task_id="T1", objective=LIVE_QUERY,
            input=PeopleLookupInput(kind="people.lookup", query=LIVE_QUERY),
        ),
        capability_runtime(),
    )
    assert result.status == "success"
    assert len(phone_calls) == 1
    assert name_calls == []


# ---------------------------------------------------------------------------
# Typed edges: not_found, ACL, explicit-doc authority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phone_not_found_has_no_document_fallback() -> None:
    from app.services.agents.supervisor_v2 import (
        V1PeopleLookupService,
        V1PeopleMultiMatchAdapter,
    )
    from app.services.agents.v2.capabilities import PeopleCapability

    phone_calls: list = []
    doc_retrieve = SpyDocumentRetrieveCapability()
    stub = phone_search_stub(phone_calls, persons=[])
    service = V1PeopleLookupService(phone_lookup=stub)
    adapter = V1PeopleMultiMatchAdapter(phone_lookup=stub)
    evidence = FakeEvidenceBuilder()
    people = PeopleCapability(
        service=service, evidence=evidence, multi_match=adapter
    )
    runtime = capability_runtime()
    registry = build_capability_registry(
        [
            CapabilityRegistration(capability=people),
            CapabilityRegistration(capability=doc_retrieve),
        ],
        runtime,
    )
    semantic = semantic_for("Tra cứu số điện thoại 0900000000")
    bindings = empty_bindings()
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert (route.route, route.reason_code) == ("fast_domain", "simple_people_lookup")
    plan = build_fast_plan(semantic, bindings, analysis, route)
    hydrator = FakeHydrator({})
    context, _, _ = full_context(registry=registry, hydrator=hydrator)
    report = await TaskScheduler(registry).execute(
        plan=plan, runtime=context, prior_results=(), bindings=bindings
    )
    assert report.results[0].status == "not_found"
    assert report.results[0].evidence_uses == ()
    assert doc_retrieve.calls == []

    state = make_state(
        semantic=semantic, bindings=bindings, analysis=analysis, route=route,
        execution=ExecutionState(
            plan=plan, task_results=report.results, evidence_evaluation=None
        ),
    )
    evaluated = await evaluate_node(state, context)
    assert evaluated["execution"].evidence_evaluation.status == "insufficient"
    state["execution"] = evaluated["execution"]
    final = await finalizer_node(state, context)
    assert final["final_response"].status == "insufficient"
    assert doc_retrieve.calls == []


@pytest.mark.asyncio
async def test_phone_lookup_denied_without_can_read_people() -> None:
    from app.services.agents.supervisor_v2 import V1PeopleLookupService
    from app.services.agents.v2.capabilities import PeopleCapability

    phone_calls: list = []
    service = V1PeopleLookupService(phone_lookup=phone_search_stub(phone_calls))
    capability = PeopleCapability(service=service, evidence=FakeEvidenceBuilder())
    result = await capability.execute(
        AgentRequest(
            contract_version="2.0", task_id="T1", objective=LIVE_QUERY,
            input=PeopleLookupInput(kind="people.lookup", query=LIVE_QUERY),
        ),
        capability_runtime(can_read_people=False),
    )
    assert result.status == "denied"
    assert phone_calls == []


def test_explicit_doc_phone_query_stays_document() -> None:
    semantic = SemanticContext(
        contextualized_query="Số điện thoại 0989755968 trong văn bản này là của ai?",
        normalized_query="Số điện thoại 0989755968 trong văn bản này là của ai?",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="văn bản này",
                normalized_reference="văn bản này",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOCUMENT_ID,
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
                binding_id="b_r1", document_id=DOCUMENT_ID,
                document_revision=REVISION, role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    analysis = analyze_query(semantic)
    assert "people" not in tuple(analysis.domains)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert (route.route, route.reason_code) != ("fast_domain", "simple_people_lookup")


# ---------------------------------------------------------------------------
# Canonicalization: bhxh/vnvc aliases, dedupe, stable non-PII record_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_normalization_and_grouped_dedupe() -> None:
    from app.services.agents.supervisor_v2 import V1PeopleLookupService

    grouped = [
        dict(LIVE_PERSONS[0]),
        dict(LIVE_PERSONS[0], _id="aaa-dup"),
        dict(LIVE_PERSONS[1]),
    ]
    phone_calls: list = []
    service = V1PeopleLookupService(phone_lookup=phone_search_stub(phone_calls, persons=grouped))
    matches = await service._lookup_many(LIVE_QUERY)
    assert len(matches) == 2
    by_name = {m.fields["name"]: m for m in matches}
    assert by_name["Nguyễn Văn An"].fields["phone"] == LIVE_PHONE
    assert by_name["Trần Thị Bình"].fields["phone"] == LIVE_PHONE
    assert by_name["Nguyễn Văn An"].fields["source"] == "bhxh"
    assert by_name["Trần Thị Bình"].fields["source"] == "vnvc"
    # Stable non-PII record ids: deterministic, no raw identity substrings.
    again = await service._lookup_many(LIVE_QUERY)
    assert [m.record_id for m in again] == [m.record_id for m in matches]
    for match in matches:
        assert "Nguyễn" not in match.record_id
        assert LIVE_PHONE not in match.record_id
        assert set(match.fields) <= {"name", "phone", "source"}
    assert matches[0].record_id != matches[1].record_id


@pytest.mark.asyncio
async def test_cccd_bhxh_dispatch_and_name_preserved() -> None:
    from app.services.agents.supervisor_v2 import V1PeopleLookupService

    async def _cccd_search(cccd: str, limit: int = 10):
        yield {
            "found": True,
            "persons": [
                {"hoTen": "Nguyễn Văn An", "soCmnd": "079203012345",
                 "_source_schema": "bhxh", "_person_group": 1}
            ],
            "lookup_type": "cccd",
        }
        yield {"found": False, "persons": []}

    async def _bhxh_search(so_bhxh: str, limit: int = 10):
        yield {
            "found": True,
            "persons": [
                {"hoTen": "Nguyễn Văn An", "maSoBhxh": "1234567890",
                 "_source_schema": "bhxh", "_person_group": 1}
            ],
            "lookup_type": "bhxh",
        }
        yield {"found": False, "persons": []}

    async def _name_search(name: str, limit: int = 10):
        assert "earched" not in name  # sanity: real name text passes through
        yield {
            "found": True,
            "persons": [
                {"hoTen": "Nguyễn Văn An", "soDienThoai": "0989000111",
                 "_source_schema": "bhxh", "_person_group": 1}
            ],
            "lookup_type": "name",
        }
        yield {"found": False, "persons": []}

    cccd_service = V1PeopleLookupService(
        cccd_lookup=_cccd_search, bhxh_lookup=_bhxh_search,
        name_lookup=_name_search,
        phone_lookup=phone_search_stub([]),
    )
    cccd_matches = await cccd_service._lookup_many("Tra cứu CCCD 079203012345")
    assert len(cccd_matches) == 1
    assert cccd_matches[0].fields["name"] == "Nguyễn Văn An"
    bhxh_matches = await cccd_service._lookup_many("Tra cứu BHXH 1234567890")
    assert len(bhxh_matches) == 1

    # Existing name behavior preserved: classifier routes, name search serves.
    name_semantic = semantic_for("Tìm ông Nguyễn Văn An")
    name_analysis = analyze_query(name_semantic)
    assert tuple(name_analysis.domains) == ("people",)
    name_route = decide_route(
        name_analysis, name_semantic, empty_bindings(),
        allowed_capabilities=FULL_FAST_CAPABILITIES,
    )
    assert (name_route.route, name_route.reason_code) == (
        "fast_domain", "simple_people_lookup"
    )
    name_matches = await cccd_service._lookup_many("Tìm ông Nguyễn Văn An")
    assert len(name_matches) == 1
    assert name_matches[0].fields["name"] == "Nguyễn Văn An"


def test_people_evidence_classification_floor_is_personal() -> None:
    from app.services.agents.v2.evidence_store.governance import classify_evidence

    assert (
        classify_evidence(
            PeopleSourceIdentity(kind="people", record_id="p_deadbeef"),
            field_names=("name", "phone", "source"),
        )
        == "personal"
    )


# ---------------------------------------------------------------------------
# Hardening (C1/I2): CCCD/BHXH fakes mirror the EXACT production signatures
# (no ``limit``) so a limit-passing dispatch fails loudly instead of being
# certified as working.
# ---------------------------------------------------------------------------


def test_production_search_signatures_limit_parity() -> None:
    """Real ``_v1_attr`` search functions: only phone/name accept ``limit``."""
    import inspect

    from app.services.agents.supervisor_v2 import V1ServiceUnavailable, _v1_attr

    try:
        cccd = _v1_attr("app.services.people.mongo_people_service", "search_by_cccd")
    except V1ServiceUnavailable as exc:
        pytest.skip(f"v1 mongo module unavailable in this env: {exc}")
    bhxh = _v1_attr("app.services.people.mongo_people_service", "search_by_bhxh")
    phone = _v1_attr("app.services.people.mongo_people_service", "search_by_phone")
    name = _v1_attr("app.services.people.mongo_people_service", "search_by_name")
    assert "limit" not in inspect.signature(cccd).parameters
    assert "limit" not in inspect.signature(bhxh).parameters
    assert "limit" in inspect.signature(phone).parameters
    assert "limit" in inspect.signature(name).parameters


@pytest.mark.asyncio
async def test_cccd_bhxh_exact_production_signatures_dispatch() -> None:
    """CCCD/BHXH searches are invoked WITHOUT ``limit`` (C1)."""
    from app.services.agents.supervisor_v2 import V1PeopleLookupService

    cccd_calls: list = []
    bhxh_calls: list = []

    async def _cccd_search(cccd: str):
        cccd_calls.append((cccd,))
        yield {
            "found": True,
            "persons": [
                {"hoTen": "Nguyễn Văn An", "soCmnd": "079203012345",
                 "_source_schema": "bhxh", "_person_group": 1}
            ],
            "lookup_type": "cccd",
        }
        yield {"found": False, "persons": []}

    async def _bhxh_search(so_bhxh: str):
        bhxh_calls.append((so_bhxh,))
        yield {
            "found": True,
            "persons": [
                {"hoTen": "Nguyễn Văn An", "maSoBhxh": "1234567890",
                 "_source_schema": "bhxh", "_person_group": 1}
            ],
            "lookup_type": "bhxh",
        }
        yield {"found": False, "persons": []}

    service = V1PeopleLookupService(
        cccd_lookup=_cccd_search,
        bhxh_lookup=_bhxh_search,
        phone_lookup=phone_search_stub([]),
    )
    cccd_matches = await service._lookup_many("Tra cứu CCCD 079203012345")
    assert len(cccd_matches) == 1
    assert cccd_matches[0].fields["name"] == "Nguyễn Văn An"
    assert len(cccd_calls) == 1
    assert "079203012345" in cccd_calls[0][0]
    bhxh_matches = await service._lookup_many("Tra cứu BHXH 1234567890")
    assert len(bhxh_matches) == 1
    assert bhxh_matches[0].fields["name"] == "Nguyễn Văn An"
    assert len(bhxh_calls) == 1


# ---------------------------------------------------------------------------
# Hardening (I3): a Mongo match without a name (``uids`` phone schema) is
# never silently dropped: placeholder identity, exact phone+source evidence.
# ---------------------------------------------------------------------------

UIDS_PHONE = "0989111222"

UIDS_PERSONS = [
    {
        "_id": "u1",
        "_source_schema": "uids",
        "_person_group": 7,
        "uid": "1000123",
        "phone": UIDS_PHONE,
    },
    {
        "_id": "u2",
        "_source_schema": "uids",
        "_person_group": 8,
        "uid": "1000456",
        "phone": UIDS_PHONE,
    },
]


def uids_search_stub(calls: list, persons: list[dict] | None = None):
    """Injectable ``search_by_phone`` stand-in yielding name-less records."""

    async def _search(phone: str, limit: int = 10):
        calls.append((phone, limit))
        found = persons if persons is not None else UIDS_PERSONS
        if found:
            yield {
                "found": True,
                "persons": [dict(p) for p in found],
                "schemas": sorted({p["_source_schema"] for p in found}),
                "lookup_type": "phone",
            }
        yield {"found": False, "persons": [], "display": "Không tìm thấy"}

    return _search


@pytest.mark.asyncio
async def test_uids_nameless_phone_match_is_preserved_with_placeholder() -> None:
    """Name-less ``uids`` matches succeed with a placeholder identity (I3)."""
    from app.services.agents.supervisor_v2 import V1PeopleLookupService
    from app.services.agents.v2.capabilities import PeopleCapability

    phone_calls: list = []
    stub = uids_search_stub(phone_calls)
    service = V1PeopleLookupService(phone_lookup=stub)
    matches = await service._lookup_many(
        f"Tra cứu số điện thoại {UIDS_PHONE}"
    )
    # Both Mongo groups survive (exact ``_person_group`` preserved in dedupe).
    assert len(matches) == 2
    assert matches[0].record_id != matches[1].record_id
    for match in matches:
        assert match.fields["name"] == "Không rõ tên"
        assert match.fields["phone"] == UIDS_PHONE
        assert match.fields["source"] == "uids"
        assert "Không rõ" not in match.record_id
        assert UIDS_PHONE not in match.record_id

    # End to end: found => success evidence, never ``not_found``.
    from app.services.agents.supervisor_v2 import V1PeopleMultiMatchAdapter
    from app.services.agents.v2.capabilities import PeopleCapability

    evidence = FakeEvidenceBuilder()
    capability = PeopleCapability(
        service=service,
        evidence=evidence,
        multi_match=V1PeopleMultiMatchAdapter(phone_lookup=stub),
    )
    result = await capability.execute(
        AgentRequest(
            contract_version="2.0", task_id="T1",
            objective=f"Tra cứu số điện thoại {UIDS_PHONE}",
            input=PeopleLookupInput(
                kind="people.lookup",
                query=f"Tra cứu số điện thoại {UIDS_PHONE}",
            ),
        ),
        capability_runtime(),
    )
    assert result.status == "success"
    assert len(result.evidence_uses) == 2
    for call in evidence.calls:
        assert UIDS_PHONE in call["content"]
        assert "uids" in call["content"]
    checkpoint_json = result.model_dump_json()
    assert "1000123" not in checkpoint_json
    assert "1000456" not in checkpoint_json


# ---------------------------------------------------------------------------
# Hardening: limit is still passed where declared; private seam is dead;
# malformed matches never leave partial rows; DOB widens the dedupe key.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phone_search_still_receives_limit() -> None:
    """Phone/name searches keep the explicit ``limit=`` kwarg."""
    from app.services.agents.supervisor_v2 import V1PeopleMultiMatchAdapter

    phone_calls: list = []

    async def _strict_phone_search(phone: str, limit: int):
        # No default: ``limit`` MUST be passed explicitly (kills the
        # never-pass-limit over-correction of the C1 fix).
        phone_calls.append((phone, limit))
        async for result in phone_search_stub([], persons=LIVE_PERSONS)(
            phone, limit=limit
        ):
            yield result

    adapter = V1PeopleMultiMatchAdapter(
        phone_lookup=_strict_phone_search, limit=25
    )
    matches = await adapter.lookup_many(LIVE_QUERY)
    assert len(matches) == 3
    assert len(phone_calls) == 1
    assert phone_calls[0][1] == 25


@pytest.mark.asyncio
async def test_capability_never_probes_private_lookup_many() -> None:
    """A service exposing only ``_lookup_many`` falls back to legacy (M6)."""
    from app.services.agents.v2.capabilities import PeopleCapability

    private_calls: list = []

    class PrivateOnlyService:
        async def _lookup_many(self, query: str):
            private_calls.append(query)
            raise AssertionError("private seam must never dispatch")

        async def lookup(self, query: str):
            return None

    capability = PeopleCapability(
        service=PrivateOnlyService(), evidence=FakeEvidenceBuilder()
    )
    result = await capability.execute(
        AgentRequest(
            contract_version="2.0", task_id="T1", objective=LIVE_QUERY,
            input=PeopleLookupInput(kind="people.lookup", query=LIVE_QUERY),
        ),
        capability_runtime(),
    )
    assert result.status == "not_found"
    assert private_calls == []


@pytest.mark.asyncio
async def test_malformed_match_fails_closed_with_zero_persists() -> None:
    """A malformed kth match persists nothing (M4 prevalidation)."""
    from types import SimpleNamespace

    from app.services.agents.v2.capabilities import PeopleCapability

    good = SimpleNamespace(
        record_id="p_good",
        fields={"name": "Nguyễn Văn An", "phone": LIVE_PHONE,
                "source": "bhxh"},
        required_fields=("name", "phone", "source"),
    )
    bad = SimpleNamespace(
        record_id="",  # malformed: no resolvable handle
        fields={"name": "Trần Thị Bình", "phone": LIVE_PHONE,
                "source": "vnvc"},
        required_fields=("name", "phone", "source"),
    )

    class CannedAdapter:
        async def lookup_many(self, query: str):
            return [good, bad]

    class LegacyService:
        async def lookup(self, query: str):
            raise AssertionError("multi-match path must be used")

    evidence = FakeEvidenceBuilder()
    capability = PeopleCapability(
        service=LegacyService(), evidence=evidence,
        multi_match=CannedAdapter(),
    )
    result = await capability.execute(
        AgentRequest(
            contract_version="2.0", task_id="T1", objective=LIVE_QUERY,
            input=PeopleLookupInput(kind="people.lookup", query=LIVE_QUERY),
        ),
        capability_runtime(),
    )
    assert result.status == "error"
    assert evidence.calls == []


@pytest.mark.asyncio
async def test_dedupe_key_splits_on_dob_without_persisting_it() -> None:
    """Same name+phone but different DOB are distinct people (M5)."""
    from app.services.agents.supervisor_v2 import V1PeopleMultiMatchAdapter

    persons = [
        {"_id": "d1", "_source_schema": "bhxh", "hoTen": "Nguyễn Văn An",
         "soDienThoai": LIVE_PHONE, "ngaySinhHienThi": "01/01/1980"},
        {"_id": "d2", "_source_schema": "bhxh", "hoTen": "Nguyễn Văn An",
         "soDienThoai": LIVE_PHONE, "ngaySinhHienThi": "02/02/1990"},
        {"_id": "d3", "_source_schema": "bhxh", "hoTen": "Nguyễn Văn An",
         "soDienThoai": LIVE_PHONE, "ngaySinhHienThi": "02/02/1990"},
    ]
    adapter = V1PeopleMultiMatchAdapter(
        phone_lookup=phone_search_stub([], persons=persons)
    )
    matches = await adapter.lookup_many(LIVE_QUERY)
    assert len(matches) == 2
    assert matches[0].record_id != matches[1].record_id
    for match in matches:
        assert set(match.fields) <= {"name", "phone", "source"}
        assert "1980" not in str(match.fields)
        assert "1990" not in str(match.fields)
