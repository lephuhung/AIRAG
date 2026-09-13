"""Phase 2 Task 2 — shared atomic capabilities and the request-scoped registry.

Boundary tests: every capability receives ONLY ``AgentRequest`` +
``CapabilityRuntimeContext``, never supervisor/graph state; People enforces the
current people permission with governed minimization; reads resolve the
planned ``TargetUnit`` + pinned revision through the constructor-injected
resolver and report ``read`` only on an observed/requested locator match;
section reads emit READ coverage (never search coverage); duplicate dependency
items dedupe to distinct uses/counts; the people scalar fails closed; the
registry intersects runtime permissions with feature flags/service
availability; and v2 ships no domain agent or domain graph wrappers.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import pytest
from pydantic import ValidationError

from app.services.agents.v2.capabilities import (
    Capability,
    CapabilityDenied,
    CapabilityNotRegistered,
    CapabilityRegistration,
    CapabilityUnavailable,
    LocatedContent,
    ResolvedTarget,
    build_capability_registry,
)
from app.services.agents.v2.contracts.binding import (
    DocumentDiscoveryCandidate,
    ScopedDocument,
)
from app.services.agents.v2.contracts.capability import (
    AbbreviationResolveInput,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentSearchInput,
    KnowledgeGraphInput,
    MemoryLookupInput,
    PeopleLookupInput,
    SectionReadInput,
)
from app.services.agents.v2.contracts.evidence import EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest
from app.services.agents.v2.contracts.locators import (
    DocumentLocator,
    PageRangeLocator,
    SectionLocator,
)
from app.services.agents.v2.contracts.planning import (
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
EVIDENCE_NS = UUID("99999999-9999-9999-9999-999999999999")


def runtime_context(
    *,
    allowed: frozenset[str] = frozenset(
        {
            "people.lookup",
            "document.search",
            "document.read",
            "section.read",
            "knowledge_graph.query",
            "memory.lookup",
            "abbreviation.resolve",
        }
    ),
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


def agent_request(task_id: str, objective: str, capability_input) -> AgentRequest:
    return AgentRequest(
        contract_version="2.0",
        task_id=task_id,
        objective=objective,
        input=capability_input,
    )


class FakeEvidenceBuilder:
    """Idempotent in-memory evidence builder (models the §15.3 use-append).

    ``use_id`` derives deterministically from
    ``(run_id, task_id, evidence_id, purpose, target_id)`` so a repeated
    persist of the same item returns the same ref instead of minting a
    duplicate the frozen validator would reject.
    """

    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.calls: list[dict] = []

    async def persist_use(
        self, *, source, content, provenance, task_id, purpose, target_id
    ) -> EvidenceUseRef:
        evidence_id = uuid5(
            EVIDENCE_NS,
            source.model_dump_json()
            + ":"
            + hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        use_id = uuid5(
            EVIDENCE_NS,
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


def pinned_binding() -> ScopedDocument:
    return ScopedDocument(
        binding_id="b_t1",
        document_id=DOCUMENT_ID,
        document_revision="rev-1",
        role="target",
    )


def resolved_target(
    target_id: str = "t1",
    locator=None,
    revision: str = "rev-1",
) -> ResolvedTarget:
    return ResolvedTarget(
        target_unit=TargetUnit(
            target_id=target_id,
            binding_id="b_t1",
            requested_locator=(
                locator
                if locator is not None
                else SectionLocator(kind="section", structure_node_id="node-5")
            ),
            completion_criteria=(),
        ),
        document=ScopedDocument(
            binding_id="b_t1",
            document_id=DOCUMENT_ID,
            document_revision=revision,
            role="target",
        ),
    )


class FakeTargetResolver:
    """Request-scoped resolver fed the authoritative plan/bindings by T6/T7."""

    def __init__(self, targets: dict[str, ResolvedTarget] | None = None) -> None:
        self._targets = dict(targets if targets is not None else {"t1": resolved_target()})

    def resolve(self, target_id: str) -> ResolvedTarget | None:
        return self._targets.get(target_id)


def all_capabilities(
    evidence: FakeEvidenceBuilder, resolver: FakeTargetResolver
) -> list:
    from app.services.agents.v2.capabilities import (
        AbbreviationCapability,
        DocumentReadCapability,
        DocumentSearchCapability,
        KnowledgeGraphCapability,
        MemoryCapability,
        PeopleCapability,
        SectionReadCapability,
    )

    return [
        PeopleCapability(
            service=FakePeopleService(
                {"nguyen van a": {"record_id": "p-1", "name": "Nguyen Van A"}}
            ),
            evidence=evidence,
            required_fields=("name",),
        ),
        DocumentSearchCapability(service=FakeSearchService(())),
        DocumentReadCapability(
            reader=FakeDocumentReader(),
            evidence=evidence,
            resolver=resolver,
        ),
        SectionReadCapability(
            reader=FakeSectionReader(),
            evidence=evidence,
            resolver=resolver,
        ),
        KnowledgeGraphCapability(
            client=FakeKgClient({"ai": (("e-1", "AI"),)}), evidence=evidence
        ),
        MemoryCapability(
            store=FakeMemoryStore({"hello": (("m-1", "hi"),)}), evidence=evidence
        ),
        AbbreviationCapability(service=FakeAbbreviationService({"nđ": "Nghị định"})),
    ]


class FakePeopleService:
    def __init__(self, records: dict[str, dict]) -> None:
        self.records = records
        self.calls: list[str] = []

    async def lookup(self, query: str):
        self.calls.append(query)
        return self.records.get(query.strip().lower())


class FakeSearchService:
    def __init__(self, candidates) -> None:
        self._candidates = tuple(candidates)
        self.calls: list[tuple] = []

    async def search(self, query: str, person_identifier, workspace_ids):
        self.calls.append((query, person_identifier, workspace_ids))
        return self._candidates


class FakeDocumentReader:
    """Correct reader by default: echoes the requested locator as observed."""

    def __init__(
        self,
        default_content: str | None = "doc-content",
        by_locator: dict[str, LocatedContent] | None = None,
    ) -> None:
        self.default_content = default_content
        self.by_locator = dict(by_locator or {})
        self.calls: list[tuple] = []

    async def read(self, binding: ScopedDocument, locator):
        self.calls.append((binding, locator))
        override = self.by_locator.get(locator.model_dump_json())
        if override is not None:
            return override
        if self.default_content is None:
            return LocatedContent(
                outcome="missing", observed_locator=None, content=None
            )
        return LocatedContent(
            outcome="read", observed_locator=locator, content=self.default_content
        )


class FakeSectionReader:
    """Correct reader by default: echoes the requested locator as observed."""

    def __init__(
        self,
        default_content: str | None = "section text",
        by_locator: dict[str, LocatedContent] | None = None,
    ) -> None:
        self.default_content = default_content
        self.by_locator = dict(by_locator or {})
        self.calls: list[tuple] = []

    async def read_section(self, binding: ScopedDocument, locator):
        self.calls.append((binding, locator))
        override = self.by_locator.get(locator.model_dump_json())
        if override is not None:
            return override
        if self.default_content is None:
            return LocatedContent(
                outcome="missing", observed_locator=None, content=None
            )
        return LocatedContent(
            outcome="read", observed_locator=locator, content=self.default_content
        )


class FakeKgClient:
    def __init__(self, answers: dict) -> None:
        self.answers = answers

    async def query(self, query: str):
        return self.answers.get(query, ())


class FakeMemoryStore:
    def __init__(self, answers: dict) -> None:
        self.answers = answers

    async def lookup(self, query: str):
        return self.answers.get(query, ())


class FakeAbbreviationService:
    def __init__(self, expansions: dict[str, str]) -> None:
        self.expansions = expansions

    def resolve(self, token: str):
        return self.expansions.get(token.strip().lower())


# ---------------------------------------------------------------------------
# Capability boundary
# ---------------------------------------------------------------------------


def test_capability_accepts_agent_request_and_capability_runtime_only() -> None:
    from app.services.agents.v2.contracts.execution import AgentRequest as Req

    evidence = FakeEvidenceBuilder()
    resolver = FakeTargetResolver()
    for capability in all_capabilities(evidence, resolver):
        assert isinstance(capability, Capability)
        parameters = inspect.signature(type(capability).execute).parameters
        assert set(parameters) == {"self", "request", "runtime"}
        assert inspect.iscoroutinefunction(capability.execute)
        hints = inspect.get_annotations(capability.execute, eval_str=True)
        assert hints["request"] is Req
        assert hints["runtime"] is CapabilityRuntimeContext


def test_capability_cannot_read_supervisor_root_state() -> None:
    import app.services.agents.v2.capabilities as package

    package_dir = Path(package.__file__).parent
    forbidden = (
        "GraphRuntimeContext",
        "SupervisorV2State",
        "supervisor",
        "checkpoint",
        "RuntimeServices",
    )
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        docstrings: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "state" not in (node.module or ""), path.name
                for alias in node.names:
                    assert alias.name not in forbidden, (path.name, alias.name)
            if isinstance(node, ast.Name) and node.id in forbidden:
                raise AssertionError(f"{path.name} reads {node.id}")
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                for name in forbidden:
                    assert name not in node.value, (path.name, name)


@pytest.mark.asyncio
async def test_model_input_cannot_supply_workspace_or_acl() -> None:
    assert "workspace_ids" not in AgentRequest.model_fields
    assert "can_read_people" not in AgentRequest.model_fields
    assert "allowed_capabilities" not in AgentRequest.model_fields
    assert "user_id" not in AgentRequest.model_fields

    with pytest.raises(ValidationError):
        AgentRequest.model_validate(
            {
                "contract_version": "2.0",
                "task_id": "T1",
                "objective": "x",
                "input": {"kind": "document.search", "query": "A"},
                "workspace_ids": [str(WORKSPACE_ID)],
            }
        )

    candidate = DocumentDiscoveryCandidate(
        candidate_id=uuid4(), document_id=DOCUMENT_ID, document_revision="rev-1"
    )
    capability_service = FakeSearchService((candidate,))
    from app.services.agents.v2.capabilities import DocumentSearchCapability

    capability = DocumentSearchCapability(service=capability_service)
    request = agent_request(
        "T1", "find A", DocumentSearchInput(kind="document.search", query="A")
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "success"
    # Workspace scope comes only from the trusted runtime, never the request.
    assert capability_service.calls == [("A", None, (WORKSPACE_ID,))]


@pytest.mark.asyncio
async def test_people_capability_enforces_current_people_permission() -> None:
    from app.services.agents.v2.capabilities import PeopleCapability

    service = FakePeopleService(
        {"nguyen van a": {"record_id": "p-1", "name": "Nguyen Van A"}}
    )
    evidence = FakeEvidenceBuilder()
    capability = PeopleCapability(
        service=service, evidence=evidence, required_fields=("name",)
    )
    assert capability.descriptor.domain == "people"
    request = agent_request(
        "T1", "who is A", PeopleLookupInput(kind="people.lookup", query="Nguyen Van A")
    )

    denied = await capability.execute(
        request, runtime_context(can_read_people=False)
    )
    assert denied.status == "denied"
    assert denied.error is not None and denied.error.code == "PERMISSION_DENIED"
    assert denied.data is None
    assert service.calls == []
    assert evidence.calls == []

    blocked = await capability.execute(
        request, runtime_context(allowed=frozenset({"document.read"}))
    )
    assert blocked.status == "denied"
    assert blocked.error is not None and blocked.error.code == "PERMISSION_DENIED"

    allowed = await capability.execute(request, runtime_context())
    assert allowed.status == "success"
    assert allowed.data is not None and allowed.data.matched is True
    assert len(allowed.evidence_uses) == 1
    # Raw People data stays governed in the Evidence Store, never in output.
    assert "Nguyen Van A" not in allowed.model_dump_json()


@pytest.mark.asyncio
async def test_document_capability_requires_pinned_authorized_revision() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability

    reader = FakeDocumentReader()
    evidence = FakeEvidenceBuilder()
    capability = DocumentReadCapability(
        reader=reader, evidence=evidence, resolver=FakeTargetResolver({})
    )
    request = agent_request(
        "T1", "read t9", DocumentReadInput(kind="document.read", target_ids=("t9",))
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "denied"
    assert result.error is not None and result.error.code == "SCOPE_VIOLATION"
    assert result.data is None
    assert reader.calls == []
    assert evidence.calls == []


@pytest.mark.asyncio
async def test_section_read_emits_read_coverage_not_search_coverage() -> None:
    from app.services.agents.v2.capabilities import SectionReadCapability

    evidence = FakeEvidenceBuilder()
    capability = SectionReadCapability(
        reader=FakeSectionReader(),
        evidence=evidence,
        resolver=FakeTargetResolver(),
    )
    request = agent_request(
        "T1", "read section", SectionReadInput(kind="section.read", target_ids=("t1",))
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "success"
    assert result.data is not None and result.data.kind == "section.read"
    assert result.data.read_unit_count == 1
    assert len(result.coverage_observations) == 1
    observation = result.coverage_observations[0]
    assert observation.target_id == "t1"
    assert observation.outcome == "read"
    assert observation.observed_locators == (
        SectionLocator(kind="section", structure_node_id="node-5"),
    )


def test_registry_intersects_runtime_permission_and_service_availability() -> None:
    evidence = FakeEvidenceBuilder()
    resolver = FakeTargetResolver()
    capabilities = all_capabilities(evidence, resolver)
    by_name = {capability.descriptor.name: capability for capability in capabilities}

    registry = build_capability_registry(
        [
            CapabilityRegistration(capability=by_name["people.lookup"]),
            CapabilityRegistration(
                capability=by_name["document.search"], feature_flag="search-beta"
            ),
            CapabilityRegistration(
                capability=by_name["document.read"], service="doc-store"
            ),
            CapabilityRegistration(capability=by_name["section.read"]),
        ],
        runtime_context(
            allowed=frozenset({"people.lookup", "document.search", "document.read"})
        ),
        active_feature_flags=frozenset(),
        available_services=frozenset({"doc-store"}),
    )
    # people.lookup: permitted; document.search: flag off; document.read:
    # permitted + service up; section.read: outside allowed_capabilities.
    assert [d.name for d in registry.catalog()] == ["document.read", "people.lookup"]
    assert registry.get("document.read").descriptor.name == "document.read"
    with pytest.raises(CapabilityUnavailable):
        registry.get("document.search")
    with pytest.raises(CapabilityDenied):
        registry.get("section.read")
    with pytest.raises(CapabilityNotRegistered):
        registry.get("write")


def test_v2_has_no_domain_agent_or_domain_graph_wrappers() -> None:
    import app.services.agents.v2 as v2
    import app.services.agents.v2.capabilities as capabilities_package

    root = Path(v2.__file__).parent
    # No domain-agent modules and no LangGraph graph/subgraph construction
    # anywhere under v2: only nodes and atomic capabilities may exist.
    # Phase-3 exception (normative amendment §6): `complex_research_graph.py`
    # is the ONE authorized adaptive planning boundary, implemented as a
    # checkpointed LangGraph subgraph inheriting the supervisor saver. It
    # must still contain no domain agent (no *_agent.py module, no *Agent
    # class, no second scheduler/checkpointer).
    COMPLEX_SUBGRAPH = "complex_research_graph.py"
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        assert not path.name.endswith("_agent.py"), path.name
        text = path.read_text()
        if path.name == COMPLEX_SUBGRAPH:
            assert not re.search(r"class\s+\w*Agent\b", text), path.name
            for banned in (
                "people_agent",
                "comparison_agent",
                "summary_agent",
                "document_agent",
                "section_agent",
                "kg_agent",
                "evaluation_agent",
                "grounding_agent",
            ):
                assert banned not in text, (path.name, banned)
            assert "class TaskScheduler" not in text, path.name
            assert "checkpointer=" not in text, path.name
            continue
        assert "StateGraph" not in text, path.name
        assert "Subgraph" not in text, path.name
    # Capability modules define atomic capabilities only: no *Agent classes
    # and no langgraph import (nodes own the framework runtime injection).
    capabilities_dir = Path(capabilities_package.__file__).parent
    for path in sorted(capabilities_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert not node.name.endswith("Agent"), f"{path.name}:{node.name}"
        assert "langgraph" not in path.read_text(), path.name


# ---------------------------------------------------------------------------
# Fix round 1: locator fidelity, dedupe, positive paths, fail-closed scalar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_read_positive_path_pins_revision_and_coverage() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability

    evidence = FakeEvidenceBuilder()
    resolver = FakeTargetResolver(
        {"t1": resolved_target("t1", DocumentLocator(kind="document"))}
    )
    capability = DocumentReadCapability(
        reader=FakeDocumentReader(default_content="full text"),
        evidence=evidence,
        resolver=resolver,
    )
    request = agent_request(
        "T1", "read t1", DocumentReadInput(kind="document.read", target_ids=("t1",))
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "success"
    assert result.data is not None and result.data.read_unit_count == 1
    assert len(result.evidence_uses) == 1
    assert len(result.coverage_observations) == 1
    call = evidence.calls[0]
    assert call["source"].document_revision == "rev-1"
    assert call["source"].document_id == DOCUMENT_ID
    assert call["purpose"] == "coverage"
    assert call["target_id"] == "t1"
    observation = result.coverage_observations[0]
    assert observation.outcome == "read"
    assert observation.observed_locators == (DocumentLocator(kind="document"),)


@pytest.mark.asyncio
async def test_document_read_partial_and_missing_branches() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability

    doc = DocumentLocator(kind="document")
    evidence = FakeEvidenceBuilder()
    resolver = FakeTargetResolver(
        {"t1": resolved_target("t1", doc), "t2": resolved_target("t2", doc)}
    )
    partial_resolver = FakeTargetResolver(
        {
            "t1": resolved_target("t1", doc),
            "t2": resolved_target("t2", doc, revision="rev-gone"),
        }
    )

    class SelectiveReader:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def read(self, binding, locator):
            self.calls.append((binding, locator))
            if binding.document_revision == "rev-gone":
                return LocatedContent(
                    outcome="missing", observed_locator=None, content=None
                )
            return LocatedContent(
                outcome="read", observed_locator=locator, content="full text"
            )

    selective = SelectiveReader()
    capability = DocumentReadCapability(
        reader=selective, evidence=evidence, resolver=partial_resolver
    )
    request = agent_request(
        "T1",
        "read both",
        DocumentReadInput(kind="document.read", target_ids=("t1", "t2")),
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "partial"
    assert result.data is not None and result.data.read_unit_count == 1
    assert len(result.evidence_uses) == 1
    assert [o.outcome for o in result.coverage_observations] == ["read", "missing"]

    missing_only = DocumentReadCapability(
        reader=FakeDocumentReader(default_content=None),
        evidence=FakeEvidenceBuilder(),
        resolver=resolver,
    )
    missing_result = await missing_only.execute(
        agent_request(
            "T2",
            "read both",
            DocumentReadInput(kind="document.read", target_ids=("t1", "t2")),
        ),
        runtime_context(),
    )
    assert missing_result.status == "not_found"
    assert missing_result.data is not None
    assert missing_result.data.read_unit_count == 0
    assert missing_result.evidence_uses == ()


@pytest.mark.asyncio
async def test_document_read_duplicate_target_ids_mint_distinct_uses() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability
    from app.services.agents.v2.contracts.validation import validate_agent_result

    evidence = FakeEvidenceBuilder()
    resolver = FakeTargetResolver(
        {"t1": resolved_target("t1", DocumentLocator(kind="document"))}
    )
    capability = DocumentReadCapability(
        reader=FakeDocumentReader(default_content="full text"),
        evidence=evidence,
        resolver=resolver,
    )
    request = agent_request(
        "T1",
        "read twice",
        DocumentReadInput(kind="document.read", target_ids=("t1", "t1")),
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "success"
    assert result.data is not None and result.data.read_unit_count == 1
    assert len(result.evidence_uses) == 1
    assert len({u.use_id for u in result.evidence_uses}) == 1
    assert len(result.coverage_observations) == 1
    assert len(evidence.calls) == 1
    validate_agent_result(
        result,
        TaskPlan(
            contract_version="2.0",
            plan_id="p1",
            goal="g",
            target_units=(
                TargetUnit(
                    target_id="t1",
                    binding_id="b_t1",
                    requested_locator=DocumentLocator(kind="document"),
                    completion_criteria=(),
                ),
            ),
            tasks=(
                TaskSpec(
                    task_id="T1",
                    capability="document.read",
                    task_objective="read twice",
                    input=request.input,
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_document_read_locator_mismatch_fails_closed() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability

    requested = PageRangeLocator(kind="page_range", start=1, end=2)
    evidence = FakeEvidenceBuilder()
    resolver = FakeTargetResolver({"t1": resolved_target("t1", requested)})
    # The dependency claims the whole document although pages 1-2 were asked.
    reader = FakeDocumentReader(
        by_locator={
            requested.model_dump_json(): LocatedContent(
                outcome="read",
                observed_locator=DocumentLocator(kind="document"),
                content="whole doc?",
            )
        }
    )
    capability = DocumentReadCapability(
        reader=reader, evidence=evidence, resolver=resolver
    )
    result = await capability.execute(
        agent_request(
            "T1", "read pages", DocumentReadInput(kind="document.read", target_ids=("t1",))
        ),
        runtime_context(),
    )
    assert result.status == "not_found"
    assert result.data is not None and result.data.read_unit_count == 0
    assert result.evidence_uses == ()
    assert result.coverage_observations[0].outcome == "missing"

    # Mirror case: a whole-document target met with a partial read.
    whole = DocumentLocator(kind="document")
    partial_reader = FakeDocumentReader(
        by_locator={
            whole.model_dump_json(): LocatedContent(
                outcome="read",
                observed_locator=PageRangeLocator(kind="page_range", start=1, end=2),
                content="pages 1-2 only",
            )
        }
    )
    whole_capability = DocumentReadCapability(
        reader=partial_reader,
        evidence=FakeEvidenceBuilder(),
        resolver=FakeTargetResolver({"t1": resolved_target("t1", whole)}),
    )
    whole_result = await whole_capability.execute(
        agent_request(
            "T2", "read doc", DocumentReadInput(kind="document.read", target_ids=("t1",))
        ),
        runtime_context(),
    )
    assert whole_result.data is not None and whole_result.data.read_unit_count == 0
    assert whole_result.evidence_uses == ()
    assert whole_result.coverage_observations[0].outcome == "missing"


@pytest.mark.asyncio
async def test_section_read_wrong_section_fails_closed() -> None:
    from app.services.agents.v2.capabilities import SectionReadCapability

    requested = SectionLocator(kind="section", structure_node_id="node-5")
    evidence = FakeEvidenceBuilder()
    reader = FakeSectionReader(
        by_locator={
            requested.model_dump_json(): LocatedContent(
                outcome="read",
                observed_locator=SectionLocator(
                    kind="section", structure_node_id="node-9"
                ),
                content="another section",
            )
        }
    )
    capability = SectionReadCapability(
        reader=reader, evidence=evidence, resolver=FakeTargetResolver()
    )
    result = await capability.execute(
        agent_request(
            "T1",
            "read section",
            SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        runtime_context(),
    )
    assert result.status == "not_found"
    assert result.data is not None and result.data.read_unit_count == 0
    assert result.evidence_uses == ()
    assert evidence.calls == []
    assert result.coverage_observations[0].outcome == "missing"
    assert result.coverage_observations[0].observed_locators == ()


@pytest.mark.asyncio
async def test_section_read_unreadable_and_truncated_branches() -> None:
    from app.services.agents.v2.capabilities import SectionReadCapability

    requested = SectionLocator(kind="section", structure_node_id="node-5")
    reader = FakeSectionReader(
        by_locator={
            requested.model_dump_json(): LocatedContent(
                outcome="unreadable",
                observed_locator=requested,
                content=None,
            )
        }
    )
    capability = SectionReadCapability(
        reader=reader,
        evidence=FakeEvidenceBuilder(),
        resolver=FakeTargetResolver(),
    )
    unreadable = await capability.execute(
        agent_request(
            "T1",
            "read section",
            SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        runtime_context(),
    )
    assert unreadable.status == "not_found"
    assert unreadable.data is not None and unreadable.data.read_unit_count == 0
    assert unreadable.evidence_uses == ()
    assert unreadable.coverage_observations[0].outcome == "unreadable"
    assert unreadable.coverage_observations[0].observed_locators == (requested,)

    truncated_reader = FakeSectionReader(
        by_locator={
            requested.model_dump_json(): LocatedContent(
                outcome="truncated",
                observed_locator=requested,
                content="partial…",
            )
        }
    )
    truncated_capability = SectionReadCapability(
        reader=truncated_reader,
        evidence=FakeEvidenceBuilder(),
        resolver=FakeTargetResolver(),
    )
    truncated = await truncated_capability.execute(
        agent_request(
            "T2",
            "read section",
            SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        runtime_context(),
    )
    assert truncated.data is not None and truncated.data.read_unit_count == 0
    assert truncated.coverage_observations[0].outcome == "truncated"


@pytest.mark.asyncio
async def test_document_search_person_identifier_fails_closed() -> None:
    from app.services.agents.v2.capabilities import DocumentSearchCapability

    service = FakeSearchService(())
    capability = DocumentSearchCapability(service=service)
    result = await capability.execute(
        agent_request(
            "T1",
            "find docs for A",
            DocumentSearchInput(
                kind="document.search", query="A", person_identifier="p-1"
            ),
        ),
        runtime_context(),
    )
    # The scalar is threaded into the port but unsupported: fail closed,
    # never silently run a plain query search.
    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "DEPENDENCY_UNAVAILABLE"
    assert result.data is None
    assert service.calls == []


@pytest.mark.asyncio
async def test_knowledge_graph_memory_abbreviation_positive_paths() -> None:
    from app.services.agents.v2.capabilities import (
        AbbreviationCapability,
        KnowledgeGraphCapability,
        MemoryCapability,
    )

    evidence = FakeEvidenceBuilder()
    kg = KnowledgeGraphCapability(
        client=FakeKgClient({"ai": (("e-1", "AI content"),)}), evidence=evidence
    )
    kg_result = await kg.execute(
        agent_request(
            "T1", "kg lookup", KnowledgeGraphInput(kind="knowledge_graph.query", query="ai")
        ),
        runtime_context(),
    )
    assert kg_result.status == "success"
    assert kg_result.data is not None and kg_result.data.matched_entity_count == 1
    assert len(kg_result.evidence_uses) == 1

    memory = MemoryCapability(
        store=FakeMemoryStore({"hello": (("m-1", "remembered"),)}), evidence=evidence
    )
    memory_result = await memory.execute(
        agent_request(
            "T2", "memory", MemoryLookupInput(kind="memory.lookup", query="hello")
        ),
        runtime_context(),
    )
    assert memory_result.status == "success"
    assert memory_result.data is not None and memory_result.data.matched_count == 1
    assert len(memory_result.evidence_uses) == 1

    abbreviations = AbbreviationCapability(
        service=FakeAbbreviationService({"nđ": "Nghị định"})
    )
    abbr_result = await abbreviations.execute(
        agent_request(
            "T3",
            "expand",
            AbbreviationResolveInput(kind="abbreviation.resolve", tokens=("NĐ", "xyz")),
        ),
        runtime_context(),
    )
    assert abbr_result.status == "success"
    assert abbr_result.data is not None
    assert abbr_result.data.resolutions[0].expansion == "Nghị định"
    assert abbr_result.data.resolutions[1].expansion is None
    assert abbr_result.evidence_uses == ()


@pytest.mark.asyncio
async def test_knowledge_graph_duplicate_matches_count_distinct() -> None:
    from app.services.agents.v2.capabilities import KnowledgeGraphCapability
    from app.services.agents.v2.contracts.validation import validate_agent_result

    evidence = FakeEvidenceBuilder()
    capability = KnowledgeGraphCapability(
        client=FakeKgClient({"ai": (("e-1", "AI"), ("e-1", "AI"))}),
        evidence=evidence,
    )
    request = agent_request(
        "T1", "kg lookup", KnowledgeGraphInput(kind="knowledge_graph.query", query="ai")
    )
    result = await capability.execute(request, runtime_context())
    assert result.status == "success"
    assert result.data is not None and result.data.matched_entity_count == 1
    assert len(result.evidence_uses) == 1
    assert len(evidence.calls) == 1
    validate_agent_result(
        result,
        TaskPlan(
            contract_version="2.0",
            plan_id="p1",
            goal="g",
            target_units=(),
            tasks=(
                TaskSpec(
                    task_id="T1",
                    capability="knowledge_graph.query",
                    task_objective="kg lookup",
                    input=request.input,
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_people_record_without_id_fails_closed() -> None:
    from app.services.agents.v2.capabilities import PeopleCapability

    service = FakePeopleService({"noid": {"name": "No Id"}})
    evidence = FakeEvidenceBuilder()
    capability = PeopleCapability(
        service=service, evidence=evidence, required_fields=("name",)
    )
    result = await capability.execute(
        agent_request(
            "T1", "who", PeopleLookupInput(kind="people.lookup", query="noid")
        ),
        runtime_context(),
    )
    assert result.status == "error"
    assert result.error is not None and result.error.code == "CONTRACT_MISMATCH"
    assert result.data is None
    assert evidence.calls == []


@pytest.mark.asyncio
async def test_dependency_failure_maps_to_typed_error() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability

    class ExplodingReader:
        async def read(self, binding, locator):
            raise ConnectionError("store down")

    class SlowReader:
        async def read(self, binding, locator):
            raise TimeoutError("timed out")

    resolver = FakeTargetResolver(
        {"t1": resolved_target("t1", DocumentLocator(kind="document"))}
    )
    failing = DocumentReadCapability(
        reader=ExplodingReader(),
        evidence=FakeEvidenceBuilder(),
        resolver=resolver,
    )
    failed = await failing.execute(
        agent_request(
            "T1", "read", DocumentReadInput(kind="document.read", target_ids=("t1",))
        ),
        runtime_context(),
    )
    assert failed.status == "error"
    assert failed.error is not None
    assert failed.error.code == "DEPENDENCY_UNAVAILABLE"
    assert failed.error.retryable is True
    assert failed.data is None

    timing_out = DocumentReadCapability(
        reader=SlowReader(),
        evidence=FakeEvidenceBuilder(),
        resolver=resolver,
    )
    timed_out = await timing_out.execute(
        agent_request(
            "T2", "read", DocumentReadInput(kind="document.read", target_ids=("t1",))
        ),
        runtime_context(),
    )
    assert timed_out.status == "error"
    assert timed_out.error is not None and timed_out.error.code == "TIMEOUT"
    assert timed_out.error.retryable is True


def test_people_descriptor_domain_and_registry_exclusion() -> None:
    from app.services.agents.v2.capabilities import PeopleCapability

    service = FakePeopleService({"a": {"record_id": "p-1", "name": "A"}})
    capability = PeopleCapability(
        service=service, evidence=FakeEvidenceBuilder(), required_fields=("name",)
    )
    assert capability.descriptor.domain == "people"

    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)],
        runtime_context(
            allowed=frozenset({"people.lookup"}), can_read_people=False
        ),
    )
    assert registry.catalog() == ()
    with pytest.raises(CapabilityDenied):
        registry.get("people.lookup")


@pytest.mark.asyncio
async def test_acquisition_id_shared_per_execute() -> None:
    from app.services.agents.v2.capabilities import DocumentReadCapability

    doc = DocumentLocator(kind="document")
    evidence = FakeEvidenceBuilder()
    capability = DocumentReadCapability(
        reader=FakeDocumentReader(default_content="full text"),
        evidence=evidence,
        resolver=FakeTargetResolver(
            {"t1": resolved_target("t1", doc), "t2": resolved_target("t2", doc)}
        ),
    )
    result = await capability.execute(
        agent_request(
            "T1",
            "read both",
            DocumentReadInput(kind="document.read", target_ids=("t1", "t2")),
        ),
        runtime_context(),
    )
    assert result.status == "success"
    assert len(evidence.calls) == 2
    assert (
        evidence.calls[0]["provenance"].acquisition_id
        == evidence.calls[1]["provenance"].acquisition_id
    )


@pytest.mark.asyncio
async def test_evidence_builder_is_idempotent() -> None:
    from app.services.agents.v2.contracts.evidence import (
        KnowledgeGraphSourceIdentity,
        Provenance,
    )

    evidence = FakeEvidenceBuilder(run_id="run-1")
    provenance = Provenance(
        acquisition_id=uuid4(), fetcher="knowledge_graph.query", fetched_at=datetime.now(UTC)
    )
    first = await evidence.persist_use(
        source=KnowledgeGraphSourceIdentity(
            kind="knowledge_graph", entity_or_relation_id="e-1"
        ),
        content="AI",
        provenance=provenance,
        task_id="T1",
        purpose="supporting",
        target_id=None,
    )
    second = await evidence.persist_use(
        source=KnowledgeGraphSourceIdentity(
            kind="knowledge_graph", entity_or_relation_id="e-1"
        ),
        content="AI",
        provenance=provenance,
        task_id="T1",
        purpose="supporting",
        target_id=None,
    )
    assert first.use_id == second.use_id
