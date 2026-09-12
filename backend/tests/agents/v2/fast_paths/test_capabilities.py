"""Phase 2 Task 2 — shared atomic capabilities and the request-scoped registry.

Boundary tests: every capability receives ONLY ``AgentRequest`` +
``CapabilityRuntimeContext``, never supervisor/graph state; People enforces the
current people permission with governed minimization; reads resolve pinned
authorized revisions through the constructor-injected resolver; section reads
emit READ coverage (never search coverage); the registry intersects runtime
permissions with feature flags/service availability; and v2 ships no domain
agent or domain graph wrappers.
"""
from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    Capability,
    CapabilityDenied,
    CapabilityNotRegistered,
    CapabilityRegistration,
    CapabilityUnavailable,
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
from app.services.agents.v2.contracts.locators import SectionLocator

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


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
    """In-memory evidence builder: mints use refs without a database."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    async def persist_use(
        self, *, source, content, provenance, task_id, purpose, target_id
    ) -> EvidenceUseRef:
        self.calls.append(
            (source, content, provenance, task_id, purpose, target_id)
        )
        return EvidenceUseRef(use_id=uuid4())


def pinned_binding() -> ScopedDocument:
    return ScopedDocument(
        binding_id="b_t1",
        document_id=DOCUMENT_ID,
        document_revision="rev-1",
        role="target",
    )


class FakeTargetResolver:
    """Request-scoped resolver fed the authoritative plan/bindings by T6/T7."""

    def __init__(self, bindings: dict[str, ScopedDocument] | None = None) -> None:
        self._bindings = dict(bindings or {"t1": pinned_binding()})

    def resolve(self, target_id: str) -> ScopedDocument | None:
        return self._bindings.get(target_id)


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
            service=FakePeopleService({"nguyen van a": {"name": "Nguyen Van A"}}),
            evidence=evidence,
            required_fields=("name",),
        ),
        DocumentSearchCapability(service=FakeSearchService(())),
        DocumentReadCapability(
            reader=FakeDocumentReader({("t1-marker",): "content"}),
            evidence=evidence,
            resolver=resolver,
        ),
        SectionReadCapability(
            reader=FakeSectionReader({"t1": ("content", "node-5")}),
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

    async def search(self, query: str, workspace_ids):
        self.calls.append((query, workspace_ids))
        return self._candidates


class FakeDocumentReader:
    def __init__(self, contents: dict) -> None:
        self.contents = contents
        self.calls: list[ScopedDocument] = []

    async def read(self, binding: ScopedDocument):
        self.calls.append(binding)
        return "doc-content"


class FakeSectionReader:
    def __init__(self, sections: dict[str, tuple[str, str]]) -> None:
        self.sections = sections
        self.calls: list[tuple] = []

    async def read_section(self, binding: ScopedDocument, target_id: str):
        self.calls.append((binding, target_id))
        return self.sections.get(target_id)


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


def test_model_input_cannot_supply_workspace_or_acl() -> None:
    assert "workspace_ids" not in AgentRequest.model_fields
    assert "can_read_people" not in AgentRequest.model_fields
    assert "allowed_capabilities" not in AgentRequest.model_fields
    assert "user_id" not in AgentRequest.model_fields

    with pytest.raises(Exception):
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
    import asyncio

    result = asyncio.run(capability.execute(request, runtime_context()))
    assert result.status == "success"
    # Workspace scope comes only from the trusted runtime, never the request.
    assert capability_service.calls == [(("A"), (WORKSPACE_ID,))]


@pytest.mark.asyncio
async def test_people_capability_enforces_current_people_permission() -> None:
    from app.services.agents.v2.capabilities import PeopleCapability

    service = FakePeopleService({"nguyen van a": {"name": "Nguyen Van A"}})
    evidence = FakeEvidenceBuilder()
    capability = PeopleCapability(
        service=service, evidence=evidence, required_fields=("name",)
    )
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

    reader = FakeDocumentReader({})
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
        reader=FakeSectionReader({"t1": ("section text", "node-5")}),
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
    for forbidden in ("missing", "unreadable", "truncated"):
        assert observation.outcome != forbidden


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
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        assert not path.name.endswith("_agent.py"), path.name
        text = path.read_text()
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
