"""Minimal valid v2 contract instances shared by the Task-6 contract tests.

The builders exist so validation tests can state one deviation per case instead
of re-declaring a whole plan/binding graph. They are test fixtures only: the
canonical shapes live in ``app.services.agents.v2.contracts``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    DocumentReadInput,
    KnowledgeGraphInput,
    PeopleLookupInput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evidence import (
    DocumentSourceIdentity,
    EvidenceRecord,
    Provenance,
)
from app.services.agents.v2.contracts.locators import ContentLocator, DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.request import KnownDocumentResource, RequestContext
from app.services.agents.v2.contracts.semantic import DocumentReference, SemanticContext

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
EVIDENCE_ID = UUID("33333333-3333-3333-3333-333333333333")
USE_ID = UUID("44444444-4444-4444-4444-444444444444")
REVISION = "rev-1"
OTHER_REVISION = "rev-2"


def request_context() -> RequestContext:
    return RequestContext(
        contract_version="2.0",
        request_id="req-1",
        thread_id="thread-1",
        original_query="Điều 5 của A nói gì?",
        known_documents=(
            KnownDocumentResource(
                resource_id="res-1",
                document_id=DOCUMENT_ID,
                source="attachment",
            ),
        ),
    )


def conversation_context() -> ConversationContext:
    return ConversationContext(
        summary="Hỏi về nghị định A.",
        active_entities=(),
        last_focus=None,
        recent_turns=(),
    )


def document_reference(
    *,
    ref_id: str = "r1",
    revision_requirement: object = None,
    resolution_status: str = "resolved",
    resolved_document_id: UUID | None = DOCUMENT_ID,
    candidate_document_ids: tuple[UUID, ...] = (),
) -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="A",
        normalized_reference="A",
        requested_role="target",
        revision_requirement=revision_requirement,
        resolution_status=resolution_status,  # type: ignore[arg-type]
        resolved_document_id=resolved_document_id,
        candidate_document_ids=candidate_document_ids,
    )


def semantic_context(*, document_refs: tuple[DocumentReference, ...] | None = None) -> SemanticContext:
    if document_refs is None:
        document_refs = (document_reference(),)
    return SemanticContext(
        contextualized_query="Điều 5 của A nói gì?",
        normalized_query="Điều 5 của A nói gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=document_refs,
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def scoped_document(
    *,
    binding_id: str = "b1",
    document_id: UUID = DOCUMENT_ID,
    document_revision: str = REVISION,
    role: str = "target",
) -> ScopedDocument:
    return ScopedDocument(
        binding_id=binding_id,
        document_id=document_id,
        document_revision=document_revision,
        role=role,  # type: ignore[arg-type]
    )


def binding_set(
    *,
    bindings: tuple[ScopedDocument, ...] | None = None,
    revision_requirement_refs: tuple[object, ...] = (),
) -> DocumentBindingSet:
    if bindings is None:
        bindings = (scoped_document(),)
    return DocumentBindingSet(
        bindings=bindings,
        revision_requirement_refs=revision_requirement_refs,  # type: ignore[arg-type]
    )


def target_unit(
    *,
    target_id: str = "t1",
    binding_id: str = "b1",
    requested_locator: ContentLocator | None = None,
    completion_criteria: tuple[object, ...] | None = None,
) -> TargetUnit:
    if requested_locator is None:
        requested_locator = DocumentLocator(kind="document")
    if completion_criteria is None:
        completion_criteria = (CoverageCriterion(kind="coverage"),)
    return TargetUnit(
        target_id=target_id,
        binding_id=binding_id,
        requested_locator=requested_locator,
        completion_criteria=completion_criteria,  # type: ignore[arg-type]
    )


def read_task(task_id: str = "T1", target_ids: tuple[str, ...] = ("t1",)) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective="Đọc Điều 5 của A",
        input=DocumentReadInput(kind="document.read", target_ids=target_ids),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def read_plan(*, target_units: tuple[TargetUnit, ...] | None = None, tasks: tuple[TaskSpec, ...] | None = None) -> TaskPlan:
    if target_units is None:
        target_units = (target_unit(),)
    if tasks is None:
        tasks = (read_task(),)
    return TaskPlan(
        contract_version="2.0",
        plan_id="p1",
        goal="Đọc Điều 5 của A",
        target_units=target_units,
        tasks=tasks,
    )


def people_plan(*, task_id: str = "T1") -> TaskPlan:
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


def kg_task(task_id: str = "T2", *, depends_on: tuple[str, ...] = ("T1",)) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="knowledge_graph.query",
        task_objective="A thuộc đơn vị nào?",
        input=KnowledgeGraphInput(kind="knowledge_graph.query", query="A"),
        depends_on=depends_on,
        origin=InitialTaskOrigin(kind="initial"),
    )


def evidence_record(
    *,
    evidence_id: UUID = EVIDENCE_ID,
    document_id: UUID = DOCUMENT_ID,
    document_revision: str = REVISION,
    locator: ContentLocator | None = None,
) -> EvidenceRecord:
    if locator is None:
        locator = DocumentLocator(kind="document")
    return EvidenceRecord(
        evidence_id=evidence_id,
        source=DocumentSourceIdentity(
            kind="document",
            document_id=document_id,
            document_revision=document_revision,
            locator=locator,
        ),
        content="Điều 5 quy định ...",
        content_hash="sha256:abc",
        provenance=Provenance(
            acquisition_id=UUID("55555555-5555-5555-5555-555555555555"),
            fetcher="document.read",
            fetched_at=datetime(2026, 9, 11, tzinfo=UTC),
        ),
    )
