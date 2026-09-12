"""Spec §9/§10/§12/§20 — locators, routing, binding audit, and clarification.

Covers "ID/relation integrity" for the binding/discovery audit path plus the
§26 facts: typed locators without a dict escape hatch, no duplicated routing
policy fields, and clarification resume validates candidate membership.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, get_args, get_origin
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from app.services.agents.v2.contracts.binding import (
    BindingAdditionRequest,
    BindingAuditRow,
    BindingPromotionRequest,
    BindingProvenance,
    DiscoveredBindingProvenance,
    PromotedBindingProvenance,
    UserBindingProvenance,
)
from app.services.agents.v2.contracts.clarification import (
    ClarificationRequest,
    ClarificationResolution,
    DocumentCandidate,
)
from app.services.agents.v2.contracts.locators import (
    ArticleLocator,
    ChunkRangeLocator,
    ContentLocator,
    DocumentLocator,
    PageRangeLocator,
    SectionLocator,
)
from app.services.agents.v2.contracts.routing import (
    QueryAnalysis,
    RouteDecision,
    SemanticDependencyHint,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_clarification_request,
    validate_clarification_resolution,
    validate_query_analysis,
)

from .factories import DOCUMENT_ID, document_reference, semantic_context

EXPECTED_LOCATOR_KINDS = ["document", "section", "article", "page_range", "chunk_range"]


def test_content_locator_is_a_closed_discriminated_union() -> None:
    assert get_origin(ContentLocator) is Annotated
    union, field = get_args(ContentLocator)
    assert field.discriminator == "kind"
    variants = get_args(union)
    kinds = [get_args(variant.model_fields["kind"].annotation)[0] for variant in variants]
    assert kinds == EXPECTED_LOCATOR_KINDS
    assert SectionLocator.model_fields["structure_node_id"].annotation is str
    assert ArticleLocator.model_fields["article_id"].annotation is str
    assert PageRangeLocator.model_fields["start"].annotation is int
    assert ChunkRangeLocator.model_fields["end"].annotation is str


def test_locator_union_uses_stable_structure_ids_and_no_dict() -> None:
    adapter = TypeAdapter(ContentLocator)
    parsed = adapter.validate_python({"kind": "article", "structure_node_id": "n1", "article_id": "a5"})
    assert isinstance(parsed, ArticleLocator)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "document", "metadata": {"page": 3}})


def test_query_analysis_carries_no_policy_or_telemetry_fields() -> None:
    analysis = QueryAnalysis(
        work_type="cross_domain",
        domains=("people", "document"),
        dependency_hints=(SemanticDependencyHint(hint_id="h1", description="People → Document"),),
    )
    assert set(QueryAnalysis.model_fields) == {"work_type", "domains", "dependency_hints"}
    for forbidden in ("capabilities", "complexity", "synthesis", "route", "budget"):
        assert forbidden not in QueryAnalysis.model_fields
    assert set(SemanticDependencyHint.model_fields) == {"hint_id", "description"}
    validate_query_analysis(analysis)
    assert analysis.domains == ("people", "document")


def test_query_analysis_rejects_duplicate_or_empty_domains() -> None:
    with pytest.raises(ContractValidationError, match="domain"):
        validate_query_analysis(QueryAnalysis(work_type="lookup", domains=()))
    with pytest.raises(ContractValidationError, match="duplicate"):
        validate_query_analysis(QueryAnalysis(work_type="lookup", domains=("people", "people")))


def test_route_decision_does_not_duplicate_domain_or_capability() -> None:
    decision = RouteDecision(route="fast_domain", reason_code="simple_people_lookup")
    assert set(RouteDecision.model_fields) == {"route", "reason_code"}
    for forbidden in ("domain", "capability", "complexity"):
        assert forbidden not in RouteDecision.model_fields
    assert decision.route == "fast_domain"


def test_binding_provenance_is_a_discriminated_audit_union() -> None:
    assert get_origin(BindingProvenance) is Annotated
    union, field = get_args(BindingProvenance)
    assert field.discriminator == "kind"
    variants = get_args(union)
    assert variants == (
        UserBindingProvenance,
        DiscoveredBindingProvenance,
        PromotedBindingProvenance,
    )
    assert set(BindingAuditRow.model_fields) == {"contract_version", "provenance"}
    row = BindingAuditRow(
        contract_version="2.0",
        provenance=DiscoveredBindingProvenance(
            kind="discovered", binding_id="b2", source_task_id="T2"
        ),
    )
    assert row.provenance.source_task_id == "T2"
    # Provenance keeps a reconstructable minimal path; no evidence-ID duplication.
    for variant in variants:
        assert "evidence_ids" not in variant.model_fields


def test_promotion_and_addition_requests_reference_existing_identity() -> None:
    addition = BindingAdditionRequest(candidate_id=uuid4(), requested_role="discovered")
    promotion = BindingPromotionRequest(source_binding_id="b2", requested_role="reference")
    assert set(BindingAdditionRequest.model_fields) == {"candidate_id", "requested_role"}
    assert set(BindingPromotionRequest.model_fields) == {"source_binding_id", "requested_role"}
    assert addition.requested_role == "discovered"
    assert promotion.requested_role == "reference"
    with pytest.raises(ValidationError):
        BindingAdditionRequest(candidate_id=uuid4(), requested_role="target")


def test_clarification_ref_ids_resolve_in_the_semantic_context() -> None:
    candidate = DocumentCandidate(
        candidate_id="cand-1",
        ordinal=0,
        ref_id="r1",
        document_id=DOCUMENT_ID,
        label="Nghị định A",
    )
    request = ClarificationRequest(
        contract_version="2.0",
        clarification_id="cl-1",
        reason="required_document_ambiguous",
        question="Bạn muốn nói tới tài liệu nào?",
        unresolved_ref_ids=("r1",),
        candidates=(candidate,),
        expires_at=datetime(2026, 9, 12, tzinfo=UTC),
    )
    semantic = semantic_context(document_refs=(document_reference(),))
    validate_clarification_request(request, semantic)

    dangling = request.model_copy(update={"unresolved_ref_ids": ("rX",)})
    with pytest.raises(ContractValidationError, match="rX"):
        validate_clarification_request(dangling, semantic)
    unknown_candidate = request.model_copy(
        update={"candidates": (candidate.model_copy(update={"ref_id": "rX"}),)}
    )
    with pytest.raises(ContractValidationError, match="rX"):
        validate_clarification_request(unknown_candidate, semantic)
    duplicate_candidate = request.model_copy(update={"candidates": (candidate, candidate)})
    with pytest.raises(ContractValidationError, match="duplicate"):
        validate_clarification_request(duplicate_candidate, semantic)
    reused_ordinal = request.model_copy(
        update={
            "candidates": (
                candidate,
                DocumentCandidate(
                    candidate_id="cand-2",
                    ordinal=0,
                    ref_id="r1",
                    document_id=DOCUMENT_ID,
                    label="Nghị định B",
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError, match="ordinal"):
        validate_clarification_request(reused_ordinal, semantic)
    nothing_found = request.model_copy(
        update={
            "reason": "required_document_not_found",
            "unresolved_ref_ids": (),
            "candidates": (),
        }
    )
    validate_clarification_request(nothing_found, semantic)


def test_clarification_resolution_selects_a_known_candidate() -> None:
    candidate = DocumentCandidate(
        candidate_id="cand-1",
        ordinal=0,
        ref_id="r1",
        document_id=DOCUMENT_ID,
        label="Nghị định A",
    )
    request = ClarificationRequest(
        contract_version="2.0",
        clarification_id="cl-1",
        reason="required_document_ambiguous",
        question="?",
        unresolved_ref_ids=("r1",),
        candidates=(candidate,),
        expires_at=datetime(2026, 9, 12, tzinfo=UTC),
    )
    validate_clarification_resolution(
        request, ClarificationResolution(contract_version="2.0", clarification_id="cl-1", selected_candidate_id=None)
    )
    validate_clarification_resolution(
        request,
        ClarificationResolution(
            contract_version="2.0", clarification_id="cl-1", selected_candidate_id="cand-1"
        ),
    )
    with pytest.raises(ContractValidationError, match="candidate"):
        validate_clarification_resolution(
            request,
            ClarificationResolution(
                contract_version="2.0", clarification_id="cl-1", selected_candidate_id="cand-9"
            ),
        )
    with pytest.raises(ContractValidationError, match="clarification_id"):
        validate_clarification_resolution(
            request,
            ClarificationResolution(
                contract_version="2.0", clarification_id="cl-9", selected_candidate_id=None
            ),
        )


def test_document_candidate_keeps_only_stable_identity_and_order() -> None:
    assert set(DocumentCandidate.model_fields) == {
        "candidate_id",
        "ordinal",
        "ref_id",
        "document_id",
        "label",
    }
    for forbidden in ("score", "debug", "source_task_id", "rank"):
        assert forbidden not in DocumentCandidate.model_fields
