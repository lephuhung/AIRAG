"""Spec §19 — synthesis hydration, admitted claim use IDs, grounding, response.

Covers the brief's Step 1 item "admitted claim use IDs" and the §26 facts:
factual synthesis requires an existing sufficient EvidenceEvaluation, every
claim EvidenceUse ID belongs to the hydrated admitted-use set, two uses of one
EvidenceRecord preserve the selected target/purpose context, and the rendered
citation carries no user-facing claim ID.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.evaluation import (
    Coverage,
    CoverageItem,
    EvidenceEvaluation,
    MissingRequirement,
)
from app.services.agents.v2.contracts.evidence import EvidenceUse
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.response import FinalResponse, RenderedCitation
from app.services.agents.v2.contracts.synthesis import (
    AnswerClaim,
    AnswerDraft,
    SynthesisEvidence,
    SynthesisInput,
    SynthesisRuntimeContext,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_admitted_claim_use_ids,
    validate_answer_draft,
    validate_evidence_evaluation,
    validate_final_response,
    validate_synthesis_input,
)

from .factories import EVIDENCE_ID, USE_ID, read_plan, semantic_context

OTHER_USE_ID = UUID("66666666-6666-6666-6666-666666666666")


def _evaluation(status: str) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        status=status,  # type: ignore[arg-type]
        coverage=Coverage(items=()),
        missing=(),
        contradictions=(),
    )


def _synthesis_input(status: str = "sufficient") -> SynthesisInput:
    return SynthesisInput(
        semantic=semantic_context(),
        evaluation=_evaluation(status),
        evidence_uses=(),
    )


def test_synthesis_input_carries_no_bindings_plan_or_runtime() -> None:
    assert set(SynthesisInput.model_fields) == {"semantic", "evaluation", "evidence_uses"}
    for forbidden in ("bindings", "current_plan", "runtime", "budget"):
        assert forbidden not in SynthesisInput.model_fields


def test_synthesis_requires_a_sufficient_evaluation() -> None:
    validate_synthesis_input(_synthesis_input("sufficient"))
    for status in ("insufficient", "contradictory", "needs_input"):
        with pytest.raises(ContractValidationError, match="sufficient"):
            validate_synthesis_input(_synthesis_input(status))


def test_synthesis_evidence_is_a_computed_projection() -> None:
    assert set(SynthesisEvidence.model_fields) == {
        "use_id",
        "content",
        "role",
        "target_id",
        "source_label",
    }
    assert set(SynthesisRuntimeContext.model_fields) == {
        "max_evidence_items",
        "max_total_chars",
        "max_total_tokens",
    }


def _two_uses_of_one_record() -> tuple[EvidenceUse, EvidenceUse]:
    coverage = EvidenceUse(
        use_id=USE_ID,
        evidence_id=EVIDENCE_ID,
        task_id="T1",
        purpose="coverage",
        target_id="t1",
    )
    supporting = EvidenceUse(
        use_id=OTHER_USE_ID,
        evidence_id=EVIDENCE_ID,
        task_id="T1",
        purpose="supporting",
        target_id="t1",
    )
    assert coverage.evidence_id == supporting.evidence_id
    return coverage, supporting


def test_claim_use_ids_must_be_members_of_the_admitted_set() -> None:
    coverage, supporting = _two_uses_of_one_record()
    admitted = frozenset({coverage.use_id})
    draft = AnswerDraft(
        content="Điều 5 quy định ...",
        claims=(AnswerClaim(claim_id="c1", text="Điều 5 quy định ...", evidence_use_ids=(coverage.use_id,)),),
    )
    validate_answer_draft(draft, admitted)

    with pytest.raises(ContractValidationError, match="admitted"):
        validate_answer_draft(draft, frozenset({supporting.use_id}))


def test_two_uses_of_one_record_preserve_selected_context() -> None:
    coverage, supporting = _two_uses_of_one_record()
    assert (coverage.purpose, coverage.target_id) != (supporting.purpose, supporting.target_id)

    admitted = frozenset({coverage.use_id, supporting.use_id})
    draft = AnswerDraft(
        content="claim",
        claims=(AnswerClaim(claim_id="c1", text="claim", evidence_use_ids=(supporting.use_id,)),),
    )
    validate_answer_draft(draft, admitted)
    assert draft.claims[0].evidence_use_ids == (supporting.use_id,)


def test_admitted_use_ids_are_non_empty_unique_and_within_the_set() -> None:
    validate_admitted_claim_use_ids((USE_ID,), frozenset({USE_ID}))
    with pytest.raises(ContractValidationError, match="unsupported"):
        validate_admitted_claim_use_ids((use_id := uuid4(),), frozenset({USE_ID}))
    assert use_id != USE_ID
    with pytest.raises(ContractValidationError, match="duplicate"):
        validate_admitted_claim_use_ids((USE_ID, USE_ID), frozenset({USE_ID}))
    with pytest.raises(ContractValidationError, match="empty"):
        validate_admitted_claim_use_ids((), frozenset({USE_ID}))


def test_answer_draft_requires_claims_with_unique_ids() -> None:
    claim = AnswerClaim(claim_id="c1", text="claim", evidence_use_ids=(USE_ID,))
    validate_answer_draft(AnswerDraft(content="claim", claims=(claim,)), frozenset({USE_ID}))

    with pytest.raises(ContractValidationError, match="claim"):
        validate_answer_draft(AnswerDraft(content="claim", claims=()), frozenset({USE_ID}))
    with pytest.raises(ContractValidationError, match="claim_id"):
        validate_answer_draft(
            AnswerDraft(content="claim", claims=(claim, claim)), frozenset({USE_ID})
        )
    with pytest.raises(ContractValidationError, match="empty"):
        validate_answer_draft(
            AnswerDraft(content="claim", claims=(AnswerClaim(claim_id="c1", text="t", evidence_use_ids=()),)),
            frozenset({USE_ID}),
        )


def test_rendered_citation_has_no_user_facing_claim_id() -> None:
    assert set(RenderedCitation.model_fields) == {"citation_id", "evidence_id", "label"}
    assert set(FinalResponse.model_fields) == {"contract_version", "status", "content", "citations"}
    validate_final_response(
        FinalResponse(
            contract_version="2.0",
            status="success",
            content="Điều 5 quy định ...",
            citations=(
                RenderedCitation(citation_id="cite-1", evidence_id=EVIDENCE_ID, label="Điều 5"),
            ),
        )
    )
    with pytest.raises(ContractValidationError, match="citation_id"):
        validate_final_response(
            FinalResponse(
                contract_version="2.0",
                status="error",
                content="",
                citations=(
                    RenderedCitation(citation_id="cite-1", evidence_id=EVIDENCE_ID, label="Điều 5"),
                    RenderedCitation(citation_id="cite-1", evidence_id=EVIDENCE_ID, label="Điều 5"),
                ),
            )
        )


def test_missing_requirement_identifies_target_and_criterion_kind() -> None:
    plan = read_plan()
    evaluation = EvidenceEvaluation(
        status="insufficient",
        coverage=Coverage(
            items=(
                CoverageItem(
                    target_id="t1",
                    observed_locators=(DocumentLocator(kind="document"),),
                    status="missing",
                ),
            )
        ),
        missing=(
            MissingRequirement(
                target_id="t1",
                criterion_kind="coverage",
                semantic_criterion_id=None,
                description="Điều 5 chưa đọc",
            ),
        ),
        contradictions=(),
    )
    validate_evidence_evaluation(evaluation, plan)

    with pytest.raises(ContractValidationError, match="semantic_criterion_id"):
        validate_evidence_evaluation(
            evaluation.model_copy(
                update={
                    "missing": (
                        MissingRequirement(
                            target_id="t1",
                            criterion_kind="coverage",
                            semantic_criterion_id="c1",
                            description="invalid",
                        ),
                    )
                }
            ),
            plan,
        )
    with pytest.raises(ContractValidationError, match="tX"):
        validate_evidence_evaluation(
            evaluation.model_copy(
                update={
                    "missing": (
                        MissingRequirement(
                            target_id="tX",
                            criterion_kind="coverage",
                            semantic_criterion_id=None,
                            description="dangling target",
                        ),
                    )
                }
            ),
            plan,
        )


def test_sufficient_evaluation_cannot_report_missing_requirements() -> None:
    plan = read_plan()
    evaluation = _evaluation("sufficient").model_copy(
        update={
            "missing": (
                MissingRequirement(
                    target_id="t1",
                    criterion_kind="coverage",
                    semantic_criterion_id=None,
                    description="nope",
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError, match="sufficient"):
        validate_evidence_evaluation(evaluation, plan)
