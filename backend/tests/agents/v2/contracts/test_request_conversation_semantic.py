"""Spec §8 — request, conversation, and semantic lifecycle contracts.

Covers "optional revision semantics" (pin-once ``None``, explicit current,
explicit pinned) and the §26 ownership facts: ``RequestContext`` alone owns
``original_query``, ``SemanticContext`` carries no binding projection, and
``DocumentBindingSet`` carries no unresolved projection.
"""
from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from app.services.agents.v2.contracts.binding import BindingRevisionRequirement, DocumentBindingSet
from app.services.agents.v2.contracts.clarification import ClarificationRequest, ClarificationResolution, DocumentCandidate
from app.services.agents.v2.contracts.conversation import (
    ActiveEntity,
    ConversationContext,
    ConversationSnapshot,
    ConversationTurn,
    EntityReference,
)
from app.services.agents.v2.contracts.request import KnownDocumentResource, RequestContext
from app.services.agents.v2.contracts.semantic import (
    CurrentRevisionRequirement,
    DocumentReference,
    PinnedRevisionRequirement,
    RevisionRequirement,
    SemanticContext,
    SemanticDraft,
    SemanticSnapshot,
)

from .factories import DOCUMENT_ID, OTHER_DOCUMENT_ID, document_reference, request_context

REQUIREMENT_ADAPTER: TypeAdapter[object] = TypeAdapter(RevisionRequirement)


def test_known_document_resource_is_identity_not_role() -> None:
    resource = KnownDocumentResource(resource_id="res-1", document_id=DOCUMENT_ID, source="attachment")
    assert set(resource.model_fields) == {"resource_id", "document_id", "source"}


def test_request_context_owns_the_only_raw_query() -> None:
    assert "original_query" in RequestContext.model_fields
    assert "original_query" not in SemanticContext.model_fields
    assert "original_query" not in SemanticDraft.model_fields


def test_semantic_context_has_no_binding_or_original_query_projection() -> None:
    forbidden = ("contract_version", "original_query", "binding_id", "bindings")
    for model in (SemanticContext, SemanticDraft):
        assert not any(name in model.model_fields for name in forbidden)


def test_document_binding_set_has_no_unresolved_projection() -> None:
    assert set(DocumentBindingSet.model_fields) == {"bindings", "revision_requirement_refs"}
    assert set(BindingRevisionRequirement.model_fields) == {"binding_id", "ref_id"}


def test_ordinary_reference_defaults_to_pin_once_none() -> None:
    reference = DocumentReference(
        ref_id="r1",
        original_span="A",
        normalized_reference="A",
        requested_role="target",
        resolution_status="resolved",
        resolved_document_id=DOCUMENT_ID,
    )
    assert reference.revision_requirement is None


def test_explicit_current_requirement_carries_no_revision() -> None:
    assert set(CurrentRevisionRequirement.model_fields) == {"kind"}
    assert set(PinnedRevisionRequirement.model_fields) == {"kind", "document_revision"}


def test_revision_requirement_discriminates_on_kind() -> None:
    current = REQUIREMENT_ADAPTER.validate_python({"kind": "current"})
    pinned = REQUIREMENT_ADAPTER.validate_python({"kind": "pinned", "document_revision": "rev-0"})
    assert isinstance(current, CurrentRevisionRequirement)
    assert isinstance(pinned, PinnedRevisionRequirement)
    assert pinned.document_revision == "rev-0"
    with pytest.raises(ValidationError):
        REQUIREMENT_ADAPTER.validate_python({"kind": "latest"})


@pytest.mark.parametrize(
    ("status", "resolved", "candidates"),
    [
        ("resolved", None, ()),
        ("ambiguous", DOCUMENT_ID, (DOCUMENT_ID, OTHER_DOCUMENT_ID)),
        ("ambiguous", None, (DOCUMENT_ID,)),
        ("not_found", None, (DOCUMENT_ID,)),
        ("unresolved", DOCUMENT_ID, ()),
        ("error", DOCUMENT_ID, ()),
    ],
)
def test_document_reference_resolution_status_invariants(
    status: str, resolved: UUID | None, candidates: tuple[UUID, ...]
) -> None:
    from app.services.agents.v2.contracts.validation import (
        ContractValidationError,
        validate_semantic_context,
    )

    reference = document_reference(
        resolution_status=status,
        resolved_document_id=resolved,
        candidate_document_ids=candidates,
    )
    with pytest.raises(ContractValidationError):
        validate_semantic_context(
            SemanticContext(
                contextualized_query="q",
                normalized_query="q",
                abbreviations=(),
                coreferences=(),
                document_refs=(reference,),
                person_refs=(),
                section_refs=(),
                blocking_ambiguities=(),
            )
        )


def test_resolved_reference_with_canonical_id_is_valid() -> None:
    from app.services.agents.v2.contracts.validation import validate_semantic_context

    validate_semantic_context(
        SemanticContext(
            contextualized_query="q",
            normalized_query="q",
            abbreviations=(),
            coreferences=(),
            document_refs=(document_reference(),),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        )
    )


def test_conversation_snapshot_owns_persistence_cas_metadata() -> None:
    assert set(ConversationSnapshot.model_fields) == {
        "contract_version",
        "thread_id",
        "summary_version",
        "built_through_message_id",
        "context",
    }
    assert "summary_version" not in ConversationContext.model_fields
    assert "thread_id" not in ConversationContext.model_fields


def test_conversation_context_has_no_open_questions_duplicate() -> None:
    assert "open_questions" not in ConversationContext.model_fields
    assert set(ConversationContext.model_fields) == {
        "summary",
        "active_entities",
        "last_focus",
        "recent_turns",
    }


def test_conversation_helpers_have_minimal_shapes() -> None:
    entity = EntityReference(ref_id="e1", kind="document", label="A")
    active = ActiveEntity(ref_id="e1", kind="document", label="A")
    turn = ConversationTurn(role="user", content="A là gì?")
    assert isinstance(active, EntityReference)
    assert set(ConversationTurn.model_fields) == {"role", "content"}
    assert set(EntityReference.model_fields) == {"ref_id", "kind", "label"}
    assert entity.label == active.label == "A"
    assert turn.role == "user"


def test_semantic_snapshot_only_versions_the_envelope() -> None:
    assert set(SemanticSnapshot.model_fields) == {"contract_version", "semantic"}


def test_clarification_request_persists_only_question_specific_refs() -> None:
    assert set(ClarificationRequest.model_fields) == {
        "contract_version",
        "clarification_id",
        "reason",
        "question",
        "unresolved_ref_ids",
        "candidates",
        "expires_at",
    }
    assert "user_text" not in ClarificationResolution.model_fields
    assert set(DocumentCandidate.model_fields) == {
        "candidate_id",
        "ordinal",
        "ref_id",
        "document_id",
        "label",
    }


def test_binding_set_validates_against_the_semantic_context() -> None:
    from app.services.agents.v2.contracts.validation import validate_binding_set

    validate_binding_set(
        DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        SemanticContext(
            contextualized_query="greeting",
            normalized_query="greeting",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        ),
    )
    assert request_context().request_id == "req-1"
