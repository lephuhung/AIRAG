"""Discovery spec §15 — document-selection clarification contracts (Task 1)."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.services.agents.v2.contracts.clarification import (
    MAX_PUBLIC_CHOICES,
    ClarificationRequest,
    ClarificationResolution,
    DocumentSelectionChoice,
    DocumentSelectionClarification,
    DocumentSelectionManifest,
    DocumentSelectionManifestEntry,
    DocumentSelectionResolution,
    DocumentSelectionSlot,
    DocumentSlotResolution,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    IncompatibleCheckpointError,
    validate_clarification_slot_exclusion,
    validate_document_selection_clarification,
    validate_document_selection_resolution,
)

from . import factories

EXPIRES = datetime(2026, 9, 18, tzinfo=UTC)


def _choice(token: str = "tok-1", title: str = "Nghị định 13") -> DocumentSelectionChoice:
    return DocumentSelectionChoice(
        choice_token=token, title=title, document_number=None
    )


def _slot(
    slot_id: str = "slot-1",
    *,
    choices: tuple[DocumentSelectionChoice, ...] | None = None,
    min_selections: int = 1,
    max_selections: int = 1,
) -> DocumentSelectionSlot:
    if choices is None:
        choices = (_choice(),)
    return DocumentSelectionSlot(
        slot_id=slot_id,
        slot_label="Tài liệu mục tiêu",
        min_selections=min_selections,
        max_selections=max_selections,
        choices=choices,
    )


def _request(
    *,
    slots: tuple[DocumentSelectionSlot, ...] | None = None,
    expires_at: datetime = EXPIRES,
) -> DocumentSelectionClarification:
    if slots is None:
        slots = (_slot(),)
    return DocumentSelectionClarification(
        kind="document_selection",
        contract_version="2.0",
        clarification_id="clar-1",
        question="Chọn tài liệu cần tóm tắt?",
        slots=slots,
        expires_at=expires_at,
    )


def _resolution(
    *,
    selections: tuple[DocumentSlotResolution, ...] | None = None,
    declined: bool = False,
) -> DocumentSelectionResolution:
    if selections is None:
        selections = (DocumentSlotResolution(slot_id="slot-1", choice_tokens=("tok-1",)),)
    return DocumentSelectionResolution(
        kind="document_selection",
        contract_version="2.0",
        clarification_id="clar-1",
        selections=selections,
        declined=declined,
    )


def test_valid_request_and_resolution_pass() -> None:
    request = _request()
    validate_document_selection_clarification(request)
    validate_document_selection_resolution(request, _resolution())


def test_legacy_semantic_kind_defaults() -> None:
    request = ClarificationRequest(
        contract_version="2.0",
        clarification_id="c1",
        reason="required_document_ambiguous",
        question="q?",
        unresolved_ref_ids=(),
        candidates=(),
        expires_at=EXPIRES,
    )
    resolution = ClarificationResolution(
        contract_version="2.0", clarification_id="c1", selected_candidate_id=None
    )
    assert request.kind == "semantic"
    assert resolution.kind == "semantic"


@pytest.mark.parametrize(
    "field,value",
    [
        ("clarification_id", " "),
        ("question", ""),
    ],
)
def test_non_blank_request_fields(field: str, value: str) -> None:
    request = _request().model_copy(update={field: value})
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_naive_expiry_rejected() -> None:
    request = _request(expires_at=datetime(2026, 9, 18))  # noqa: DTZ001 - intentionally naive
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_duplicate_slot_id_rejected() -> None:
    request = _request(slots=(_slot("s1"), _slot("s1", choices=(_choice("tok-2"),))))
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_global_token_uniqueness_across_slots() -> None:
    request = _request(slots=(_slot("s1"), _slot("s2")))
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_choices_bounded_by_max_public() -> None:
    choices = tuple(_choice(f"tok-{i}") for i in range(MAX_PUBLIC_CHOICES + 1))
    request = _request(
        slots=(
            _slot(
                "s1",
                choices=choices,
                min_selections=1,
                max_selections=len(choices),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_empty_choices_rejected() -> None:
    request = _request(slots=(_slot("s1", choices=(), min_selections=0, max_selections=0),))
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_blank_choice_token_and_title_rejected() -> None:
    request = _request(slots=(_slot("s1", choices=(_choice(token=" "),)),))
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)
    request = _request(slots=(_slot("s1", choices=(_choice(title=" "),)),))
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_slot_cardinality_checked_against_choices() -> None:
    request = _request(
        slots=(_slot("s1", min_selections=2, max_selections=2),)
    )
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_declined_xor_selections() -> None:
    request = _request()
    validate_document_selection_resolution(
        request, _resolution(selections=(), declined=True)
    )
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            request, _resolution(declined=True)
        )
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            request, _resolution(selections=(), declined=False)
        )


def test_resolution_rejects_unknown_slot_and_token() -> None:
    request = _request()
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            request,
            _resolution(
                selections=(
                    DocumentSlotResolution(slot_id="nope", choice_tokens=("tok-1",)),
                )
            ),
        )
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            request,
            _resolution(
                selections=(
                    DocumentSlotResolution(slot_id="slot-1", choice_tokens=("forged",)),
                )
            ),
        )


def test_resolution_enforces_slot_cardinality() -> None:
    request = _request(
        slots=(
            _slot(
                "s1",
                choices=(_choice("t1"), _choice("t2")),
                min_selections=1,
                max_selections=1,
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            request,
            _resolution(
                selections=(
                    DocumentSlotResolution(
                        slot_id="s1", choice_tokens=("t1", "t2")
                    ),
                )
            ),
        )


def test_resolution_clarification_id_mismatch() -> None:
    request = _request()
    resolution = _resolution().model_copy(update={"clarification_id": "other"})
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(request, resolution)


def test_clarification_slot_mutual_exclusion() -> None:
    semantic_request = ClarificationRequest(
        contract_version="2.0",
        clarification_id="c1",
        reason="required_document_ambiguous",
        question="q?",
        unresolved_ref_ids=(),
        candidates=(),
        expires_at=EXPIRES,
    )
    validate_clarification_slot_exclusion(None, None)
    validate_clarification_slot_exclusion(semantic_request, None)
    validate_clarification_slot_exclusion(None, _request())
    with pytest.raises(ContractValidationError):
        validate_clarification_slot_exclusion(semantic_request, _request())


def test_empty_request_slots_rejected() -> None:
    request = _request(slots=())
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_min_selections_must_be_at_least_one() -> None:
    request = _request(
        slots=(_slot("s1", min_selections=0, max_selections=1),)
    )
    with pytest.raises(ContractValidationError):
        validate_document_selection_clarification(request)


def test_resolution_must_cover_every_request_slot() -> None:
    request = _request(
        slots=(
            _slot("s1"),
            _slot("s2", choices=(_choice("tok-2"),)),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            request,
            _resolution(
                selections=(
                    DocumentSlotResolution(slot_id="s1", choice_tokens=("tok-1",)),
                )
            ),
        )
    validate_document_selection_resolution(
        request,
        _resolution(
            selections=(
                DocumentSlotResolution(slot_id="s1", choice_tokens=("tok-1",)),
                DocumentSlotResolution(slot_id="s2", choice_tokens=("tok-2",)),
            )
        ),
    )


def test_manifest_stores_digests_not_tokens() -> None:
    manifest = DocumentSelectionManifest(
        clarification_id="clar-1",
        entries=(
            DocumentSelectionManifestEntry(
                choice_token_digest="hmac:abc",
                slot_id="slot-1",
                aggregate_id=uuid4(),
                document_id=factories.DOCUMENT_ID,
                document_revision="rev-1",
            ),
        ),
        expires_at=EXPIRES,
        status="pending",
    )
    assert manifest.entries[0].choice_token_digest == "hmac:abc"


def test_exception_identity_reexported() -> None:
    from app.services.agents.v2.contracts import validation, validation_support

    assert validation.ContractValidationError is validation_support.ContractValidationError
    assert (
        validation.IncompatibleCheckpointError
        is validation_support.IncompatibleCheckpointError
    )
    assert issubclass(
        validation_support.IncompatibleCheckpointError,
        validation_support.ContractValidationError,
    )
    assert issubclass(validation.IncompatibleCheckpointError, ValueError)
    with pytest.raises(IncompatibleCheckpointError):
        raise validation.IncompatibleCheckpointError("boom")
