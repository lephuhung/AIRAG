"""Spec §15 — EvidenceRecord/EvidenceUse separation and use-purpose rules.

Covers the brief's Step 1 items "EvidenceRecord/EvidenceUse separation" and
"purpose/target rules", plus the §26 facts: revision mismatch cannot complete
coverage, discovery uses cannot synthesize, targetless supporting uses require
a validated targetless task, and derived evidence needs recursive validated
sources and never creates read coverage.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.capability import DocumentSearchOutput
from app.services.agents.v2.contracts.evidence import (
    DerivedSourceIdentity,
    EvidencePurpose,
    EvidenceRecord,
    EvidenceStoreRow,
    EvidenceUse,
    EvidenceUseEnvelope,
    EvidenceUseRef,
    PeopleSourceIdentity,
    StoragePolicy,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_derived_evidence_faithfulness,
    validate_evidence_evaluation,
    validate_evidence_record,
    validate_evidence_store_row,
    validate_evidence_use,
    validate_evidence_use_resolution,
    validate_synthesis_use,
    validate_target_use_revision,
)

from .factories import (
    DOCUMENT_ID,
    EVIDENCE_ID,
    OTHER_REVISION,
    REVISION,
    USE_ID,
    binding_set,
    evidence_record,
    read_plan,
)


def _use(purpose: str, target_id: str | None, *, task_id: str = "T1") -> EvidenceUse:
    return EvidenceUse(
        use_id=USE_ID,
        evidence_id=EVIDENCE_ID,
        task_id=task_id,
        purpose=purpose,  # type: ignore[arg-type]
        target_id=target_id,
    )


def test_evidence_record_contains_identity_only() -> None:
    assert set(EvidenceRecord.model_fields) == {
        "evidence_id",
        "source",
        "content",
        "content_hash",
        "provenance",
    }
    for forbidden in ("task_id", "run_id", "target_id", "purpose", "label", "storage_policy"):
        assert forbidden not in EvidenceRecord.model_fields


def test_evidence_use_contains_usage_only() -> None:
    assert set(EvidenceUse.model_fields) == {
        "use_id",
        "evidence_id",
        "task_id",
        "purpose",
        "target_id",
    }
    for forbidden in ("content", "content_hash", "document_revision", "run_id"):
        assert forbidden not in EvidenceUse.model_fields


def test_envelope_owns_run_identity() -> None:
    assert set(EvidenceUseEnvelope.model_fields) == {"contract_version", "run_id", "use"}
    envelope = EvidenceUseEnvelope(contract_version="2.0", run_id="run-1", use=_use("coverage", "t1"))
    assert envelope.use.use_id == USE_ID
    assert set(EvidenceUseRef.model_fields) == {"use_id"}
    assert envelope.use.task_id == "T1"


def test_storage_policy_stays_in_the_store_row() -> None:
    assert set(EvidenceStoreRow.model_fields) == {"contract_version", "record", "storage_policy"}
    assert set(StoragePolicy.model_fields) == {"classification", "expires_at"}
    validate_evidence_store_row(
        EvidenceStoreRow(
            contract_version="2.0",
            record=evidence_record(),
            storage_policy=StoragePolicy(classification="normal", expires_at=None),
        )
    )


def test_evidence_purpose_is_closed() -> None:
    from typing import get_args

    assert set(get_args(EvidencePurpose)) == {"discovery", "coverage", "supporting"}


def test_coverage_use_requires_a_target() -> None:
    with pytest.raises(ContractValidationError, match="coverage"):
        validate_evidence_use(_use("coverage", None))


def test_discovery_use_requires_no_target() -> None:
    validate_evidence_use(_use("discovery", None))
    with pytest.raises(ContractValidationError, match="discovery"):
        validate_evidence_use(_use("discovery", "t1"))


def test_supporting_use_may_be_targetless() -> None:
    validate_evidence_use(_use("supporting", None))
    validate_evidence_use(_use("supporting", "t1"))


def test_use_resolution_requires_a_checkpointed_task_and_target() -> None:
    plan = read_plan()
    validate_evidence_use_resolution(_use("coverage", "t1"), plan)

    with pytest.raises(ContractValidationError, match="task"):
        validate_evidence_use_resolution(_use("coverage", "t1", task_id="TX"), plan)
    with pytest.raises(ContractValidationError, match="target"):
        validate_evidence_use_resolution(_use("coverage", "tX"), plan)


def test_revision_mismatch_cannot_complete_coverage() -> None:
    plan = read_plan()
    bindings = binding_set()

    validate_target_use_revision(_use("coverage", "t1"), plan, bindings, evidence_record())
    with pytest.raises(ContractValidationError, match="revision"):
        validate_target_use_revision(
            _use("coverage", "t1"), plan, bindings, evidence_record(document_revision=OTHER_REVISION)
        )
    with pytest.raises(ContractValidationError, match="revision"):
        validate_target_use_revision(
            _use("coverage", "t1"), plan, bindings, evidence_record(document_id=uuid4())
        )


def test_target_use_requires_a_document_record() -> None:
    people_record = evidence_record().model_copy(
        update={"source": PeopleSourceIdentity(kind="people", record_id="p1")}
    )
    with pytest.raises(ContractValidationError, match="document"):
        validate_target_use_revision(_use("coverage", "t1"), read_plan(), binding_set(), people_record)


def test_derived_evidence_requires_recursive_validated_sources() -> None:
    source = evidence_record()
    derived = evidence_record(evidence_id=uuid4()).model_copy(
        update={
            "source": DerivedSourceIdentity(
                kind="derived",
                source_evidence_ids=(source.evidence_id,),
            )
        }
    )
    validate_derived_evidence_faithfulness(
        derived, (source,), faithfulness_validated=True
    )
    with pytest.raises(ContractValidationError, match="validation"):
        validate_derived_evidence_faithfulness(
            derived, (source,), faithfulness_validated=False
        )
    with pytest.raises(ContractValidationError, match="source"):
        validate_derived_evidence_faithfulness(derived, (), faithfulness_validated=True)
    with pytest.raises(ContractValidationError, match="derived"):
        validate_derived_evidence_faithfulness(source, (source,), faithfulness_validated=True)


def test_derived_evidence_never_creates_read_coverage() -> None:
    derived = evidence_record().model_copy(
        update={
            "source": DerivedSourceIdentity(
                kind="derived", source_evidence_ids=(uuid4(),)
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_target_use_revision(_use("coverage", "t1"), read_plan(), binding_set(), derived)


def test_discovery_use_is_not_synthesis_eligible() -> None:
    plan = read_plan()
    with pytest.raises(ContractValidationError, match="discovery"):
        validate_synthesis_use(_use("discovery", None), plan)
    validate_synthesis_use(_use("coverage", "t1"), plan)


def test_targetless_supporting_use_requires_a_targetless_task() -> None:
    plan = read_plan()
    with pytest.raises(ContractValidationError, match="targetless"):
        validate_evidence_use_resolution(_use("supporting", None), plan)


def test_derived_record_carries_no_document_coordinates() -> None:
    assert set(DerivedSourceIdentity.model_fields) == {"kind", "source_evidence_ids"}


def test_search_output_structurally_cannot_report_read_coverage() -> None:
    assert "coverage_observations" not in DocumentSearchOutput.model_fields
    assert set(DocumentSearchOutput.model_fields) == {"kind", "candidates"}


def test_record_identity_and_use_identity_are_distinct() -> None:
    record = evidence_record(evidence_id=EVIDENCE_ID)
    first = EvidenceUse(
        use_id=uuid4(), evidence_id=record.evidence_id, task_id="T1", purpose="coverage", target_id="t1"
    )
    second = EvidenceUse(
        use_id=uuid4(), evidence_id=record.evidence_id, task_id="T1", purpose="supporting", target_id="t1"
    )
    assert first.evidence_id == second.evidence_id
    assert first.use_id != second.use_id
    assert (first.purpose, first.target_id) != (second.purpose, second.target_id)


def test_evidence_record_requires_hash_and_revision() -> None:
    with pytest.raises(ContractValidationError, match="content_hash"):
        validate_evidence_record(evidence_record().model_copy(update={"content_hash": " "}))
    with pytest.raises(ContractValidationError, match="document_revision"):
        validate_evidence_record(
            evidence_record().model_copy(
                update={
                    "source": evidence_record().source.model_copy(
                        update={"document_revision": " "}
                    )
                }
            )
        )
    validate_evidence_record(evidence_record())
    assert REVISION and DOCUMENT_ID is not None and UUID(str(DOCUMENT_ID)) == DOCUMENT_ID
