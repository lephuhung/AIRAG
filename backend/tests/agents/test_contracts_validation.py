"""
Tests for contracts_validation module (A.7).

Per spec A.7 / Q16: validate_task_plan, validate_evidence_collection.
"""

from __future__ import annotations

import uuid
import pytest


def test_validate_task_plan_duplicate_task_id():
    """Duplicate task_id raises ValidationError."""
    from app.services.agents.contracts_validation import validate_task_plan, ValidationError
    from app.services.agents.semantic_preprocessor import PreprocessingResult
    from app.services.agents.deep_research.contracts import TaskSpec

    tasks = [
        TaskSpec(task_id="t1", work_type="retrieve_section", target_ref="r1"),
        TaskSpec(task_id="t1", work_type="retrieve_section", target_ref="r2"),  # duplicate
    ]
    ctx = PreprocessingResult(
        original_query="test",
        preprocessing_status="ok",
        preprocessor_trace=[],
        document_refs=[],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_task_plan(tasks, ctx)
    assert "duplicate task_id" in str(exc_info.value).lower()


def test_validate_task_plan_acyclic():
    """Cyclic depends_on raises ValidationError."""
    from app.services.agents.contracts_validation import validate_task_plan, ValidationError
    from app.services.agents.semantic_preprocessor import PreprocessingResult
    from app.services.agents.deep_research.contracts import TaskSpec

    tasks = [
        TaskSpec(task_id="t1", work_type="retrieve_section", target_ref="r1", depends_on=["t3"]),
        TaskSpec(task_id="t2", work_type="retrieve_section", target_ref="r2", depends_on=["t1"]),
        TaskSpec(task_id="t3", work_type="retrieve_section", target_ref="r1", depends_on=["t2"]),
    ]
    ctx = PreprocessingResult(
        original_query="test",
        preprocessing_status="ok",
        preprocessor_trace=[],
        document_refs=[],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_task_plan(tasks, ctx)
    assert "cyclic" in str(exc_info.value).lower()


def test_validate_task_plan_target_ref_in_context():
    """target_ref not in semantic_context raises ValidationError."""
    from app.services.agents.contracts_validation import validate_task_plan, ValidationError
    from app.services.agents.semantic_preprocessor import PreprocessingResult, DocumentRefEntry
    from app.services.agents.deep_research.contracts import TaskSpec

    tasks = [
        TaskSpec(task_id="t1", work_type="retrieve_section", target_ref="r999"),  # unknown ref
    ]
    ctx = PreprocessingResult(
        original_query="Nghị định 13 và Nghị định 24",
        preprocessing_status="ok",
        preprocessor_trace=[],
        document_refs=[
            DocumentRefEntry(
                ref_id="r1",
                original_span="Nghị định 13",
                span_offset=(0, 12),
                reference="nghi dinh 13",
                resolution_status="resolved",
            ),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_task_plan(tasks, ctx)
    assert "target_ref" in str(exc_info.value).lower()
    assert "r999" in str(exc_info.value)


def test_validate_task_plan_valid():
    """Valid plan with no cycles and known refs passes."""
    from app.services.agents.contracts_validation import validate_task_plan
    from app.services.agents.semantic_preprocessor import PreprocessingResult, DocumentRefEntry
    from app.services.agents.deep_research.contracts import TaskSpec

    tasks = [
        TaskSpec(task_id="t1", work_type="retrieve_section", target_ref="r1"),
        TaskSpec(task_id="t2", work_type="retrieve_section", target_ref="r2", depends_on=["t1"]),
    ]
    ctx = PreprocessingResult(
        original_query="Nghị định 13 và Nghị định 24",
        preprocessing_status="ok",
        preprocessor_trace=[],
        document_refs=[
            DocumentRefEntry(
                ref_id="r1",
                original_span="Nghị định 13",
                span_offset=(0, 12),
                reference="nghi dinh 13",
                resolution_status="resolved",
            ),
            DocumentRefEntry(
                ref_id="r2",
                original_span="Nghị định 24",
                span_offset=(16, 28),
                reference="nghi dinh 24",
                resolution_status="resolved",
            ),
        ],
    )
    # Should NOT raise
    validate_task_plan(tasks, ctx)


def test_validate_evidence_collection_missing_id():
    """Evidence ID in task_result but not in evidence raises ValidationError."""
    from app.services.agents.contracts_validation import validate_evidence_collection, ValidationError
    from app.services.agents.deep_research.contracts import Evidence, TaskResult, Coverage, Provenance

    evidence = [
        Evidence(
            evidence_id="t1:c1",
            task_id="t1",
            source_id=str(uuid.uuid4()),
            raw_content="test",
            content_hash="a" * 64,
            content_size_bytes=4,
            raw_content_bytes=4,
            provenance=Provenance(
                fetcher="deep_worker",
                fetched_at=0.0,
                fetched_by=uuid.uuid4(),
                workspace_scope=[],
                run_id="r1",
            ),
        ),
    ]
    result = TaskResult(
        task_id="t1",
        status="ok",
        evidence_ids=["t1:c1", "t1:c999"],  # c999 not in evidence
    )

    with pytest.raises(ValidationError) as exc_info:
        validate_evidence_collection(evidence, result)
    assert "t1:c999" in str(exc_info.value)


def test_validate_evidence_collection_valid():
    """Valid evidence collection passes."""
    from app.services.agents.contracts_validation import validate_evidence_collection
    from app.services.agents.deep_research.contracts import Evidence, TaskResult, Coverage, Provenance

    eid = str(uuid.uuid4())
    evidence = [
        Evidence(
            evidence_id=eid,
            task_id="t1",
            source_id=str(uuid.uuid4()),
            raw_content="x",
            content_hash="a" * 64,
            content_size_bytes=1,
            raw_content_bytes=1,
            provenance=Provenance(
                fetcher="deep_worker",
                fetched_at=0.0,
                fetched_by=uuid.uuid4(),
                workspace_scope=[],
                run_id="r1",
            ),
        ),
    ]
    result = TaskResult(
        task_id="t1",
        status="ok",
        evidence_ids=[eid],
        coverage=Coverage(requested=1, resolved=1, read=1, truncated=0),
    )
    # Should NOT raise
    validate_evidence_collection(evidence, result)
