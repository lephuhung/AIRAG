"""Spec §11/§13.3/§13.4/§16 — capability contracts.

Covers the brief's Step 2 ownership rule ("capability.py owns the frozen
CapabilityDescriptor, CapabilityInput, CapabilityOutput, and the runtime-only
CapabilityRuntimeContext") and the §26 fact "typed capability input/output
without generic dict fallback".
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, get_args, get_origin
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from app.services.agents.v2.contracts.binding import DocumentDiscoveryCandidate
from app.services.agents.v2.contracts.capability import (
    AbbreviationResolveInput,
    AbbreviationResolveOutput,
    CapabilityDescriptor,
    CapabilityInput,
    CapabilityOutput,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
    DocumentSearchInput,
    DocumentSearchOutput,
    KnowledgeGraphInput,
    KnowledgeGraphOutput,
    MemoryLookupInput,
    MemoryLookupOutput,
    PeopleLookupInput,
    PeopleLookupOutput,
    SectionReadInput,
    SectionReadOutput,
    WriteInput,
    WriteOutput,
)
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult

EXPECTED_KINDS = [
    "people.lookup",
    "document.search",
    "document.read",
    "section.read",
    "write",
    "knowledge_graph.query",
    "memory.lookup",
    "abbreviation.resolve",
]

CAPABILITY_INPUT_VARIANTS = (
    PeopleLookupInput,
    DocumentSearchInput,
    DocumentReadInput,
    SectionReadInput,
    WriteInput,
    KnowledgeGraphInput,
    MemoryLookupInput,
    AbbreviationResolveInput,
)

CAPABILITY_OUTPUT_VARIANTS = (
    PeopleLookupOutput,
    DocumentSearchOutput,
    DocumentReadOutput,
    SectionReadOutput,
    WriteOutput,
    KnowledgeGraphOutput,
    MemoryLookupOutput,
    AbbreviationResolveOutput,
)


def _union_variants(alias: object) -> tuple[type[object], ...]:
    assert get_origin(alias) is Annotated
    union, field = get_args(alias)
    assert getattr(field, "discriminator", None) == "kind"
    return get_args(union)


@pytest.mark.parametrize("alias", [CapabilityInput, CapabilityOutput], ids=["input", "output"])
def test_capability_unions_are_closed_discriminated_unions(alias: object) -> None:
    variants = _union_variants(alias)
    assert len(variants) == 8
    kinds = [get_args(variant.model_fields["kind"].annotation)[0] for variant in variants]
    assert kinds == EXPECTED_KINDS
    assert all(
        get_origin(variant.model_fields["kind"].annotation) is Literal for variant in variants
    )


def test_capability_variants_are_frozen_strict_and_extra_forbidden() -> None:
    for variant in CAPABILITY_INPUT_VARIANTS + CAPABILITY_OUTPUT_VARIANTS:
        assert variant.model_config["extra"] == "forbid"
        assert variant.model_config["frozen"] is True
        assert variant.model_config["strict"] is True


def test_unknown_capability_kind_fails_closed() -> None:
    adapter = TypeAdapter(CapabilityInput)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "document.summarize"})


def test_agent_request_parses_a_discriminated_input() -> None:
    request = AgentRequest.model_validate(
        {
            "contract_version": "2.0",
            "task_id": "T1",
            "objective": "Đọc Điều 5 của A",
            "input": {"kind": "document.read", "target_ids": ("t1",)},
        }
    )
    assert isinstance(request.input, DocumentReadInput)
    assert request.input.target_ids == ("t1",)
    assert set(AgentRequest.model_fields) == {"contract_version", "task_id", "objective", "input"}
    for forbidden in ("capability", "depends_on", "runtime", "workspace_ids"):
        assert forbidden not in AgentRequest.model_fields


def test_agent_result_carries_only_checkpoint_safe_execution_facts() -> None:
    assert set(AgentResult.model_fields) == {
        "contract_version",
        "task_id",
        "status",
        "data",
        "evidence_uses",
        "coverage_observations",
        "error",
    }
    for forbidden in ("request_id", "missing_requirements", "sufficiency"):
        assert forbidden not in AgentResult.model_fields


def test_read_inputs_reference_targets_instead_of_copying_documents() -> None:
    read = DocumentReadInput(kind="document.read", target_ids=("t1", "t2"))
    section = SectionReadInput(kind="section.read", target_ids=("t1",))
    assert read.target_ids == ("t1", "t2")
    assert section.target_ids == ("t1",)
    for variant in (DocumentReadInput, SectionReadInput):
        for forbidden in ("document_id", "document_revision", "role", "binding_id"):
            assert forbidden not in variant.model_fields


def test_read_outputs_expose_counts_not_content() -> None:
    for variant in (DocumentReadOutput, SectionReadOutput):
        assert set(variant.model_fields) == {"kind", "read_unit_count"}
        for forbidden in ("content", "text", "chunks", "markdown"):
            assert forbidden not in variant.model_fields


def test_search_returns_discovery_candidates_owned_by_the_result() -> None:
    assert set(DocumentDiscoveryCandidate.model_fields) == {
        "candidate_id",
        "document_id",
        "document_revision",
    }
    assert "task_id" not in DocumentDiscoveryCandidate.model_fields
    output = DocumentSearchOutput(
        kind="document.search",
        candidates=(
            DocumentDiscoveryCandidate(
                candidate_id=UUID("77777777-7777-7777-7777-777777777777"),
                document_id=UUID("11111111-1111-1111-1111-111111111111"),
                document_revision="rev-1",
            ),
        ),
    )
    assert output.candidates[0].document_revision == "rev-1"


def test_people_output_is_minimized() -> None:
    assert set(PeopleLookupOutput.model_fields) == {"kind", "matched"}
    for forbidden in ("cccd", "dob", "address", "record", "content"):
        assert forbidden not in PeopleLookupOutput.model_fields


def test_capability_descriptor_is_the_minimal_planner_catalog_entry() -> None:
    descriptor = CapabilityDescriptor(
        name="document.search",
        domain="document",
        operation_type="search",
        supports_parallel=True,
    )
    assert set(CapabilityDescriptor.model_fields) == {
        "name",
        "domain",
        "operation_type",
        "supports_parallel",
    }
    for forbidden in ("version", "description", "behavior_flags", "provider"):
        assert forbidden not in CapabilityDescriptor.model_fields
    assert descriptor.operation_type == "search"


def test_capability_runtime_context_carries_current_trusted_authority() -> None:
    context = CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=UUID("88888888-8888-8888-8888-888888888888"),
        workspace_ids=(UUID("99999999-9999-9999-9999-999999999999"),),
        can_read_people=True,
        allowed_capabilities=frozenset({"people.lookup", "document.read"}),
        deadline_at=datetime(2026, 9, 11, tzinfo=UTC),
    )
    assert context.can_read_people is True
    assert "config_revision" not in CapabilityRuntimeContext.model_fields
    assert "contract_version" not in CapabilityRuntimeContext.model_fields
