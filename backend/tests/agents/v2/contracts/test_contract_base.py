"""Spec §3 — strict/frozen base, envelope versioning, and contract purity.

Covers the brief's Step 1 items "strict/frozen/envelope versioning" and the
§26 acceptance facts "root-versioned mutable SupervisorV2State, strict/frozen
nested persisted business contracts, and unversioned runtime context" and
"typed capability input/output without generic dict fallback".
"""
from __future__ import annotations

import pathlib
import re
import typing
from typing import Any, Literal, get_args, get_origin

import pytest
from pydantic import ValidationError

from app.services.agents.v2.contracts import (
    binding,
    capability,
    clarification,
    conversation,
    evaluation,
    evidence,
    execution,
    locators,
    planning,
    request,
    response,
    routing,
    semantic,
    state,
    synthesis,
)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION, ContractModel, RuntimeModel
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
)
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.state import GraphRuntimeContext, RuntimeServices, SupervisorV2State

from .factories import conversation_context, request_context

CONTRACT_MODULES = (
    binding,
    capability,
    clarification,
    conversation,
    evaluation,
    evidence,
    execution,
    locators,
    planning,
    request,
    response,
    routing,
    semantic,
    state,
    synthesis,
)

# Spec §3 — independently persisted or externally transported envelopes.
VERSIONED_ENVELOPES = (
    request.RequestContext,
    conversation.ConversationSnapshot,
    semantic.SemanticSnapshot,
    planning.TaskPlan,
    execution.AgentRequest,
    execution.AgentResult,
    evidence.EvidenceUseEnvelope,
    evidence.EvidenceStoreRow,
    binding.BindingAuditRow,
    clarification.ClarificationRequest,
    clarification.ClarificationResolution,
    response.FinalResponse,
)

# Spec §3 — leaf/value types that inherit the enclosing envelope version.
UNVERSIONED_LEAF_TYPES = (
    binding.ScopedDocument,
    binding.BindingRevisionRequirement,
    binding.DocumentBindingSet,
    locators.DocumentLocator,
    locators.SectionLocator,
    planning.TargetUnit,
    planning.TaskSpec,
    planning.DiscoveryPolicy,
    planning.ResearchBudgetView,
    evaluation.CoverageItem,
    evaluation.Coverage,
    evaluation.EvidenceEvaluation,
    evidence.EvidenceRecord,
    evidence.EvidenceUse,
    evidence.Provenance,
    evidence.StoragePolicy,
    capability.CapabilityDescriptor,
    capability.CapabilityRuntimeContext,
    conversation.ConversationContext,
    semantic.SemanticContext,
    synthesis.SynthesisInput,
)


def _all_contract_models() -> set[type[ContractModel]]:
    found: set[type[ContractModel]] = set()
    stack: list[type[ContractModel]] = [ContractModel]
    while stack:
        model = stack.pop()
        for subclass in model.__subclasses__():
            if subclass not in found:
                found.add(subclass)
                stack.append(subclass)
    return found


def _contains_escape_hatch(annotation: object) -> bool:
    if annotation in (Any, dict, typing.Mapping, typing.MutableMapping):
        return True
    origin = get_origin(annotation)
    if origin in (dict, typing.Mapping, typing.MutableMapping):
        return True
    return any(_contains_escape_hatch(argument) for argument in get_args(annotation))


def test_base_enforces_strict_frozen_and_extra_forbid() -> None:
    assert ContractModel.model_config["extra"] == "forbid"
    assert ContractModel.model_config["frozen"] is True
    assert ContractModel.model_config["strict"] is True

    context = request_context()
    with pytest.raises(ValidationError):
        context.original_query = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query="q",
            known_documents=(),
            unexpected="x",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        RequestContext(
            contract_version="2.0",
            request_id=1,  # type: ignore[arg-type]
            thread_id="thread-1",
            original_query="q",
            known_documents=(),
        )


@pytest.mark.parametrize("envelope", VERSIONED_ENVELOPES, ids=lambda c: c.__name__)
def test_versioned_envelope_declares_exactly_2_0(envelope: type[ContractModel]) -> None:
    field = envelope.model_fields["contract_version"]
    assert get_origin(field.annotation) is Literal
    assert get_args(field.annotation) == ("2.0",)
    assert field.is_required()
    assert CONTRACT_VERSION == "2.0"


@pytest.mark.parametrize("leaf", UNVERSIONED_LEAF_TYPES, ids=lambda c: c.__name__)
def test_leaf_types_do_not_repeat_contract_version(leaf: type[ContractModel]) -> None:
    assert "contract_version" not in leaf.model_fields


def test_supervisor_state_is_a_mutable_root_versioned_typeddict() -> None:
    assert typing.is_typeddict(SupervisorV2State)
    assert set(SupervisorV2State.__annotations__) == {
        "contract_version",
        "request",
        "conversation",
        "semantic",
        "bindings",
        "query_analysis",
        "route_decision",
        "execution",
        "clarification",
        "synthesis",
        "final_response",
        "checkpoint_schema_revision",
        "discovery_need",
        "discovery",
        "document_selection_clarification",
        "research_target_selection",
    }
    hints = typing.get_type_hints(SupervisorV2State)
    assert get_args(hints["contract_version"]) == ("2.0",)
    # Nested values stay frozen business contracts; only the aggregate is mutable.
    assert hints["request"] is RequestContext
    assert issubclass(hints["execution"], ContractModel)


def test_runtime_context_is_unversioned_and_never_carries_trusted_identity_in_contracts() -> None:
    assert "contract_version" not in GraphRuntimeContext.model_fields
    assert "contract_version" not in CapabilityRuntimeContext.model_fields
    assert "user_id" not in RequestContext.model_fields
    assert "workspace_ids" not in RequestContext.model_fields
    assert "run_id" not in RequestContext.model_fields


def test_runtime_services_is_mutable_and_unversioned() -> None:
    assert issubclass(RuntimeServices, RuntimeModel)
    assert not issubclass(RuntimeServices, ContractModel)
    services = RuntimeServices()
    assert "contract_version" not in RuntimeServices.model_fields
    services.__dict__  # mutable container, deliberately not frozen


def test_no_contract_field_uses_a_dict_or_any_escape_hatch() -> None:
    offenders: list[str] = []
    for model in sorted(_all_contract_models(), key=lambda c: c.__name__):
        for name, field in model.model_fields.items():
            if _contains_escape_hatch(field.annotation):
                offenders.append(f"{model.__name__}.{name}")
    assert offenders == []


def test_capability_types_are_owned_only_by_capability_module() -> None:
    owned = {"CapabilityDescriptor", "CapabilityRuntimeContext", "CapabilityInput", "CapabilityOutput"}
    for module in CONTRACT_MODULES:
        for name, value in vars(module).items():
            if name in owned and isinstance(value, type) and issubclass(value, ContractModel):
                assert value.__module__ == capability.__name__, (
                    f"{module.__name__} redefines {name}"
                )


def test_contracts_package_imports_no_runtime_frameworks() -> None:
    package_dir = (
        pathlib.Path(__file__).resolve().parents[4]
        / "app"
        / "services"
        / "agents"
        / "v2"
        / "contracts"
    )
    forbidden = re.compile(
        r"^\s*(?:from|import)\s+(fastapi|langgraph|sqlalchemy|sqlmodel|redis|app|psycopg)\b",
        re.M,
    )
    for source in sorted(package_dir.glob("*.py")):
        assert not forbidden.search(source.read_text()), f"impure import in {source.name}"


def test_capability_descriptor_domain_is_a_shared_literal() -> None:
    assert get_origin(CapabilityDescriptor.model_fields["domain"].annotation) is Literal
    assert conversation_context().summary == "Hỏi về nghị định A."
