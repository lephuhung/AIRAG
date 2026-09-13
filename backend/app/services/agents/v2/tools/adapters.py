"""Request-scoped framework adapters: the agent-visible tool catalog (Phase 3, Task 2).

The visible catalog is exactly::

    base capability registry
    ∩ current runtime permissions
    ∩ feature flags
    ∩ service availability

The intersection itself is owned by ``build_capability_registry``; this module
re-exposes it as the planner-facing catalog plus a pure translator
(``AgentToolCall`` -> ``CapabilityInvocationProposal``). Adapters never call a
capability: translation produces a proposal that ``AgentToolGateway.propose``
must accept before anything exists in a plan.

The framework-facing schema contains only allowed ``CapabilityInput`` fields.
Runtime authority (workspace IDs, permission flags, deadlines, service objects)
is injected server-side and can never appear as a model-supplied parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import CapabilityRegistry
from ..contracts.base import ContractModel
from ..contracts.capability import (
    AbbreviationResolveInput,
    CapabilityDescriptor,
    CapabilityInput,
    DocumentReadInput,
    DocumentSearchInput,
    KnowledgeGraphInput,
    MemoryLookupInput,
    PeopleLookupInput,
    SectionReadInput,
    WriteInput,
)
from .gateway import CapabilityInvocationProposal

_TOOL_INPUT_TYPES: dict[str, type[ContractModel]] = {
    "people.lookup": PeopleLookupInput,
    "document.search": DocumentSearchInput,
    "document.read": DocumentReadInput,
    "section.read": SectionReadInput,
    "write": WriteInput,
    "knowledge_graph.query": KnowledgeGraphInput,
    "memory.lookup": MemoryLookupInput,
    "abbreviation.resolve": AbbreviationResolveInput,
}


class UnknownAgentTool(LookupError):
    """The planner referenced a tool with no governed input schema."""


def build_agent_tool_catalog(registry: CapabilityRegistry) -> tuple[CapabilityDescriptor, ...]:
    """The planner-facing catalog: permitted capabilities in deterministic order."""
    return registry.catalog()


@dataclass(frozen=True)
class AgentToolCall:
    """One framework tool invocation, interpreted as a proposal (never a dispatch)."""

    capability: str
    objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()


class AgentToolAdapter:
    """Request-scoped translator from framework tool calls to task proposals.

    Holds only the intersected registry view. It exposes the visible catalog,
    the allowed input schema per tool, and a pure ``to_proposal`` translation.
    It has no dispatch path by construction.
    """

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def describe(self) -> tuple[CapabilityDescriptor, ...]:
        """Catalog entries the current runtime permits the planner to propose."""
        return self._registry.catalog()

    def visible_tool_names(self) -> frozenset[str]:
        """Names the current runtime permits the planner to propose."""
        return self._registry.capability_names()

    def is_visible(self, capability: str) -> bool:
        """Whether one capability is proposable under the current runtime."""
        return capability in self._registry.capability_names()

    def input_fields(self, capability: str) -> tuple[str, ...]:
        """Allowed model-supplied fields for one visible tool (input schema only)."""
        model = _TOOL_INPUT_TYPES.get(capability)
        if model is None:
            raise UnknownAgentTool(f"tool {capability!r} has no governed input schema")
        return tuple(model.model_fields)

    def to_proposal(self, call: AgentToolCall) -> CapabilityInvocationProposal:
        """Pure translation: a tool call becomes a proposal for the gateway."""
        return CapabilityInvocationProposal(
            capability=call.capability,
            objective=call.objective,
            input=call.input,
            depends_on=tuple(call.depends_on),
        )
