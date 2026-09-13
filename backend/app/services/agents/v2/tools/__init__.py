"""Governed agent-facing tool gateway (Phase 3, Task 2).

The agent/propose boundary is the ONLY adaptive planning boundary and it NEVER
executes a capability. ``AgentToolGateway`` validates proposals and projects
observations ONLY: it never persists a plan, never dispatches work, and never
calls a capability. Execution belongs to the shared scheduler; plan persistence
belongs to the LangGraph graph and its saver.

T4 extends ``observations.py`` and T5 extends ``gateway.py``; both modules stay
additive-friendly and define no replan/discovery policy beyond this boundary.
"""
from .adapters import AgentToolAdapter, AgentToolCall, build_agent_tool_catalog
from .discovery_candidates import (
    CandidateBindingDenied,
    CandidateNotFound,
    DiscoveryCandidateRegistry,
)
from .gateway import (
    AgentToolGateway,
    CapabilityInvocationProposal,
    ProposalRejection,
    ToolProposalOutcome,
    UnplannedCapabilityDispatch,
    require_planned_dispatch,
)
from .observations import (
    AgentToolObservation,
    DocumentReadObservation,
    DocumentSearchObservation,
    KnowledgeGraphObservation,
    NoObservation,
    ObservationProjectionUnavailable,
    ObservationProjector,
    PeopleLookupObservation,
    SectionReadObservation,
    ToolObservationProjection,
)

__all__ = [
    "AgentToolAdapter",
    "AgentToolCall",
    "AgentToolGateway",
    "AgentToolObservation",
    "CandidateBindingDenied",
    "CandidateNotFound",
    "CapabilityInvocationProposal",
    "DiscoveryCandidateRegistry",
    "DocumentReadObservation",
    "DocumentSearchObservation",
    "KnowledgeGraphObservation",
    "NoObservation",
    "ObservationProjectionUnavailable",
    "ObservationProjector",
    "PeopleLookupObservation",
    "ProposalRejection",
    "SectionReadObservation",
    "ToolObservationProjection",
    "ToolProposalOutcome",
    "UnplannedCapabilityDispatch",
    "build_agent_tool_catalog",
    "require_planned_dispatch",
]
