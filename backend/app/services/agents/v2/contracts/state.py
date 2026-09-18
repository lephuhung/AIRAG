"""Checkpoint aggregate versus request-scoped runtime (spec §7, §21).

``SupervisorV2State`` is the mutable LangGraph aggregate whose root
``contract_version`` selects the checkpoint schema; its nested business values
stay strict/frozen ``ContractModel`` values and are replaced rather than mutated.
``GraphRuntimeContext`` and ``RuntimeServices`` are injected per request and are
never checkpointed, so trusted identity and current authorization always replace
historical values on resume.
"""
from __future__ import annotations

from typing import Any, Final, Literal, TypedDict

from .base import ContractModel, ContractVersion, RuntimeModel
from .binding import DocumentBindingSet
from .capability import CapabilityRuntimeContext
from .clarification import ClarificationRequest, DocumentSelectionClarification
from .conversation import ConversationContext
from .evaluation import EvidenceEvaluation
from .execution import AgentResult
from .planning import TaskPlan
from .request import RequestContext
from .response import FinalResponse
from .routing import QueryAnalysis, RouteDecision
from .semantic import SemanticContext
from .synthesis import SynthesisCheckpoint
from ..discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    DiscoveryNeed,
    ResearchTargetSelection,
)

CHECKPOINT_SCHEMA_REVISION: Final[Literal[2]] = 2
CheckpointSchemaRevision = Literal[1, 2]


class RuntimeServices(RuntimeModel):
    """Spec §7: request-scoped runtime service container; never checkpointed.

    The container is runtime-only and mutable so the runtime can inject the
    services a phase needs (the capability registry in Phase 2, retention leases
    in Task 9). It is never serialized into ``SupervisorV2State``.

    ``retention_leases`` is the single owner of revision retention-lease SQL
    (Task 9 / R12): the supervisor, complex subgraph, and every checkpoint
    writer consume ``persistence.retention_leases.RevisionRetentionLeaseRepository``
    and none duplicates lease SQL. It is typed ``Any`` because the contracts
    package deliberately imports no runtime-framework/application module (see
    ``test_contracts_package_imports_no_runtime_frameworks``); the repository is
    a request-scoped service, not a business contract.

    ``semantic_adapter`` exposes ``build_draft(request, conversation)`` and
    ``binding_resolver`` exposes ``resolve(document_refs, capability_runtime)``
    (Phase 2, Task 1 production injection path; concrete implementations are
    wired by T6/T7). Both default to ``None``; the Task 1 nodes fail closed
    with a typed error when the service they need is absent and never fall back
    to a silent default. Typed ``Any`` for the same framework-free reason as
    ``retention_leases``.

    ``capability_registry`` is the single request-scoped capability registry
    (Phase 2, Task 3): the shared ``TaskScheduler`` is the only consumer that
    dispatches through it, and Phase 3 reuses the scheduler rather than
    resolving capabilities itself. ``chat_messages``, ``authorization``, and
    ``evidence_hydrator`` are request-scoped Phase-2 services wired by T6/T7.
    All default to ``None``.

    ``answer_draft_channel`` is the runtime-only ephemeral synthesize → ground
    → finalizer handoff (Phase 2, Task 4): the synthesize node stores the
    validated draft there keyed by run id, the ground node consumes it, and
    the finalizer consumes the grounded result. It is never checkpointed, is
    not a business contract, and defaults to ``None`` (consumers re-derive
    deterministically once from checkpointed state on a miss, e.g. after a
    restart between nodes).

    ``answer_draft_builder`` is the runtime-only request-scoped
    ``StructuredLLMDraftBuilder`` (grounded-LLM synthesis, spec §7.1): the
    bounded-model seam that turns the checkpointed handle manifest plus
    selected evidence into a strictly parsed ``ParsedCandidate``. It is
    constructed once per ingress over the privacy-safe effective-``main``
    provider, is never checkpointed, and defaults to ``None`` — consumers
    fail closed when it is absent. Typed ``Any`` for the same
    framework-free reason as the other slots.

    ``pinned_target_resolver`` is the runtime-only request-scoped
    ``PinnedTargetResolver`` (P0 factual-retrieval live-gate fix): the exact
    same instance the document capabilities resolve scoped targets through.
    The shared ``TaskScheduler`` feeds it deterministically from the
    authoritative checkpointed plan + bindings immediately before any
    dispatch, so fresh (non-resume) turns resolve pinned targets; until fed
    it resolves nothing and the capabilities fail closed. Never
    checkpointed, never fed from ``AgentRequest``/``CapabilityRuntimeContext``.

    ``intent_classifier`` is the runtime-only request-scoped v2 semantic
    ``IntentClassifier`` (Phase 4A, Task 2): the typed v1 intent
    adapter/cache. ``IntentDecision`` values it produces are advisory-only
    semantic facts consumed by the deterministic route policy; they are
    never checkpointed and never carry legacy supervisor control fields
    (``next_agent`` / ``pending_intent`` / task plans). The ingress owner
    constructs exactly one classifier per turn; consumers reuse
    ``services.intent_classifier`` and never construct a classifier per
    call. Defaults to ``None``; consumers fail closed when the service
    is absent.

    ``adaptive_planner`` is the runtime-only request-scoped governed
    ``AdaptivePlanner`` (Phase 5, Task 10): proposal-only initial planning
    behind the existing validate/lease/checkpoint/scheduler boundary. It
    defaults to ``None``; the complex boundary keeps its deterministic
    skill behavior when the service is absent and returns the typed
    unavailable boundary for work no skill covers. Like every other slot
    it is typed ``Any`` for the same framework-free reason and is never
    checkpointed.

    ``adaptive_replanner`` is the runtime-only request-scoped governed
    ``AdaptiveReplanner`` (Phase 5, Task 11): proposal-only bounded
    append-only replanning from minimized evaluator gaps behind the same
    validate/lease/checkpoint/scheduler boundary. It defaults to ``None``;
    the complex boundary keeps its deterministic gap policy when the
    service is absent, and the deterministic proposal still wins when it
    can construct. Typed ``Any`` for the same framework-free reason and
    never checkpointed.

    ``citation_resolver`` is the runtime-only request-scoped
    ``CitationResolver`` (grounded-LLM synthesis, spec §11.1): the
    server-side dependency boundary the ``CitationProjector`` resolves
    public citation metadata through. It is never checkpointed and defaults
    to ``None`` — the synthesis subgraph fails closed
    (``citation_unresolvable``) when it is absent rather than fabricating
    citation identity.

    ``people_lookup`` is the runtime-only request-scoped v1-backed People
    lookup service (the same instance the ``people.lookup`` capability is
    built over). It exposes ``people_display_snapshot()`` — the raw
    sanitized person records plus the consolidated v1 display text captured
    during this run's lookups — so the non-LLM people presentation and the
    streaming adapter can render the v1 people card without a second Mongo
    round trip. The snapshot never enters governed evidence, the
    checkpoint, or the model-facing prompt. Defaults to ``None``; consumers
    fall back to the governed extractive presentation when it is absent.
    """

    retention_leases: Any = None
    semantic_adapter: Any = None
    binding_resolver: Any = None
    capability_registry: Any = None
    chat_messages: Any = None
    authorization: Any = None
    evidence_hydrator: Any = None
    answer_draft_channel: Any = None
    answer_draft_builder: Any = None
    pinned_target_resolver: Any = None
    intent_classifier: Any = None
    adaptive_planner: Any = None
    adaptive_replanner: Any = None
    citation_resolver: Any = None
    people_lookup: Any = None


class GraphRuntimeContext(ContractModel):
    """Spec §7: injected, request-scoped runtime context."""

    capability_runtime: CapabilityRuntimeContext
    services: RuntimeServices


class ExecutionState(ContractModel):
    """Spec §21: the checkpointed execution aggregate.

    Lives with the checkpoint aggregates rather than in the task-contract module so
    that ``TaskPlan``/``AgentResult`` do not need to import each other. There is no
    aggregate ``evidence_refs``; consumers collect use refs from task results.
    """

    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evidence_evaluation: EvidenceEvaluation | None


class SupervisorV2State(TypedDict, total=True):
    """Spec §7: the mutable checkpoint aggregate.

    ``synthesis`` is the additive bounded-synthesis slot (spec §13.1–13.5):
    ``None`` is the canonical idle value and pre-synthesis root-``2.0``
    checkpoints normalize to it through the explicit legacy branch.
    """

    contract_version: ContractVersion
    request: RequestContext
    conversation: ConversationContext
    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis | None
    route_decision: RouteDecision | None
    execution: ExecutionState
    clarification: ClarificationRequest | None
    synthesis: SynthesisCheckpoint | None
    final_response: FinalResponse | None
    checkpoint_schema_revision: CheckpointSchemaRevision
    discovery_need: DiscoveryNeed | None
    discovery: DiscoveryCheckpoint | None
    document_selection_clarification: DocumentSelectionClarification | None
    research_target_selection: ResearchTargetSelection | None
