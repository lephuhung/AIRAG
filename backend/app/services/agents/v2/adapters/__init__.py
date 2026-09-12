"""Server-internal legacy → v2 adapter ports (spec §5, §8, §9, §25).

This package is the **server-internal port layer**, not the agent-facing
``tools/`` gateway created in Phase 3. Each module translates one legacy input or
output family into canonical v2 contracts through validation (never through the
v1 ``model_construct()`` bypass):

- ``semantic``      — Phase-1 ``PreprocessingResult`` / persisted semantic payload
                      → ``SemanticDraft`` and, with bindings, the finalized
                      ``SemanticSnapshot``;
- ``document``      — resolved document references → immutable revisions and
                      ``DocumentBindingSet`` via ``persistence.document_views``;
- ``conversation``  — legacy chat rows → ``ConversationContext`` /
                      ``ConversationSnapshot``;
- ``deep_research`` — legacy deep-research task results → typed ``AgentResult``.

No module here redefines or field-extends a frozen v2 contract, and none of them
stores a canonical v2 object back into the legacy shape.
"""
from __future__ import annotations

from .conversation import (
    DEFAULT_RECENT_TURN_LIMIT,
    ConversationAdapterError,
    LegacyChatMessage,
    LegacyExchangeSummary,
    context_from_legacy,
    snapshot_from_legacy,
)
from .deep_research import (
    DeepResearchAdapterError,
    agent_result_from_legacy,
    agent_status_from_legacy,
)
from .document import (
    DocumentAdapterError,
    DocumentBindingResolution,
    ResolvedDocumentBindings,
    resolve_document_binding,
    resolve_document_bindings,
)
from .semantic import (
    PERSISTED_SEMANTIC_VERSION,
    FinalizedSemantics,
    SemanticAdapterError,
    draft_from_persisted_semantic,
    draft_from_preprocessing,
    finalize_semantic_context,
    finalize_semantics,
)

__all__ = [
    "ConversationAdapterError",
    "DEFAULT_RECENT_TURN_LIMIT",
    "DeepResearchAdapterError",
    "DocumentAdapterError",
    "DocumentBindingResolution",
    "FinalizedSemantics",
    "LegacyChatMessage",
    "LegacyExchangeSummary",
    "PERSISTED_SEMANTIC_VERSION",
    "ResolvedDocumentBindings",
    "SemanticAdapterError",
    "agent_result_from_legacy",
    "agent_status_from_legacy",
    "context_from_legacy",
    "draft_from_persisted_semantic",
    "draft_from_preprocessing",
    "finalize_semantic_context",
    "finalize_semantics",
    "resolve_document_binding",
    "resolve_document_bindings",
    "snapshot_from_legacy",
]
