"""Context and semantic-finalization nodes (Phase 2, Task 1).

Owns the semantic lifecycle helpers ``build_semantic_draft`` and
``finalize_semantic`` plus the node wrappers ``context_node`` and
``semantic_finalizer_node``. The draft is ephemeral and never checkpointed, so
every node rebuilds it through the request-scoped ``semantic_adapter`` service
(concrete implementations wired by T6/T7); when the service is absent the node
fails closed — there is no silent fallback draft.

Ownership rules honored here: ``RequestContext.original_query`` stays the sole
raw-query owner (the draft/context carry no ``original_query`` field — the
frozen contracts forbid it); known-document attachments are identity-only
candidates and are never auto-bound; prompt-injection content remains data.
"""
from __future__ import annotations

import inspect
import unicodedata
from typing import Any

from langgraph.runtime import Runtime

from ..adapters.conversation import DEFAULT_RECENT_TURN_LIMIT
from ..adapters.document import binding_id_for_ref
from ..contracts.binding import DocumentBindingSet
from ..contracts.conversation import ConversationContext, ConversationTurn
from ..contracts.request import RequestContext
from ..contracts.semantic import (
    BlockingAmbiguity,
    DocumentReference,
    SemanticContext,
    SemanticDraft,
)
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import (
    validate_binding_set,
    validate_conversation_context,
    validate_semantic_context,
)

__all__ = [
    "ContextNodeError",
    "build_semantic_draft",
    "finalize_semantic",
    "finalize_blocking_ambiguities",
    "context_node",
    "semantic_finalizer_node",
]


class ContextNodeError(ValueError):
    """Deterministic context/semantic construction failed; the node must fail."""


def _context_of(runtime: Any) -> GraphRuntimeContext:
    """Unwrap the framework ``Runtime`` to the logical ``GraphRuntimeContext``.

    Accepts the injected ``Runtime[GraphRuntimeContext]`` or, for unit tests,
    the context itself.
    """
    if isinstance(runtime, GraphRuntimeContext):
        return runtime
    context = getattr(runtime, "context", None)
    if isinstance(context, GraphRuntimeContext):
        return context
    raise ContextNodeError(
        f"node runtime carries no GraphRuntimeContext (got {type(runtime).__name__})"
    )


def _normalize_validated(draft: SemanticDraft) -> str:
    """Validated normalized form of the draft's contextualized query."""
    normalized = unicodedata.normalize("NFC", draft.provisional_contextualized_query).strip()
    if not normalized:
        raise ContextNodeError("semantic draft carries no contextualized query")
    return normalized


async def build_semantic_draft(
    request: RequestContext,
    conversation: ConversationContext,
    runtime: GraphRuntimeContext,
) -> SemanticDraft:
    """Build the ephemeral semantic draft via the request-scoped adapter.

    Delegates to ``runtime.services.semantic_adapter.build_draft(request,
    conversation)``; fails closed with ``ContextNodeError`` when no adapter is
    wired. Never falls back to a silent default: an unwired adapter must fail
    the node rather than checkpoint semantics with empty references.
    """
    adapter = runtime.services.semantic_adapter
    if adapter is None:
        raise ContextNodeError(
            "no semantic adapter wired on runtime.services; refusing to "
            "synthesize a default draft"
        )
    draft = adapter.build_draft(request, conversation)
    if inspect.isawaitable(draft):
        draft = await draft
    if not isinstance(draft, SemanticDraft):
        raise ContextNodeError(
            f"semantic adapter returned {type(draft).__name__}, not SemanticDraft"
        )
    return draft


def finalize_blocking_ambiguities(
    preliminary: tuple[BlockingAmbiguity, ...],
    bindings: DocumentBindingSet,
) -> tuple[BlockingAmbiguity, ...]:
    """Keep essential ambiguities not discharged by a binding.

    A binding discharges the ambiguity on its source reference; the binding-ID
    convention is owned by ``adapters/document.py`` (``binding_id_for_ref``).
    """
    bound = {binding.binding_id for binding in bindings.bindings}
    return tuple(
        ambiguity
        for ambiguity in preliminary
        if binding_id_for_ref(ambiguity.ambiguity_id) not in bound
    )


def _apply_resolution(
    document_refs: tuple[DocumentReference, ...],
    bindings: DocumentBindingSet,
) -> tuple[DocumentReference, ...]:
    """Project the binding outcome onto the canonical document references.

    A checkpointed binding whose ref is absent from the current draft is a
    stale prior-turn pin: it is SKIPPED here (continuity lives in the merged
    binding set, which this projection never prunes). Fail-closed only for a
    current draft ref: resolved yet unpinned, non-resolved yet pinned, or
    pinning a different document than the resolution.
    """
    by_binding_id = {binding.binding_id: binding for binding in bindings.bindings}
    projected: list[DocumentReference] = []
    for reference in document_refs:
        binding = by_binding_id.get(binding_id_for_ref(reference.ref_id))
        if binding is None:
            if reference.resolution_status == "resolved":
                raise ContextNodeError(
                    f"resolved reference {reference.ref_id} has no pinned binding"
                )
            projected.append(reference)
            continue
        if reference.resolution_status != "resolved":
            raise ContextNodeError(
                f"binding {binding.binding_id} exists for "
                f"{reference.resolution_status} reference {reference.ref_id}"
            )
        if binding.document_id != reference.resolved_document_id:
            raise ContextNodeError(
                f"binding {binding.binding_id} pins document {binding.document_id} "
                f"but the reference resolved to {reference.resolved_document_id}"
            )
        projected.append(reference)
    return tuple(projected)


def finalize_semantic(
    draft: SemanticDraft,
    bindings: DocumentBindingSet,
) -> SemanticContext:
    """Finalize the persisted query meaning from a draft plus bindings.

    Cross-validation runs against the current projection only: stale
    prior-turn pins (and their relations) were validated on their own turn
    and must not fail this turn's finalizer.
    """
    current_ids = {binding_id_for_ref(reference.ref_id) for reference in draft.document_refs}
    current = DocumentBindingSet(
        bindings=tuple(
            binding for binding in bindings.bindings if binding.binding_id in current_ids
        ),
        revision_requirement_refs=tuple(
            relation
            for relation in bindings.revision_requirement_refs
            if relation.binding_id in current_ids
        ),
    )
    context = SemanticContext(
        contextualized_query=draft.provisional_contextualized_query,
        normalized_query=_normalize_validated(draft),
        abbreviations=draft.abbreviations,
        coreferences=draft.coreferences,
        document_refs=_apply_resolution(draft.document_refs, bindings),
        person_refs=draft.person_refs,
        section_refs=draft.section_refs,
        blocking_ambiguities=finalize_blocking_ambiguities(
            draft.preliminary_ambiguities,
            current,
        ),
    )
    validate_semantic_context(context)
    validate_binding_set(current, context)
    return context


async def context_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Carry the current user turn into the discourse window (idempotent)."""
    request = state["request"]
    conversation = state["conversation"]
    turns = list(conversation.recent_turns)
    if not (
        turns
        and turns[-1].role == "user"
        and turns[-1].content == request.original_query
    ):
        turns.append(ConversationTurn(role="user", content=request.original_query))
    updated = ConversationContext(
        summary=conversation.summary,
        active_entities=conversation.active_entities,
        last_focus=conversation.last_focus,
        recent_turns=tuple(turns[-DEFAULT_RECENT_TURN_LIMIT:]),
    )
    validate_conversation_context(updated)
    return {"conversation": updated}


async def semantic_finalizer_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Rebuild the ephemeral draft via the adapter and finalize semantics."""
    context = _context_of(runtime)
    draft = await build_semantic_draft(
        state["request"], state["conversation"], context
    )
    return {"semantic": finalize_semantic(draft, state["bindings"])}
