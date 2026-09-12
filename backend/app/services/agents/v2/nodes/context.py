"""Context and semantic-finalization nodes (Phase 2, Task 1).

Owns the semantic lifecycle helpers ``build_semantic_draft`` and
``finalize_semantic`` plus the node wrappers ``context_node`` and
``semantic_finalizer_node``. The draft is ephemeral and never checkpointed, so
every node rebuilds it deterministically from ``(request, conversation)``;
production NLP richness arrives through the injectable ``draft_builder`` seam
(later tasks), never through a second adapter — the Phase-1 adapters in
``adapters/`` are reused, never duplicated.

Ownership rules honored here: ``RequestContext.original_query`` stays the sole
raw-query owner (the draft/context carry no ``original_query`` field — the
frozen contracts forbid it); known-document attachments are identity-only
candidates and are never auto-bound; prompt-injection content remains data.
"""
from __future__ import annotations

import inspect
import unicodedata
from collections.abc import Awaitable, Callable
from typing import Any, Union

from langgraph.runtime import Runtime

from ..adapters.conversation import DEFAULT_RECENT_TURN_LIMIT
from ..contracts.binding import DocumentBindingSet
from ..contracts.conversation import ConversationContext, ConversationTurn
from ..contracts.request import RequestContext
from ..contracts.semantic import (
    BlockingAmbiguity,
    CoreferenceResolution,
    DocumentReference,
    EntityReference,
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

#: Sync or async seam producing a draft from discourse inputs. Production wiring
#: (later tasks) plugs the real preprocessor here; tests inject fakes.
DraftBuilder = Callable[
    [RequestContext, ConversationContext], Union[SemanticDraft, Awaitable[SemanticDraft]]
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


def _default_draft(
    request: RequestContext, conversation: ConversationContext
) -> SemanticDraft:
    """Deterministic fallback draft: NFC query, discourse carry-over, no refs.

    Document references require preprocessor resolution, so the default carries
    none — attachments in ``request.known_documents`` stay unbound candidates
    (irrelevant-attachment exclusion). Person entities active in the discourse
    window carry over as person refs; an explicit discourse focus carries over
    as a follow-up coreference marker.
    """
    provisional = unicodedata.normalize("NFC", request.original_query).strip()
    if not provisional:
        raise ContextNodeError("request carries no query to contextualize")
    person_refs = tuple(
        EntityReference(ref_id=entity.ref_id, kind="person", label=entity.label)
        for entity in conversation.active_entities
        if entity.kind == "person"
    )
    coreferences: tuple[CoreferenceResolution, ...] = ()
    if conversation.last_focus is not None:
        coreferences = (
            CoreferenceResolution(
                mention=conversation.last_focus.label,
                resolved_ref_id=conversation.last_focus.ref_id,
            ),
        )
    return SemanticDraft(
        provisional_contextualized_query=provisional,
        abbreviations=(),
        coreferences=coreferences,
        document_refs=(),
        person_refs=person_refs,
        section_refs=(),
        preliminary_ambiguities=(),
    )


async def build_semantic_draft(
    request: RequestContext,
    conversation: ConversationContext,
    runtime: GraphRuntimeContext,
    *,
    draft_builder: DraftBuilder | None = None,
) -> SemanticDraft:
    """Build the ephemeral semantic draft (never persisted, never versioned)."""
    if draft_builder is None:
        return _default_draft(request, conversation)
    draft = draft_builder(request, conversation)
    if inspect.isawaitable(draft):
        draft = await draft
    if not isinstance(draft, SemanticDraft):
        raise ContextNodeError(
            f"draft builder returned {type(draft).__name__}, not SemanticDraft"
        )
    return draft


def finalize_blocking_ambiguities(
    preliminary: tuple[BlockingAmbiguity, ...],
    bindings: DocumentBindingSet,
) -> tuple[BlockingAmbiguity, ...]:
    """Keep essential ambiguities not discharged by a binding.

    A binding discharges the ambiguity on its source reference; the binding-ID
    convention ``b_{ref_id}`` is owned by ``adapters/document.py``.
    """
    bound = {binding.binding_id for binding in bindings.bindings}
    return tuple(
        ambiguity
        for ambiguity in preliminary
        if f"b_{ambiguity.ambiguity_id}" not in bound
    )


def _apply_resolution(
    document_refs: tuple[DocumentReference, ...],
    bindings: DocumentBindingSet,
) -> tuple[DocumentReference, ...]:
    """Project the binding outcome onto the canonical document references.

    Fail-closed: a binding for an unresolved/unknown reference (which the real
    Binding Resolver can never produce) raises instead of checkpointing a
    contradictory semantic/binding pair.
    """
    by_binding_id = {binding.binding_id: binding for binding in bindings.bindings}
    known_ref_ids = {reference.ref_id for reference in document_refs}
    for binding_id in by_binding_id:
        if not binding_id.startswith("b_") or binding_id[2:] not in known_ref_ids:
            raise ContextNodeError(
                f"binding {binding_id} has no matching semantic document reference"
            )
    projected: list[DocumentReference] = []
    for reference in document_refs:
        binding = by_binding_id.get(f"b_{reference.ref_id}")
        if binding is None:
            projected.append(reference)
            continue
        if reference.resolution_status != "resolved":
            raise ContextNodeError(
                f"binding b_{reference.ref_id} exists for "
                f"{reference.resolution_status} reference {reference.ref_id}"
            )
        if binding.document_id != reference.resolved_document_id:
            raise ContextNodeError(
                f"binding b_{reference.ref_id} pins document {binding.document_id} "
                f"but the reference resolved to {reference.resolved_document_id}"
            )
        projected.append(reference)
    return tuple(projected)


def finalize_semantic(
    draft: SemanticDraft,
    bindings: DocumentBindingSet,
) -> SemanticContext:
    """Finalize the persisted query meaning from a draft plus bindings."""
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
            bindings,
        ),
    )
    validate_semantic_context(context)
    validate_binding_set(bindings, context)
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
    *,
    draft_builder: DraftBuilder | None = None,
) -> dict:
    """Rebuild the ephemeral draft deterministically and finalize semantics."""
    context = _context_of(runtime)
    draft = await build_semantic_draft(
        state["request"], state["conversation"], context, draft_builder=draft_builder
    )
    return {"semantic": finalize_semantic(draft, state["bindings"])}
