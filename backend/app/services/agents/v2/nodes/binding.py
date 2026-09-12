"""Binding node: the revision-pin point (Phase 2, Task 1).

Resolves the ephemeral semantic draft into immutable revision pins and, before
returning the checkpointable binding update, acquires/refreshes a retention
lease for every pinned revision and commits the lease. The two databases never
share a transaction, so the guarantee is safe ordering (see
``persistence/retention_leases.py``):

- lease commit succeeds, checkpoint fails → harmless orphan lease reclaimed by
  TTL/sweep;
- lease write fails → the node fails and the binding pin is not checkpointed.

Resume keeps prior pins (the node never unpins; GC owns expiry) and refreshes
their leases before continuing. A draft with no resolved references binds
nothing and touches no lease service, so non-factual flows run with no lease
wiring. Without a wired resolver or lease service the node fails closed.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Union
from uuid import UUID

from langgraph.runtime import Runtime

from ..adapters.document import ResolvedDocumentBindings
from ..contracts.binding import DocumentBindingSet
from ..contracts.semantic import DocumentReference, SemanticDraft
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from .context import _context_of, build_semantic_draft
from .context import DraftBuilder

__all__ = ["BindingNodeError", "resolve_bindings", "binding_node"]

#: Seam producing checkpoint bindings from resolved references for one
#: workspace. Later tasks close over the request DB session (and default role)
#: here; tests inject fakes. May be sync or async.
BindingsResolver = Callable[
    [tuple[DocumentReference, ...], Any],
    Union[ResolvedDocumentBindings, Awaitable[ResolvedDocumentBindings]],
]


class BindingNodeError(ValueError):
    """Binding/lease acquisition failed; the pin must not be checkpointed."""


async def resolve_bindings(
    draft: SemanticDraft,
    runtime: GraphRuntimeContext,
    *,
    resolve: BindingsResolver | None = None,
) -> ResolvedDocumentBindings:
    """Resolve the draft's document references into checkpoint bindings."""
    references = draft.document_refs
    if not any(reference.resolution_status == "resolved" for reference in references):
        return ResolvedDocumentBindings(
            references=references,
            binding_set=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        )
    if resolve is None:
        raise BindingNodeError(
            "draft carries resolved references but no binding resolver is wired"
        )
    workspaces = runtime.capability_runtime.workspace_ids
    resolved = resolve(references, workspaces[0] if workspaces else None)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    if not isinstance(resolved, ResolvedDocumentBindings):
        raise BindingNodeError(
            f"binding resolver returned {type(resolved).__name__}, "
            "not ResolvedDocumentBindings"
        )
    return resolved


def _merge_binding_sets(
    prior: DocumentBindingSet, new: DocumentBindingSet
) -> DocumentBindingSet:
    """Union pins by binding ID (new wins); relations unioned by binding ID."""
    bindings = {binding.binding_id: binding for binding in prior.bindings}
    for binding in new.bindings:
        bindings[binding.binding_id] = binding
    relations = {
        relation.binding_id: relation for relation in prior.revision_requirement_refs
    }
    for relation in new.revision_requirement_refs:
        relations[relation.binding_id] = relation
    ordered = [bindings[b.binding_id] for b in prior.bindings if b.binding_id in bindings]
    ordered.extend(b for bid, b in bindings.items() if bid not in {x.binding_id for x in prior.bindings})
    return DocumentBindingSet(
        bindings=tuple(ordered),
        revision_requirement_refs=tuple(relations.values()),
    )


async def _commit_lease_session(repo: Any) -> None:
    session = getattr(repo, "session", None)
    commit = getattr(session, "commit", None)
    if commit is None:
        raise BindingNodeError(
            "retention-lease repository exposes no commitable session; "
            "the lease cannot be committed before the checkpointable update"
        )
    result = commit()
    if inspect.isawaitable(result):
        await result


async def binding_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
    *,
    draft_builder: DraftBuilder | None = None,
    bindings_resolver: BindingsResolver | None = None,
) -> dict:
    """Pin immutable revisions, commit their retention leases, return bindings."""
    context = _context_of(runtime)
    run_id = context.capability_runtime.run_id
    draft = await build_semantic_draft(
        state["request"], state["conversation"], context, draft_builder=draft_builder
    )
    resolved = await resolve_bindings(draft, context, resolve=bindings_resolver)
    merged = _merge_binding_sets(state["bindings"], resolved.binding_set)
    if merged.bindings:
        repo = context.services.retention_leases
        if repo is None:
            raise BindingNodeError(
                "bindings pin revisions but no retention-lease service is wired"
            )
        for binding in merged.bindings:
            try:
                revision_id = UUID(binding.document_revision)
            except ValueError as exc:
                raise BindingNodeError(
                    f"binding {binding.binding_id} pins an unparsable revision "
                    f"{binding.document_revision!r}"
                ) from exc
            acquired = repo.acquire_or_refresh(run_id, revision_id)
            if inspect.isawaitable(acquired):
                await acquired
        await _commit_lease_session(repo)
    return {"bindings": merged}
