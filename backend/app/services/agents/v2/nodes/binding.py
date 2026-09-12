"""Binding node: the revision-pin point (Phase 2, Task 1).

Resolves the ephemeral semantic draft into immutable revision pins and, before
returning the checkpointable binding update, acquires/refreshes a retention
lease for every pin referenced by the current semantic projection and commits
the lease. The two databases never share a transaction, so the guarantee is
safe ordering (see ``persistence/retention_leases.py``):

- lease commit succeeds, checkpoint fails → harmless orphan lease reclaimed by
  TTL/sweep;
- lease write fails → the node fails and the binding pin is not checkpointed.

Resolution delegates to the request-scoped ``binding_resolver`` service
(``resolve(document_refs, capability_runtime)``; concrete implementation wired
by T6/T7) with the whole trusted ``CapabilityRuntimeContext`` — the resolver
service owns multi-workspace iteration and is never collapsed to one workspace
here. When the service is absent the node fails closed.

The lease set is exactly the resolver's output for THIS turn
(``binding_set.bindings``): every returned pin is validated and leased before
any acquire is issued, so a pin can never be checkpointed without a lease.
Stale merged pins from prior turns are kept in the checkpoint (continuity) but
are NOT lease-refreshed. The lease session is expected to be a dedicated unit
of work (T6 owns the session wiring; committing a shared session would also
commit unrelated pending writes).
"""
from __future__ import annotations

import inspect
from typing import Any

from langgraph.runtime import Runtime

from ..contracts.binding import DocumentBindingSet
from ..contracts.semantic import DocumentReference, SemanticDraft
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from .context import _context_of, build_semantic_draft

__all__ = ["BindingNodeError", "resolve_bindings", "binding_node"]


class BindingNodeError(ValueError):
    """Binding/lease acquisition failed; the pin must not be checkpointed."""


async def resolve_bindings(
    draft: SemanticDraft,
    runtime: GraphRuntimeContext,
) -> DocumentBindingSet:
    """Resolve the draft's document references into a checkpoint binding set.

    Delegates to ``runtime.services.binding_resolver.resolve(
    draft.document_refs, runtime.capability_runtime)``; fails closed when no
    resolver is wired. Always delegates — even with no references — so the
    resolver service observes every resolution.
    """
    resolver = runtime.services.binding_resolver
    if resolver is None:
        raise BindingNodeError(
            "no binding resolver wired on runtime.services; refusing to "
            "synthesize an empty binding set"
        )
    resolved = resolver.resolve(draft.document_refs, runtime.capability_runtime)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    if not isinstance(resolved, DocumentBindingSet):
        raise BindingNodeError(
            f"binding resolver returned {type(resolved).__name__}, "
            "not DocumentBindingSet"
        )
    return resolved


def _merge_binding_sets(
    prior: DocumentBindingSet, new: DocumentBindingSet
) -> DocumentBindingSet:
    """Union pins by binding ID (new wins); drop stale relations.

    A relation survives only when its binding still exists and was not replaced
    by the new set without a fresh relation.
    """
    merged = {binding.binding_id: binding for binding in prior.bindings}
    new_ids = {binding.binding_id for binding in new.bindings}
    merged.update({binding.binding_id: binding for binding in new.bindings})
    relations = {
        relation.binding_id: relation
        for relation in prior.revision_requirement_refs
        if relation.binding_id not in new_ids and relation.binding_id in merged
    }
    relations.update(
        {
            relation.binding_id: relation
            for relation in new.revision_requirement_refs
            if relation.binding_id in merged
        }
    )
    prior_ids = {binding.binding_id for binding in prior.bindings}
    ordered = [
        merged[binding.binding_id]
        for binding in prior.bindings
        if binding.binding_id in merged
    ]
    ordered.extend(
        binding for binding in new.bindings if binding.binding_id not in prior_ids
    )
    return DocumentBindingSet(
        bindings=tuple(ordered),
        revision_requirement_refs=tuple(relations.values()),
    )


def _prune_stale_relations(
    merged: DocumentBindingSet,
    document_refs: tuple[DocumentReference, ...],
) -> DocumentBindingSet:
    """Drop relations whose ref is absent from the current draft (NEW-3).

    Pins are retained for continuity; only the relation is pruned, because the
    frozen aggregate validator resolves every relation's ``ref_id`` against
    the current semantic. Extends the ``_merge_binding_sets`` precedent, which
    already drops relations for replaced binding IDs.
    """
    current_ref_ids = {reference.ref_id for reference in document_refs}
    relations = tuple(
        relation
        for relation in merged.revision_requirement_refs
        if relation.ref_id in current_ref_ids
    )
    if len(relations) == len(merged.revision_requirement_refs):
        return merged
    return DocumentBindingSet(
        bindings=merged.bindings, revision_requirement_refs=relations
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
) -> dict:
    """Pin immutable revisions, commit their retention leases, return bindings."""
    context = _context_of(runtime)
    run_id = context.capability_runtime.run_id
    draft = await build_semantic_draft(
        state["request"], state["conversation"], context
    )
    binding_set = await resolve_bindings(draft, context)
    merged = _prune_stale_relations(
        _merge_binding_sets(state["bindings"], binding_set),
        draft.document_refs,
    )
    new_pins = binding_set.bindings
    if new_pins:
        repo = context.services.retention_leases
        if repo is None:
            raise BindingNodeError(
                "bindings pin revisions but no retention-lease service is wired"
            )
        for binding in new_pins:
            revision = binding.document_revision
            if not isinstance(revision, str) or not revision.strip():
                raise BindingNodeError(
                    f"binding {binding.binding_id} pins a blank revision; "
                    "no lease was issued"
                )
        for binding in new_pins:
            acquired = repo.acquire_or_refresh(run_id, binding.document_revision)
            if inspect.isawaitable(acquired):
                await acquired
        await _commit_lease_session(repo)
    return {"bindings": merged}
