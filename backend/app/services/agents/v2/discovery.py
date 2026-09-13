"""Discovery candidate identity and binding-handoff semantics (Phase 3, Task 5).

Pure deterministic semantics over the ephemeral
``DiscoveryCandidateRegistry`` index (R33):

- candidate identity is a UUID minted at creation; parallel discovery tasks
  cannot collide and the containing ``AgentResult.task_id`` supplies lineage;
- the discovered revision is pinned exactly: first persisted candidate wins,
  resume never rebinds to a newer revision silently;
- additions create supporting/discovered bindings ONLY, through the Binding
  Resolver (never the tools layer); the planner cannot create new user
  targets autonomously;
- promotion requires an explicit validated policy/user action;
- policy-disabled discovery and ACL denial fail closed;
- current/latest (``CurrentRevisionRequirement``) rebinding is explicit: an
  ordinary pin stays pinned; only an authorized refresh rebinds to latest.

This module creates no binding, performs no authorization itself, and exposes
no revision to the model: it validates handoff requests the Binding Resolver
owns. Frozen contracts are imported, never redefined.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from .contracts.binding import (
    BindingAdditionRequest,
    BindingPromotionRequest,
    DocumentBindingSet,
    DocumentDiscoveryCandidate,
)
from .contracts.planning import DiscoveryPolicy
from .tools.discovery_candidates import (
    CandidateNotFound,
    DiscoveryCandidateRegistry,
    candidate_addition_request,
)

__all__ = [
    "DiscoveryDenied",
    "DiscoveryDisabled",
    "PromotionRequiresApproval",
    "mint_candidate",
    "pin_for_rebinding",
    "request_addition",
    "require_candidate_visible",
    "require_discovery_enabled",
    "resolve_pin",
    "validate_promotion",
]

_ALLOWED_ADDITION_ROLES = ("discovered", "supporting")
_ALLOWED_PROMOTION_ROLES = ("reference", "supporting")


class DiscoveryDisabled(ValueError):
    """Discovery is disabled by policy for the requested role."""


class DiscoveryDenied(ValueError):
    """The candidate's document is not visible under current authorization."""


class PromotionRequiresApproval(ValueError):
    """Promotion needs an explicit validated policy/user action."""


def mint_candidate(
    document_id: UUID, document_revision: str
) -> DocumentDiscoveryCandidate:
    """Mint one discovery candidate with a globally unique UUID identity."""
    if not isinstance(document_id, UUID):
        raise TypeError(
            f"candidate document_id must be a UUID, got {type(document_id).__name__}"
        )
    if not isinstance(document_revision, str) or not document_revision.strip():
        raise DiscoveryDenied(
            "a discovery candidate requires a non-blank pinned revision"
        )
    return DocumentDiscoveryCandidate(
        candidate_id=uuid4(),
        document_id=document_id,
        document_revision=document_revision,
    )


def require_discovery_enabled(
    policy: DiscoveryPolicy, requested_role: str
) -> DiscoveryPolicy:
    """Fail closed when the policy authorizes no discovery for the role."""
    if requested_role not in _ALLOWED_ADDITION_ROLES:
        # The planner cannot create new user targets autonomously: only the
        # supporting/discovered roles may ever come from discovery.
        raise DiscoveryDisabled(
            f"discovery cannot create role {requested_role!r}: additions create "
            "supporting/discovered bindings only"
        )
    if requested_role == "discovered" and not policy.allow_reference_discovery:
        raise DiscoveryDisabled(
            "discovery is disabled by policy (allow_reference_discovery=False)"
        )
    if requested_role == "supporting" and not policy.allow_supporting_discovery:
        raise DiscoveryDisabled(
            "discovery is disabled by policy (allow_supporting_discovery=False)"
        )
    if policy.max_discovered_documents < 1:
        raise DiscoveryDisabled(
            "discovery is disabled by policy (max_discovered_documents < 1)"
        )
    return policy


def request_addition(
    registry: DiscoveryCandidateRegistry,
    candidate_id: UUID,
    requested_role: str,
    *,
    policy: DiscoveryPolicy,
) -> BindingAdditionRequest:
    """Validated handoff: indexed candidate -> request for the Binding Resolver.

    Policy first (disabled discovery fails before any lookup), then the pure
    registry handoff. Creates no binding and performs no authorization: the
    Binding Resolver revalidates current ACL/scope against the server-side
    candidate and pins the immutable discovered revision itself.
    """
    require_discovery_enabled(policy, requested_role)
    return candidate_addition_request(registry, candidate_id, requested_role)


def require_candidate_visible(
    candidate: DocumentDiscoveryCandidate,
    visible_document_ids: frozenset[UUID] | set[UUID],
) -> DocumentDiscoveryCandidate:
    """ACL gate: the candidate's document must be visible right now.

    ``visible_document_ids`` is the server-side set resolved under CURRENT
    runtime authorization (never model-supplied). An invisible document fails
    closed without leaking metadata.
    """
    if candidate.document_id not in visible_document_ids:
        raise DiscoveryDenied(
            "the discovered document is not visible under current authorization"
        )
    return candidate


def resolve_pin(
    registry: DiscoveryCandidateRegistry, candidate_id: UUID
) -> tuple[UUID, str]:
    """Exact pinned identity for one candidate: ``(document_id, revision)``.

    The pin is immutable: the registry keeps the first persisted candidate per
    identity, so a newer published revision never moves an existing pin. The
    revision stays server-side; model observations carry the UUID only.
    """
    for candidate in registry.candidates():
        if candidate.candidate_id == candidate_id:
            return candidate.document_id, candidate.document_revision
    raise CandidateNotFound(f"unknown discovery candidate {candidate_id}")


def validate_promotion(
    bindings: DocumentBindingSet,
    request: BindingPromotionRequest,
    *,
    approved: bool,
    reason: str,
) -> BindingPromotionRequest:
    """Validate a promotion proposal: explicit user/policy action is required.

    The planner cannot promote autonomously: ``approved`` must be an explicit
    ``True`` (validated policy/user action) with a non-blank reason, the
    source binding must exist, and the target role stays within
    reference/supporting (never a new user target).
    """
    if request.requested_role not in _ALLOWED_PROMOTION_ROLES:
        raise PromotionRequiresApproval(
            f"promotion cannot create role {request.requested_role!r}: "
            "promotion yields reference/supporting bindings only"
        )
    if approved is not True:
        raise PromotionRequiresApproval(
            "promotion requires an explicit validated policy/user action; "
            "the planner cannot promote autonomously"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise PromotionRequiresApproval(
            "promotion requires a non-blank reason recording the explicit action"
        )
    known = {binding.binding_id for binding in bindings.bindings}
    if request.source_binding_id not in known:
        raise PromotionRequiresApproval(
            f"promotion references unknown binding {request.source_binding_id!r}"
        )
    return request


def pin_for_rebinding(
    *,
    pinned_revision: str,
    latest_revision: str,
    is_current_required: bool,
    user_authorized_refresh: bool,
) -> str:
    """Revision to use on resume/reuse for one binding.

    Ordinary pins resolve once and stay pinned even when a newer revision
    publishes. Explicit current/latest semantics rebind to latest ONLY with
    user authorization; without it the existing pin stands. Never mutates a
    binding: the Binding Resolver creates the rebound binding.
    """
    if (
        is_current_required
        and user_authorized_refresh
        and isinstance(latest_revision, str)
        and latest_revision.strip()
    ):
        return latest_revision
    return pinned_revision
