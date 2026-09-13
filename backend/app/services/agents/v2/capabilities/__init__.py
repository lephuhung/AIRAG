"""Canonical capability protocol and the ACL-filtered capability registry (spec §11, §13, §16).

This package is the single owner of the ``Capability`` protocol. Phase 2 imports
and re-exports it and must never define a second one. The frozen
``CapabilityDescriptor``, ``CapabilityInput``, ``CapabilityOutput``, and the
runtime-only ``CapabilityRuntimeContext`` are imported verbatim from
``contracts/capability.py`` and are never redefined or field-extended here.

Authorization boundary: the request-scoped registry is

    base capabilities ∩ permissions ∩ feature flags ∩ service availability

Unauthorized capabilities are absent from the planner catalog (``catalog``) and
remain denied at execution (``get``). Workspace IDs and permission flags come
only from the trusted ``CapabilityRuntimeContext``; no planner/model input can
supply them.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityInput,
    CapabilityOutput,
    CapabilityRuntimeContext,
)
from ..contracts.evidence import (
    EvidencePurpose,
    EvidenceSourceIdentity,
    EvidenceUseRef,
    Provenance,
)
from ..contracts.execution import (
    AgentError,
    AgentErrorCode,
    AgentRequest,
    AgentResult,
)
from ..contracts.binding import ScopedDocument
from ..contracts.evaluation import CoverageOutcome
from ..contracts.locators import ContentLocator
from ..contracts.planning import TargetUnit

__all__ = [
    "Capability",
    "CapabilityDescriptor",
    "CapabilityInput",
    "CapabilityOutput",
    "CapabilityRuntimeContext",
    "CapabilityRegistration",
    "CapabilityRegistry",
    "CapabilityRegistryError",
    "CapabilityDenied",
    "CapabilityUnavailable",
    "CapabilityNotRegistered",
    "build_capability_registry",
    "EvidenceBuilder",
    "PinnedTargetResolver",
    "ResolvedTarget",
    "LocatedContent",
    "denied_result",
    "error_result",
    "dependency_error",
    "PeopleCapability",
    "PeopleLookupService",
    "DocumentSearchCapability",
    "DocumentSearchService",
    "DocumentReadCapability",
    "DocumentContentReader",
    "DocumentRetrieveCapability",
    "DocumentRetrievalService",
    "RevisionRetrievedChunk",
    "SectionReadCapability",
    "SectionContentReader",
    "KnowledgeGraphCapability",
    "KnowledgeGraphClient",
    "MemoryCapability",
    "MemoryStore",
    "AbbreviationCapability",
    "AbbreviationResolverService",
]


@runtime_checkable
class Capability(Protocol):
    """Spec §11: one executable capability.

    ``descriptor`` is the frozen planner catalog entry. ``execute`` receives the
    selected ``AgentRequest`` plus the current trusted runtime authorization; a
    capability never reads authorization from the request or from checkpoint
    state.
    """

    descriptor: CapabilityDescriptor

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult: ...


class CapabilityRegistryError(Exception):
    """Base class for capability-registry failures."""


class CapabilityNotRegistered(CapabilityRegistryError, LookupError):
    """No capability with that name exists in the deployment."""


class CapabilityDenied(CapabilityRegistryError, PermissionError):
    """The capability exists but current runtime permissions exclude it."""


class CapabilityUnavailable(CapabilityRegistryError):
    """The capability exists and is permitted but a deployment gate excludes it
    (its feature flag is off or the service it depends on is not available)."""


#: Why a registered capability was excluded from the request-scoped registry.
ExclusionReason = Literal["permission", "feature_flag", "service"]


@dataclass(frozen=True)
class CapabilityRegistration:
    """One base-capability registration with its deployment gates.

    ``feature_flag`` names the flag that must be active; ``service`` names the
    backend service that must be available. ``None`` means the capability has no
    gate of that kind.
    """

    capability: Capability
    feature_flag: str | None = None
    service: str | None = None


def _is_permitted(
    descriptor: CapabilityDescriptor, runtime: CapabilityRuntimeContext
) -> bool:
    """Current trusted permission intersection for one capability."""
    if descriptor.name not in runtime.allowed_capabilities:
        return False
    if descriptor.domain == "people" and not runtime.can_read_people:
        return False
    return True


def _is_available(
    registration: CapabilityRegistration,
    active_feature_flags: frozenset[str],
    available_services: frozenset[str],
) -> ExclusionReason | None:
    """Deployment gate intersection for one registration."""
    if (
        registration.feature_flag is not None
        and registration.feature_flag not in active_feature_flags
    ):
        return "feature_flag"
    if registration.service is not None and registration.service not in available_services:
        return "service"
    return None


class CapabilityRegistry:
    """A request-scoped view over the capabilities the current runtime may use.

    Instances are produced by :func:`build_capability_registry`; constructing one
    directly would bypass the permission/availability intersection.
    """

    def __init__(
        self,
        capabilities: Mapping[str, Capability],
        *,
        runtime: CapabilityRuntimeContext,
        excluded: Mapping[str, ExclusionReason] | None = None,
    ) -> None:
        self._runtime = runtime
        self._capabilities = dict(capabilities)
        self._excluded = dict(excluded or {})

    @property
    def runtime(self) -> CapabilityRuntimeContext:
        """The trusted runtime authorization this registry was intersected with."""
        return self._runtime

    def capability_names(self) -> frozenset[str]:
        """Names the current runtime may use."""
        return frozenset(self._capabilities)

    def catalog(self) -> tuple[CapabilityDescriptor, ...]:
        """The planner catalog: permitted capabilities in deterministic order."""
        return tuple(
            self._capabilities[name].descriptor
            for name in sorted(self._capabilities)
        )

    def get(self, name: str) -> Capability:
        """Resolve a permitted capability, or raise a typed denial.

        :raises CapabilityDenied: registered but outside runtime permission.
        :raises CapabilityUnavailable: registered but gated by flag/service.
        :raises CapabilityNotRegistered: not part of the deployment at all.
        """
        capability = self._capabilities.get(name)
        if capability is not None:
            return capability
        reason = self._excluded.get(name)
        if reason == "permission":
            raise CapabilityDenied(
                f"capability {name!r} is not permitted for this request"
            )
        if reason in ("feature_flag", "service"):
            raise CapabilityUnavailable(
                f"capability {name!r} is unavailable ({reason})"
            )
        raise CapabilityNotRegistered(f"capability {name!r} is not registered")


def build_capability_registry(
    registrations: Iterable[CapabilityRegistration],
    runtime: CapabilityRuntimeContext,
    *,
    active_feature_flags: frozenset[str] = frozenset(),
    available_services: frozenset[str] = frozenset(),
) -> CapabilityRegistry:
    """Intersect the base capabilities with the current runtime authorization.

    ``runtime.allowed_capabilities`` (plus ``can_read_people`` for the ``people``
    domain) is the permission set; ``active_feature_flags`` and
    ``available_services`` are the deployment gates. A capability excluded for
    any reason never appears in the catalog and never resolves via ``get``.
    """
    permitted: dict[str, Capability] = {}
    excluded: dict[str, ExclusionReason] = {}
    for registration in registrations:
        name = registration.capability.descriptor.name
        unavailable = _is_available(
            registration, active_feature_flags, available_services
        )
        if unavailable is not None:
            excluded[name] = unavailable
            continue
        if not _is_permitted(registration.capability.descriptor, runtime):
            excluded[name] = "permission"
            continue
        permitted[name] = registration.capability
    return CapabilityRegistry(permitted, runtime=runtime, excluded=excluded)


# ---------------------------------------------------------------------------
# Phase 2 request-scoped capability dependencies (Task 2)
# ---------------------------------------------------------------------------


@runtime_checkable
class EvidenceBuilder(Protocol):
    """Request-scoped evidence sink owned by the Evidence Store boundary.

    Capability instances are constructed with an ``EvidenceBuilder`` (wired by
    T6/T7) and call it to persist minimized content plus mint the governed
    ``EvidenceUse`` the returned ``AgentResult`` references. The builder owns
    ``EvidenceRecord``/``EvidenceUse`` persistence; the capability only ever
    sees the resulting ``EvidenceUseRef``.
    """

    async def persist_use(
        self,
        *,
        source: EvidenceSourceIdentity,
        content: str,
        provenance: Provenance,
        task_id: str,
        purpose: EvidencePurpose,
        target_id: str | None,
    ) -> EvidenceUseRef: ...


@dataclass(frozen=True)
class ResolvedTarget:
    """One planned target with its authoritative coordinate and pinned identity.

    ``target_unit`` is the plan's authoritative requirement (its
    ``requested_locator`` is the canonical coordinate read coverage is
    measured against); ``document`` is the pinned, currently-authorized
    revision identity. The resolver (fed the checkpointed plan/bindings by
    T6/T7) is the only producer of this pair.
    """

    target_unit: TargetUnit
    document: ScopedDocument


@dataclass(frozen=True)
class LocatedContent:
    """A typed content-reader outcome for one planned locator.

    ``outcome`` distinguishes ``read`` / ``missing`` / ``unreadable`` /
    ``truncated`` (spec §14); ``observed_locator`` is the coordinate the
    dependency actually read (``None`` when nothing was observed); ``content``
    carries text only on paths where the capability persists evidence.
    """

    outcome: CoverageOutcome
    observed_locator: ContentLocator | None
    content: str | None


@runtime_checkable
class PinnedTargetResolver(Protocol):
    """Request-scoped plan/binding resolver for read capabilities.

    The resolver is constructed with (fed) the authoritative checkpointed
    plan/bindings by T6/T7 and maps a planned ``target_id`` to its
    ``ResolvedTarget`` (planned ``TargetUnit`` + pinned currently-authorized
    ``ScopedDocument``). ``None`` means the target is unknown, unpinned, or no
    longer authorized, and the capability fails closed. The resolver never
    reads supervisor/graph state and is never fed from ``AgentRequest`` or
    ``CapabilityRuntimeContext``.
    """

    def resolve(self, target_id: str) -> ResolvedTarget | None: ...


def denied_result(
    task_id: str, *, code: AgentErrorCode, message: str
) -> AgentResult:
    """A typed denial: the capability exists but may not run here."""
    from ..contracts.base import CONTRACT_VERSION

    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status="denied",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(code=code, message=message, retryable=False),
    )


def dependency_error(task_id: str, *, capability: str, exc: BaseException) -> AgentResult:
    """Map an escaped dependency failure onto a typed execution error.

    Dependency exceptions must never escape ``execute``: a timeout maps to
    ``TIMEOUT`` (retryable), a connection/OS failure to
    ``DEPENDENCY_UNAVAILABLE`` (retryable), and anything else to
    ``INTERNAL_ERROR`` (not retryable). Only the exception class name enters
    the message so dependency internals never leak into the checkpoint.
    ``asyncio.CancelledError`` is a ``BaseException`` and is never caught by
    callers of this helper — cancellation keeps propagating to the T3
    dispatch boundary.
    """
    from ..contracts.base import CONTRACT_VERSION

    if isinstance(exc, TimeoutError):
        code: AgentErrorCode = "TIMEOUT"
        retryable = True
    elif isinstance(exc, (ConnectionError, OSError)):
        code = "DEPENDENCY_UNAVAILABLE"
        retryable = True
    else:
        code = "INTERNAL_ERROR"
        retryable = False
    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(
            code=code,
            message=f"{capability} dependency failed ({type(exc).__name__})",
            retryable=retryable,
        ),
    )


def error_result(
    task_id: str, *, code: AgentErrorCode, message: str
) -> AgentResult:
    """A typed execution error: the capability ran but could not produce facts."""
    from ..contracts.base import CONTRACT_VERSION

    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(code=code, message=message, retryable=False),
    )


# Re-exported after the protocols above so capability modules can import the
# request-scoped dependencies from this package without a partial-init cycle.
from .abbreviation import AbbreviationCapability, AbbreviationResolverService
from .document import (
    DocumentContentReader,
    DocumentReadCapability,
    DocumentRetrievalService,
    DocumentRetrieveCapability,
    DocumentSearchCapability,
    DocumentSearchService,
    RevisionRetrievedChunk,
)
from .knowledge_graph import KnowledgeGraphCapability, KnowledgeGraphClient
from .memory import MemoryCapability, MemoryStore
from .people import PeopleCapability, PeopleLookupService
from .section import SectionContentReader, SectionReadCapability
