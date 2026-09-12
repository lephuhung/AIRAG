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
from ..contracts.execution import AgentRequest, AgentResult

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
