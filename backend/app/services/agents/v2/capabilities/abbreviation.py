"""Abbreviation resolution capability (Phase 2, Task 2; spec §8.4, §11).

``AbbreviationCapability`` (``abbreviation.resolve``) adapts the deterministic
abbreviation service and returns ``AbbreviationResolveOutput`` with one
``AbbreviationResolution`` per requested token. Unknown tokens resolve to a
``None`` expansion (the token is preserved); blocking ambiguity is reported
through ``BlockingAmbiguity``, never here.

Resolutions carry no personal or document content, so no evidence is persisted
and no coverage is observed. The frozen ``Domain`` literal has no abbreviation
member, so the descriptor is catalogued under the ``document`` domain with the
``resolve`` operation type.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from . import denied_result, dependency_error, error_result
from ..contracts.base import CONTRACT_VERSION
from ..contracts.capability import (
    AbbreviationResolveInput,
    AbbreviationResolveOutput,
    CapabilityDescriptor,
    CapabilityRuntimeContext,
)
from ..contracts.execution import AgentRequest, AgentResult
from ..contracts.semantic import AbbreviationResolution


@runtime_checkable
class AbbreviationResolverService(Protocol):
    """The deterministic abbreviation service (server-side dependency)."""

    def resolve(self, token: str) -> str | None:
        """Return the expansion for ``token`` or ``None`` when unknown."""
        ...


class AbbreviationCapability:
    """Atomic ``abbreviation.resolve`` capability: pure, evidence-free."""

    descriptor = CapabilityDescriptor(
        name="abbreviation.resolve",
        domain="document",
        operation_type="resolve",
        supports_parallel=True,
    )

    def __init__(self, *, service: AbbreviationResolverService) -> None:
        self._service = service

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if "abbreviation.resolve" not in runtime.allowed_capabilities:
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="abbreviation.resolve is not permitted for this request",
            )
        if not isinstance(request.input, AbbreviationResolveInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="abbreviation.resolve requires an abbreviation.resolve input",
            )
        try:
            resolutions = tuple(
                AbbreviationResolution(
                    abbreviation=token, expansion=self._service.resolve(token)
                )
                for token in request.input.tokens
            )
        except Exception as exc:
            return dependency_error(
                request.task_id, capability="abbreviation.resolve", exc=exc
            )
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=AbbreviationResolveOutput(
                kind="abbreviation.resolve", resolutions=resolutions
            ),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )
