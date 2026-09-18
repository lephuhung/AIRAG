"""Cycle-free validation errors and primitive helpers (discovery spec §3.1).

This module owns the canonical contract exception identities and the shared
non-blank/uniqueness/version primitives so data-only discovery contracts and
their validators can depend on them without importing ``contracts/validation``.
``contracts/validation`` re-exports the exception classes for compatibility.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import NoReturn

from .base import CONTRACT_VERSION


class ContractValidationError(ValueError):
    """A canonical contract value violates a frozen v2 invariant."""


class IncompatibleCheckpointError(ContractValidationError):
    """A checkpoint/fixture is not a compatible v2 payload and must not be migrated."""


def _fail(message: str) -> NoReturn:
    raise ContractValidationError(message)


def _require_non_blank(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-blank string")


def _require_unique(values: Iterable[object], field: str) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            _fail(f"duplicate {field}: {value!r}")
        seen.add(value)


def _require_contract_version(declared: object, boundary: str) -> None:
    if declared != CONTRACT_VERSION:
        raise IncompatibleCheckpointError(
            f"{boundary} declares contract_version {declared!r}, expected {CONTRACT_VERSION!r}"
        )
