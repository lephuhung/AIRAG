"""Shared bases for the canonical v2 contracts (spec §3).

Every persisted or externally transported v2 business contract derives from
:class:`ContractModel`: extra fields are forbidden, instances are immutable, and
Python-mode validation is strict so a value cannot be silently coerced into a
different business fact.

``RuntimeModel`` is the deliberate counterpart for request-scoped runtime
values. Runtime context is never checkpointed, is not a business contract, and
is mutable because the runtime injects services into it.
"""
from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Spec §3: strict, frozen, extra-forbidding base for business contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RuntimeModel(BaseModel):
    """Base for request-scoped runtime values that are never checkpointed.

    Deliberately mutable and not a ``ContractModel``: runtime/trusted authority
    is injected per request and must never be serialized into checkpoint state.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)


ContractVersion = Literal["2.0"]
"""Spec §3: the only value a versioned v2 envelope may declare."""

CONTRACT_VERSION: Final[ContractVersion] = "2.0"
