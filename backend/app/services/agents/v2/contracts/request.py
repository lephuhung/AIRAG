"""The persisted request boundary (spec §8.1).

``RequestContext`` owns the persisted request identity and the raw query. Trusted
``user_id``, run lineage, ACL, and deadlines belong to the runtime
``CapabilityRuntimeContext`` and never appear here. Known resources provide
identity only: an attachment is a contextual candidate until semantics binds it.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from .base import ContractModel, ContractVersion


class KnownDocumentResource(ContractModel):
    """Spec §8.1: identity of a resource known at ingress, with no role/revision."""

    resource_id: str
    document_id: UUID
    source: Literal["attachment", "ui_selection", "conversation", "api_explicit"]


class RequestContext(ContractModel):
    """Spec §8.1: the persisted request boundary."""

    contract_version: ContractVersion
    request_id: str
    thread_id: str
    original_query: str
    known_documents: tuple[KnownDocumentResource, ...]
