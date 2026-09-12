"""Evidence identity versus evidence use (spec §15).

``EvidenceRecord`` says what the evidence is — immutable global identity, no
run/task/target usage, no storage policy, no presentation metadata.
``EvidenceUse`` says how one run/task uses that evidence, and
``EvidenceUseEnvelope`` owns the run key. The only graph reference chain is
``EvidenceUseRef → EvidenceUse → EvidenceRecord``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .base import ContractModel, ContractVersion
from .locators import ContentLocator


class DocumentSourceIdentity(ContractModel):
    """Spec §15.1: document evidence identity.

    Workspace membership is deliberately absent; it is resolved from the
    authoritative ``(document_id, document_revision)`` revision record.
    """

    kind: Literal["document"]
    document_id: UUID
    document_revision: str
    locator: ContentLocator


class PeopleSourceIdentity(ContractModel):
    kind: Literal["people"]
    record_id: str


class KnowledgeGraphSourceIdentity(ContractModel):
    kind: Literal["knowledge_graph"]
    entity_or_relation_id: str


class MemorySourceIdentity(ContractModel):
    kind: Literal["memory"]
    memory_id: str


class DerivedSourceIdentity(ContractModel):
    """Spec §15.3: derived evidence carries source lineage only.

    It never copies document/revision/locator coordinates, and validation state
    lives in the Evidence Store rather than on this model.
    """

    kind: Literal["derived"]
    source_evidence_ids: tuple[UUID, ...]


EvidenceSourceIdentity = Annotated[
    DocumentSourceIdentity
    | PeopleSourceIdentity
    | KnowledgeGraphSourceIdentity
    | MemorySourceIdentity
    | DerivedSourceIdentity,
    Field(discriminator="kind"),
]


class Provenance(ContractModel):
    """Spec §15.1: one acquisition may yield multiple minimized records."""

    acquisition_id: UUID
    fetcher: str
    fetched_at: datetime


class EvidenceRecord(ContractModel):
    """Spec §15.1: immutable evidence identity and content."""

    evidence_id: UUID
    source: EvidenceSourceIdentity
    content: str
    content_hash: str
    provenance: Provenance


EvidenceClassification = Literal["normal", "personal", "sensitive_personal"]


class StoragePolicy(ContractModel):
    """Spec §15.3: Evidence Store governance metadata, not semantic evidence.

    ``expires_at`` is the stable deletion deadline selected at insertion time so
    later configuration changes cannot silently extend retention.
    """

    classification: EvidenceClassification
    expires_at: datetime | None


class EvidenceStoreRow(ContractModel):
    """Spec §3/§15.3: the persisted evidence boundary."""

    contract_version: ContractVersion
    record: EvidenceRecord
    storage_policy: StoragePolicy


EvidencePurpose = Literal["discovery", "coverage", "supporting"]


class EvidenceUse(ContractModel):
    """Spec §15.2: how the current run/task uses immutable evidence.

    Run identity stays on the envelope because the Evidence Use Store is keyed by
    run; ``EvidenceUse`` itself stays run-local.
    """

    use_id: UUID
    evidence_id: UUID
    task_id: str
    purpose: EvidencePurpose
    target_id: str | None


class EvidenceUseEnvelope(ContractModel):
    """Spec §3/§15.2: the persisted evidence-use boundary."""

    contract_version: ContractVersion
    run_id: str
    use: EvidenceUse


class EvidenceUseRef(ContractModel):
    """Spec §15.2: the only graph reference to an evidence use."""

    use_id: UUID


# Spec §3: models embedding discriminated-union aliases rebuild explicitly.
EvidenceRecord.model_rebuild()
