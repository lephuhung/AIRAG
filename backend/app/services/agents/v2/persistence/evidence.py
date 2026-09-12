"""Encrypted evidence record / EvidenceUse persistence (spec §15.1/§15.2/§24).

This module owns the two Evidence Store tables:

- :class:`EvidenceRepository` — idempotent insertion of the **already
  encrypted** evidence row (source identity + content hash is the record
  idempotency key) and the idempotent ``EvidenceUse`` append arbitrated by the
  null-safe unique index ``uq_evidence_use_key``.
- :class:`EncryptedEvidenceRecord` — the storage transfer object. It carries no
  plaintext by construction (only ``ciphertext``/``nonce``/``key id``/
  ``algorithm`` plus the identity and storage-policy metadata), so the
  repository cannot accidentally persist a plaintext column.

Like every v2 repository this module only mutates and ``flush`` es; the caller /
unit of work owns the transaction boundary. Minimization, classification,
encryption, and the hydration authorization gate live in
``app.services.agents.v2.evidence_store.governance`` and are applied *before*
anything reaches this module.

Workspace/ACL membership is deliberately not stored here. ``revision_id``
points at the authoritative immutable document revision and
:meth:`EvidenceRepository.resolve_revision_workspace` resolves the workspace
through it (spec §24), never from copied evidence metadata.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.evidence_record import EvidenceRecord as EvidenceRecordRow
from app.models.evidence_use import EvidenceUse as EvidenceUseRow
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.evidence import (
    EvidenceClassification,
    EvidenceSourceIdentity,
    EvidenceUseEnvelope,
    Provenance,
)
from app.services.agents.v2.contracts.validation import validate_evidence_use

#: The persisted ``source`` column is the frozen discriminated union, so it is
#: validated through the same adapter the contract uses on load.
_SOURCE_IDENTITY_ADAPTER: TypeAdapter[EvidenceSourceIdentity] = TypeAdapter(
    EvidenceSourceIdentity
)


class EvidencePersistenceError(Exception):
    """Base class for typed evidence-persistence failures."""


@dataclass(frozen=True)
class EncryptedEvidenceRecord:
    """One ``evidence_records`` row, minus the plaintext.

    Built by the governance layer after minimization/classification/encryption.
    ``source`` and ``provenance`` are the typed frozen contracts; the repository
    serializes them to JSONB and validates them back on load.
    """

    evidence_id: uuid.UUID
    contract_version: str
    content_hash: str
    classification: EvidenceClassification
    expires_at: Optional[datetime]
    revision_id: Optional[uuid.UUID]
    source: EvidenceSourceIdentity
    provenance: Provenance
    validation_state: Optional[str]
    ciphertext: bytes
    encryption_key_id: str
    nonce: bytes
    encryption_algorithm: str
    payload_purged_at: Optional[datetime] = None


@dataclass(frozen=True)
class RevisionWorkspace:
    """The authoritative revision → document → workspace resolution."""

    revision_id: uuid.UUID
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    source_deleted_at: Optional[datetime]


class EvidenceRepository:
    """Persistence for encrypted evidence records and their uses.

    Mutates and ``flush`` es only; it never commits.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -- records -----------------------------------------------------------

    async def find_record_id_by_identity(
        self,
        *,
        source: EvidenceSourceIdentity,
        content_hash: str,
    ) -> Optional[uuid.UUID]:
        """Return the id of the record with this source identity + content hash.

        This is the read half of the §15.3 record idempotency rule ("evidence
        insertion is idempotent by source identity + content hash").
        """
        stmt = select(EvidenceRecordRow.evidence_id).where(
            EvidenceRecordRow.content_hash == content_hash,
            EvidenceRecordRow.source == source.model_dump(mode="json"),
        )
        return await self.session.scalar(stmt)

    async def insert_record(
        self, record: EncryptedEvidenceRecord
    ) -> uuid.UUID:
        """Insert the encrypted record, idempotently.

        Returns the existing row's UUID when a record with the same source
        identity + content hash is already stored, so a retried capability
        execution cannot create an uncontrolled duplicate.
        """
        existing = await self.find_record_id_by_identity(
            source=record.source, content_hash=record.content_hash
        )
        if existing is not None:
            return existing

        stmt = (
            pg_insert(EvidenceRecordRow)
            .values(
                evidence_id=record.evidence_id,
                contract_version=record.contract_version,
                ciphertext=record.ciphertext,
                encryption_key_id=record.encryption_key_id,
                nonce=record.nonce,
                encryption_algorithm=record.encryption_algorithm,
                content_hash=record.content_hash,
                classification=record.classification,
                expires_at=record.expires_at,
                revision_id=record.revision_id,
                source=record.source.model_dump(mode="json"),
                provenance=record.provenance.model_dump(mode="json"),
                validation_state=record.validation_state,
                payload_purged_at=record.payload_purged_at,
            )
            .on_conflict_do_nothing(index_elements=["evidence_id"])
            .returning(EvidenceRecordRow.evidence_id)
        )
        inserted = (await self.session.execute(stmt)).scalar_one_or_none()
        await self.session.flush()
        if inserted is not None:
            return inserted
        # The primary key already exists — return that immutable row's UUID.
        stored = await self.session.scalar(
            select(EvidenceRecordRow.evidence_id).where(
                EvidenceRecordRow.evidence_id == record.evidence_id
            )
        )
        if stored is None:
            raise EvidencePersistenceError(
                f"evidence_records insert for {record.evidence_id} neither "
                "inserted nor resolved an existing row"
            )
        return stored

    async def load_record(
        self, evidence_id: uuid.UUID
    ) -> Optional[EncryptedEvidenceRecord]:
        """Load the encrypted row (never plaintext) or ``None``."""
        row = await self.session.scalar(
            select(EvidenceRecordRow).where(
                EvidenceRecordRow.evidence_id == evidence_id
            )
        )
        if row is None:
            return None
        return self._deserialize(row)

    async def resolve_revision_workspace(
        self, revision_id: uuid.UUID
    ) -> Optional[RevisionWorkspace]:
        """Resolve the workspace that owns ``revision_id`` (spec §24).

        The workspace is read from the authoritative ``documents`` row behind
        the immutable revision, never from evidence metadata.
        """
        resolved = (
            await self.session.execute(
                select(
                    DocumentRevision.revision_id,
                    DocumentRevision.document_id,
                    Document.workspace_id,
                    Document.source_deleted_at,
                )
                .join(Document, Document.id == DocumentRevision.document_id)
                .where(DocumentRevision.revision_id == revision_id)
            )
        ).first()
        if resolved is None:
            return None
        return RevisionWorkspace(
            revision_id=resolved[0],
            document_id=resolved[1],
            workspace_id=resolved[2],
            source_deleted_at=resolved[3],
        )

    # -- uses --------------------------------------------------------------

    async def append_use(self, envelope: EvidenceUseEnvelope) -> uuid.UUID:
        """Append the EvidenceUse, idempotently.

        ``ON CONFLICT (run_id, task_id, evidence_id, purpose, target_id) DO
        NOTHING`` — PostgreSQL infers the null-safe ``uq_evidence_use_key``
        index, including the targetless case. On conflict the existing row's
        UUID is selected by the same five-column null-safe key and returned, so
        a retry reuses the original ``use_id``.
        """
        validate_evidence_use(envelope.use)
        use = envelope.use
        stmt = (
            pg_insert(EvidenceUseRow)
            .values(
                use_id=use.use_id,
                run_id=envelope.run_id,
                evidence_id=use.evidence_id,
                task_id=use.task_id,
                purpose=use.purpose,
                target_id=use.target_id,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    "run_id",
                    "task_id",
                    "evidence_id",
                    "purpose",
                    "target_id",
                ]
            )
            .returning(EvidenceUseRow.use_id)
        )
        inserted = (await self.session.execute(stmt)).scalar_one_or_none()
        await self.session.flush()
        if inserted is not None:
            return inserted

        existing = await self.session.scalar(
            select(EvidenceUseRow.use_id).where(
                EvidenceUseRow.run_id == envelope.run_id,
                EvidenceUseRow.task_id == use.task_id,
                EvidenceUseRow.evidence_id == use.evidence_id,
                EvidenceUseRow.purpose == use.purpose,
                self._target_predicate(use.target_id),
            )
        )
        if existing is None:
            raise EvidencePersistenceError(
                "evidence_uses append for "
                f"run={envelope.run_id!r} task={use.task_id!r} "
                f"evidence={use.evidence_id} purpose={use.purpose!r} "
                f"target={use.target_id!r} neither inserted nor resolved an "
                "existing row"
            )
        return existing

    async def load_use(self, use_id: uuid.UUID) -> Optional[EvidenceUseEnvelope]:
        """Load one persisted use, or ``None``.

        ``evidence_uses`` carries no ``contract_version`` (the approved Task 8
        amendment adds only the run/task/purpose/target key), so the loaded
        envelope declares this code's :data:`CONTRACT_VERSION`.
        """
        row = await self.session.scalar(
            select(EvidenceUseRow).where(EvidenceUseRow.use_id == use_id)
        )
        if row is None:
            return None
        return EvidenceUseEnvelope.model_validate(
            {
                "contract_version": CONTRACT_VERSION,
                "run_id": row.run_id,
                "use": {
                    "use_id": row.use_id,
                    "evidence_id": row.evidence_id,
                    "task_id": row.task_id,
                    "purpose": row.purpose,
                    "target_id": row.target_id,
                },
            },
            strict=False,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _target_predicate(target_id: Optional[str]):
        """Null-safe ``target_id`` predicate (``IS NULL`` for targetless uses)."""
        if target_id is None:
            return EvidenceUseRow.target_id.is_(None)
        return EvidenceUseRow.target_id == target_id

    @staticmethod
    def _deserialize(row: EvidenceRecordRow) -> EncryptedEvidenceRecord:
        return EncryptedEvidenceRecord(
            evidence_id=row.evidence_id,
            contract_version=row.contract_version,
            content_hash=row.content_hash,
            classification=row.classification,
            expires_at=row.expires_at,
            revision_id=row.revision_id,
            source=_SOURCE_IDENTITY_ADAPTER.validate_python(
                row.source, strict=False
            ),
            provenance=Provenance.model_validate(row.provenance, strict=False),
            validation_state=row.validation_state,
            ciphertext=row.ciphertext,
            encryption_key_id=row.encryption_key_id,
            nonce=row.nonce,
            encryption_algorithm=row.encryption_algorithm,
            payload_purged_at=row.payload_purged_at,
        )
