"""EvidenceRecord ORM — Phase 1B (Task 8 amendment).

Maps the ``evidence_records`` table. One row per encrypted evidence
payload plus the immutable evidence identity and the Evidence Store
storage policy:

1. The plaintext never lives in this table — only ``ciphertext`` (AES-256-GCM),
   ``encryption_key_id``, ``nonce`` and ``encryption_algorithm``.
2. ``source`` / ``provenance`` persist the typed frozen-contract JSONB payloads
   (``EvidenceSourceIdentity`` / ``Provenance``) and ``content_hash`` is the
   identity half of the idempotency key (source identity + content hash). The
   ``uq_evidence_record_identity`` unique index over ``(content_hash, source)``
   is that key's arbiter (spec §15.3).
3. ``classification`` / ``expires_at`` are the §15.3 storage policy; the
   classification is computed deterministically at insertion and ``expires_at``
   is the stable deletion deadline selected once.
4. ``revision_id`` is the authoritative document revision the evidence was read
   from. Workspace/ACL membership is deliberately NOT copied onto the record
   (spec §24) — it resolves through that immutable revision row.
5. ``payload_purged_at`` is the GC tombstone for evidence that has aged out;
   the row is retained so the binding audit history stays queryable but the
   underlying ciphertext is logically gone.

The encryption/minimization/hydration policy lives in
``app.services.agents.v2.evidence_store.governance``; this module is the
persistence mapping only.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    contract_version: Mapped[str] = mapped_column(Text, nullable=False)

    # Encrypted payload (never plaintext).
    ciphertext: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    # Identifier of the key used to encrypt this row (keyring key id).
    encryption_key_id: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    # AES-GCM nonce.
    nonce: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    encryption_algorithm: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    # Immutable evidence identity (spec §15.1).
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Storage policy (spec §15.3) — never semantic evidence.
    classification: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Derived-evidence validation state; NULL for non-derived evidence.
    validation_state: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Authoritative document revision for document evidence; NULL for
    # People/KG/Memory/Derived evidence. Workspace resolves through it.
    revision_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id"),
        nullable=True,
    )

    # GC tombstone: when set, the row's ciphertext is logically gone
    # even though the row is retained for binding-audit history.
    payload_purged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        # The record idempotency arbiter (spec §15.3): the repository's
        # ``ON CONFLICT (content_hash, source) DO NOTHING`` infers this unique
        # index, so a retry or concurrent identical write collides instead of
        # duplicating. Mirrors ``uq_evidence_record_identity`` in the migration.
        Index(
            "uq_evidence_record_identity",
            "content_hash",
            "source",
            unique=True,
        ),
    )
