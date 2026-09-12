"""EvidenceRecord ORM — Phase 1B.

Maps the ``evidence_records`` table. One row per encrypted evidence
payload. The table is the encrypted-at-rest storage layer for every
piece of evidence the v2 pipeline binds to a response — Chroma
chunks, KG facts, source quotes, etc. — so that:

1. The plaintext never lives in this table.
2. The encryption key id, nonce, and algorithm are pinned to the
   ciphertext so the decryption path is fully reproducible.
3. ``payload_purged_at`` is the GC tombstone for evidence that has
   aged out; the row is retained so the binding audit history
   remains queryable but the underlying ciphertext is logically
   gone.

Phase 1D will read these rows via the binding-decision code. The
encryption layer lives in ``app.services.agents.v2.contracts``;
this module deliberately does not expose any plaintext column.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    LargeBinary,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Encrypted payload (never plaintext).
    ciphertext: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    # Identifier of the key used to encrypt this row (e.g. KMS key id).
    encryption_key_id: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    # AES-GCM nonce (or equivalent for the chosen algorithm).
    nonce: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    encryption_algorithm: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    # GC tombstone: when set, the row's ciphertext is logically gone
    # even though the row is retained for binding-audit history.
    payload_purged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
