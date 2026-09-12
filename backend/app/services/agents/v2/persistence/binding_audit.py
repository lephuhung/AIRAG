"""Append/read persistence for the v2 binding-audit boundary.

Spec §9 / §24: binding lineage is a persisted **audit** concern — a
discriminated :class:`BindingProvenance` recorded on a
:class:`BindingAuditRow` — not part of the hot-path binding values. Resolving
revision policy from a binding never reads this store; audit/security
investigation does. Only the Binding Resolver writes rows here.

Like every v2 repository this module only mutates and ``flush`` es; the caller
/ unit of work owns the transaction boundary. The table is append-only: there
is no dedup key, because recording the same decision twice is two audit facts.

Versioning (spec §3, §25): an unsupported persisted ``contract_version`` is
rejected (:class:`IncompatibleBindingAuditVersion`) rather than migrated.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.binding_audit import BindingAudit
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import BindingAuditRow


class IncompatibleBindingAuditVersion(Exception):
    """A persisted audit row declares a ``contract_version`` this code does not
    support. The row is rejected, never migrated (spec §25)."""


def _now() -> datetime:
    """Timezone-aware ``now()`` — the project standard."""
    return datetime.now(timezone.utc)


class BindingAuditRepository:
    """Append binding-decision rows and read a thread's audit trail."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def append(
        self,
        row: BindingAuditRow,
        *,
        thread_id: str,
        recorded_at: Optional[datetime] = None,
    ) -> uuid.UUID:
        """Append one audit row and return its allocated ``audit_id``."""
        if row.contract_version != CONTRACT_VERSION:
            raise IncompatibleBindingAuditVersion(
                f"BindingAuditRow declares contract_version "
                f"{row.contract_version!r}; only {CONTRACT_VERSION!r} is "
                "supported and incompatible rows are rejected rather than "
                "migrated"
            )
        audit_id = uuid.uuid4()
        provenance = row.provenance.model_dump(mode="json")
        self.session.add(
            BindingAudit(
                audit_id=audit_id,
                thread_id=thread_id,
                contract_version=row.contract_version,
                provenance_kind=row.provenance.kind,
                provenance=provenance,
                recorded_at=recorded_at or _now(),
            )
        )
        await self.session.flush()
        return audit_id

    async def read_for_thread(self, thread_id: str) -> tuple[BindingAuditRow, ...]:
        """Return the thread's audit rows in recorded order (oldest first)."""
        rows = (
            await self.session.scalars(
                select(BindingAudit)
                .where(BindingAudit.thread_id == thread_id)
                .order_by(BindingAudit.recorded_at, BindingAudit.audit_id)
            )
        ).all()
        return tuple(self._deserialize(row) for row in rows)

    @staticmethod
    def _deserialize(row: BindingAudit) -> BindingAuditRow:
        if row.contract_version != CONTRACT_VERSION:
            raise IncompatibleBindingAuditVersion(
                f"binding_audit row {row.audit_id} declares contract_version "
                f"{row.contract_version!r}; only {CONTRACT_VERSION!r} is "
                "supported and incompatible rows are rejected rather than "
                "migrated"
            )
        return BindingAuditRow.model_validate(
            {
                "contract_version": row.contract_version,
                "provenance": row.provenance,
            },
            strict=False,
        )
