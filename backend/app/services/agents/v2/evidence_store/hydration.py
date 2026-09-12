"""Concrete governed hydrator: the Evidence Store admission boundary (T4).

``GovernorEvidenceHydrator`` implements the ``nodes.evaluate.EvidenceHydrator``
Protocol on top of the Phase-1 ``EvidenceGovernor``. It is the production
caller the frozen validators were built for: every admission enforces current
ACL/workspace, expiry, source tombstone/availability, and revision
compatibility through ``EvidenceGovernor.hydrate_use`` (decryption last),
then re-checks the pinned revision deterministically with
``validate_target_use_revision`` for coverage purposes, excludes discovery
uses from synthesis with ``validate_synthesis_use``, and validates derived
lineage with ``validate_derived_evidence_faithfulness``. Denials are skipped
(audited inside the governor) and absent from the admitted set — never
raised, never silent.

Overflow tail persistence goes through the same governed pipeline
(``persist_record`` with a ``DerivedSourceIdentity`` + ``validation_state``
``validated`` and a raised classification floor, then an idempotent
``append_use`` of a supporting use), so the derived record carries source
lineage and the supporting use can never create read coverage (coverage
requires purpose ``coverage``; the builder only counts that purpose).

Flush/commit ownership follows every v2 repository: this adapter mutates and
flushes only — the caller (T6/T7 unit of work) commits the evidence session,
and must do so before the checkpoint that retains the new uses, mirroring
the retention-lease safe ordering. Concrete governor/session wiring is
injected by T6/T7; this adapter class is T4's deliverable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from ..contracts.binding import DocumentBindingSet
from ..contracts.evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceClassification,
    EvidenceRecord,
    EvidenceUse,
    EvidenceUseEnvelope,
    EvidenceUseRef,
    Provenance,
)
from ..contracts.planning import TaskPlan
from ..contracts.state import GraphRuntimeContext
from ..contracts.synthesis import SynthesisRuntimeContext
from ..contracts.validation import (
    ContractValidationError,
    validate_derived_evidence_faithfulness,
    validate_evidence_use,
    validate_evidence_use_resolution,
    validate_synthesis_use,
    validate_target_use_revision,
)
from ..nodes.evaluate import EvidenceHydrator, HydratedEvidence
from ..nodes.synthesize import apply_budget_split
from .governance import (
    DERIVED_VALIDATED,
    EvidenceAccessDenied,
    EvidenceGovernor,
    EvidenceHydrationRequest,
    decrypt_content,
)

__all__ = [
    "HydrationError",
    "GovernorEvidenceHydrator",
    "max_classification",
]


class HydrationError(ValueError):
    """Hydration inputs are unusable; nothing may be admitted or persisted."""


_CLASSIFICATION_RANK: dict[str, int] = {
    "normal": 0,
    "personal": 1,
    "sensitive_personal": 2,
}


def max_classification(
    values: tuple[EvidenceClassification, ...],
) -> EvidenceClassification:
    """Raise the floor to the highest source classification (never downgrade)."""
    best: EvidenceClassification = "normal"
    for value in values:
        if _CLASSIFICATION_RANK[value] > _CLASSIFICATION_RANK[best]:
            best = value
    return best


def _source_label(record: EvidenceRecord) -> str:
    source = record.source
    if isinstance(source, DocumentSourceIdentity):
        return f"document:{str(source.document_id)[:8]}"
    return source.kind


class GovernorEvidenceHydrator(EvidenceHydrator):
    """Production hydration over the governed Evidence Store.

    Constructed with a request-scoped ``EvidenceGovernor`` (session, keyring,
    auditor, clock) by T6/T7; the current plan/bindings/budget arrive per
    call, following the ``execute_node`` precedent of passing bindings
    explicitly.
    """

    def __init__(self, governor: EvidenceGovernor) -> None:
        self._governor = governor

    @property
    def governor(self) -> EvidenceGovernor:
        """The underlying governed store boundary (T6/T7 wiring escape hatch)."""
        return self._governor

    # -- admission ------------------------------------------------------

    async def _admit(
        self,
        ref: EvidenceUseRef,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
        *,
        for_synthesis: bool,
    ) -> HydratedEvidence | None:
        """Admit one use or return ``None`` (denials stay in the audit log)."""
        repository = self._governor.repository
        capability_runtime = runtime.capability_runtime
        envelope = await repository.load_use(ref.use_id)
        if envelope is None:
            return None
        use = envelope.use
        if envelope.run_id != capability_runtime.run_id:
            return None
        try:
            validate_evidence_use(use)
            validate_evidence_use_resolution(use, plan)
            if for_synthesis:
                validate_synthesis_use(use, plan)
            elif use.purpose == "discovery":
                return None
        except ContractValidationError:
            return None
        binding = None
        if use.target_id is not None:
            unit = next(
                (
                    item
                    for item in plan.target_units
                    if item.target_id == use.target_id
                ),
                None,
            )
            if unit is None:
                return None
            binding = next(
                (
                    item
                    for item in bindings.bindings
                    if item.binding_id == unit.binding_id
                ),
                None,
            )
            if binding is None:
                return None
        try:
            content = await self._governor.hydrate_use(
                use_id=use.use_id,
                request=EvidenceHydrationRequest(
                    runtime=capability_runtime,
                    required_binding=binding,
                ),
            )
        except EvidenceAccessDenied:
            return None
        row = await repository.load_record(use.evidence_id)
        if row is None:
            return None
        record = EvidenceRecord(
            evidence_id=row.evidence_id,
            source=row.source,
            content=content,
            content_hash=row.content_hash,
            provenance=row.provenance,
        )
        if use.purpose == "coverage" and use.target_id is not None:
            try:
                validate_target_use_revision(use, plan, bindings, record)
            except ContractValidationError:
                return None
        if isinstance(record.source, DerivedSourceIdentity):
            if row.validation_state != DERIVED_VALIDATED:
                return None
            if not await self._derived_lineage_valid(record):
                return None
        return HydratedEvidence(
            use_id=use.use_id,
            evidence_id=use.evidence_id,
            task_id=use.task_id,
            purpose=use.purpose,
            target_id=use.target_id,
            content=content,
            role=binding.role if binding is not None else None,
            source_label=_source_label(record),
            classification=row.classification,
            locator=(
                record.source.locator
                if isinstance(record.source, DocumentSourceIdentity)
                else None
            ),
            document_revision=(
                record.source.document_revision
                if isinstance(record.source, DocumentSourceIdentity)
                else None
            ),
        )

    async def _derived_lineage_valid(self, record: EvidenceRecord) -> bool:
        """Recursive derived validation: store-validated + readable lineage.

        The governor already walked the ancestors with full ACL/expiry/
        tombstone checks during ``hydrate_use``; this re-asserts the frozen
        validator over really decrypted ancestor records so
        ``validate_derived_evidence_faithfulness`` has a production caller.
        Key/decryption infrastructure failures propagate (loud); unresolvable
        lineage denies.
        """
        source = record.source
        if not isinstance(source, DerivedSourceIdentity):
            return True
        repository = self._governor.repository
        ancestors: list[EvidenceRecord] = []
        for evidence_id in source.source_evidence_ids:
            row = await repository.load_record(evidence_id)
            if row is None:
                return False
            # A derived ancestor must itself be validated; non-derived
            # ancestors carry no validation state (NULL).
            if isinstance(row.source, DerivedSourceIdentity):
                if row.validation_state != DERIVED_VALIDATED:
                    return False
            # Key/decryption infrastructure failures propagate (loud);
            # only unresolvable lineage denies.
            content = decrypt_content(self._governor.keyring, row)
            ancestors.append(
                EvidenceRecord(
                    evidence_id=row.evidence_id,
                    source=row.source,
                    content=content,
                    content_hash=row.content_hash,
                    provenance=row.provenance,
                )
            )
        try:
            validate_derived_evidence_faithfulness(
                record,
                tuple(ancestors),
                faithfulness_validated=True,
            )
        except ContractValidationError:
            return False
        return True

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> tuple[HydratedEvidence, ...]:
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            item = await self._admit(
                ref, runtime, plan, bindings, for_synthesis=False
            )
            if item is not None:
                admitted.append(item)
        return tuple(admitted)

    async def hydrate_for_synthesis(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
        budget: SynthesisRuntimeContext,
    ) -> tuple[HydratedEvidence, ...]:
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            item = await self._admit(
                ref, runtime, plan, bindings, for_synthesis=True
            )
            if item is not None:
                admitted.append(item)
        head, tail = apply_budget_split(tuple(admitted), budget)
        if not tail:
            return tuple(head)
        derived = await self.persist_derived_summary(
            content="\n\n".join(item.content for item in tail),
            source_evidence_ids=tuple(item.evidence_id for item in tail),
            task_id=tail[0].task_id,
            target_id=tail[0].target_id,
            provenance=Provenance(
                acquisition_id=uuid4(),
                fetcher="synthesis.overflow",
                fetched_at=datetime.now(timezone.utc),
            ),
            classification=max_classification(
                tuple(item.classification for item in tail)
            ),
            run_id=runtime.capability_runtime.run_id,
            plan=plan,
            bindings=bindings,
        )
        return tuple(head) + (derived,)

    async def persist_derived_summary(
        self,
        *,
        content: str,
        source_evidence_ids: tuple[UUID, ...],
        task_id: str,
        target_id: str | None,
        provenance: Provenance,
        classification: EvidenceClassification,
        run_id: str,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> HydratedEvidence:
        """Persist the overflow tail as validated derived evidence + use.

        The record is idempotent by content-hash + source identity and the
        use append is idempotent by run/task/evidence/purpose/target, so
        deterministic re-derivation converges. The supporting purpose means
        the new use can never create read coverage. The use is returned
        hydrated directly: the governor denies a target-bound non-document
        use at its revision gate, so callers must not route it back through
        hydration (node flows never do — input refs come from task
        results). Flush-only; the caller commits.
        """
        if not content or not content.strip():
            raise HydrationError("derived overflow content must be non-blank")
        if not source_evidence_ids:
            raise HydrationError("derived overflow requires source lineage")
        evidence_id = await self._governor.persist_record(
            source=DerivedSourceIdentity(
                kind="derived", source_evidence_ids=source_evidence_ids
            ),
            content=content,
            provenance=provenance,
            validation_state=DERIVED_VALIDATED,
            detected_classification=classification,
        )
        use = EvidenceUse(
            use_id=uuid4(),
            evidence_id=evidence_id,
            task_id=task_id,
            purpose="supporting",
            target_id=target_id,
        )
        validate_evidence_use(use)
        validate_evidence_use_resolution(use, plan)
        stored_use_id = await self._governor.repository.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0", run_id=run_id, use=use
            )
        )
        binding = None
        if target_id is not None:
            unit = next(
                (
                    item
                    for item in plan.target_units
                    if item.target_id == target_id
                ),
                None,
            )
            if unit is not None:
                binding = next(
                    (
                        item
                        for item in bindings.bindings
                        if item.binding_id == unit.binding_id
                    ),
                    None,
                )
        return HydratedEvidence(
            use_id=stored_use_id,
            evidence_id=evidence_id,
            task_id=task_id,
            purpose="supporting",
            target_id=target_id,
            content=content,
            role=binding.role if binding is not None else None,
            source_label="derived",
            classification=classification,
            locator=None,
            document_revision=None,
        )
