"""Document binding port: legacy resolution output → immutable v2 bindings (spec §9).

This is a **server-internal** port, not the agent-facing tool gateway. It takes an
already-resolved :class:`DocumentReference` (the Binding Resolver's input from
semantics) and pins the immutable revision the v2 run will use, producing a
``ScopedDocument`` and the ``DocumentBindingSet`` checkpoint projection.

Immutable revision lookup always goes through
:mod:`app.services.agents.v2.persistence.document_views`:

- an ordinary or explicit-current reference resolves the document's current
  revision identity (the mutable ``Document.current_revision_id`` pointer selects
  the revision, but every artifact fact comes from the immutable revision build
  manifest);
- an explicit pinned reference resolves that exact revision through the
  workspace-scoped guard, so a caller can never pin a revision owned by another
  workspace.

The adapter never reads the mutable document-view artifacts (``markdown_s3_key``,
``chunk_count``, ``raw_chunks_json``). A legacy document with no current revision
cannot be bound and is rejected with ``RevisionNotReady``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ..contracts.binding import (
    BindingRevisionRequirement,
    DocumentBindingSet,
    DocumentRole,
    ScopedDocument,
)
from ..contracts.semantic import (
    CurrentRevisionRequirement,
    DocumentReference,
    PinnedRevisionRequirement,
)
from ..persistence import document_views


class DocumentAdapterError(ValueError):
    """A legacy/resolved reference cannot be translated into a v2 binding."""


@dataclass(frozen=True)
class DocumentBindingResolution:
    """One reference's binding outcome.

    ``binding`` is ``None`` for a reference that is not resolved; the resolved
    reference is always echoed back so the Semantic Finalizer can keep the
    canonical document-reference projection.
    """

    reference: DocumentReference
    binding: ScopedDocument | None


@dataclass(frozen=True)
class ResolvedDocumentBindings:
    """The Binding Resolver's output: canonical references + checkpoint bindings."""

    references: tuple[DocumentReference, ...]
    binding_set: DocumentBindingSet


def _binding_role(
    reference: DocumentReference, default_role: DocumentRole | None
) -> DocumentRole:
    if reference.requested_role is not None:
        return reference.requested_role
    if default_role is not None:
        return default_role
    raise DocumentAdapterError(
        f"document reference {reference.ref_id!r} has no semantic role and no "
        "default role was supplied; target/reference roles are required"
    )


async def resolve_document_binding(
    db: AsyncSession,
    reference: DocumentReference,
    *,
    workspace_id: UUID,
    default_role: DocumentRole | None = None,
) -> DocumentBindingResolution:
    """Pin the immutable revision for one resolved reference.

    An unresolved/ambiguous/not_found/error reference binds nothing (it is owned
    by semantics and clarification), so the original reference is returned
    unchanged.
    """
    if reference.resolution_status != "resolved":
        return DocumentBindingResolution(reference=reference, binding=None)

    document_id = reference.resolved_document_id
    if document_id is None:
        raise DocumentAdapterError(
            f"resolved document reference {reference.ref_id!r} has no canonical "
            "document id"
        )

    role = _binding_role(reference, default_role)
    requirement = reference.revision_requirement
    if isinstance(requirement, PinnedRevisionRequirement):
        try:
            revision_id = UUID(requirement.document_revision)
        except ValueError as exc:
            raise DocumentAdapterError(
                f"pinned revision for reference {reference.ref_id!r} is not a "
                f"revision id: {requirement.document_revision!r}"
            ) from exc
        identity = await document_views.load_revision_identity_for_workspace(
            db, revision_id, workspace_id
        )
    else:
        identity = await document_views.load_current_revision_identity(db, document_id)
        if identity is None:
            raise document_views.RevisionNotReady(
                document_id,
                "document has no current revision (legacy document); reindex it "
                "to publish a revision before v2 binding",
            )

    binding = ScopedDocument(
        binding_id=f"b_{reference.ref_id}",
        document_id=document_id,
        document_revision=str(identity.revision_id),
        role=role,
    )
    return DocumentBindingResolution(reference=reference, binding=binding)


async def resolve_document_bindings(
    db: AsyncSession,
    references: Sequence[DocumentReference],
    *,
    workspace_id: UUID,
    default_role: DocumentRole | None = None,
) -> ResolvedDocumentBindings:
    """Resolve every reference and assemble the checkpoint binding set.

    A revision-requirement relation is recorded only for a binding whose source
    reference carries an explicit ``CurrentRevisionRequirement`` (the sole
    consumer is resume/reuse freshness revalidation), and only when the reference
    actually produced a binding.
    """
    resolutions: list[DocumentBindingResolution] = []
    for reference in references:
        resolutions.append(
            await resolve_document_binding(
                db, reference, workspace_id=workspace_id, default_role=default_role
            )
        )

    bindings = tuple(
        resolution.binding
        for resolution in resolutions
        if resolution.binding is not None
    )
    relations = tuple(
        BindingRevisionRequirement(
            binding_id=resolution.binding.binding_id,
            ref_id=resolution.reference.ref_id,
        )
        for resolution in resolutions
        if resolution.binding is not None
        and isinstance(resolution.reference.revision_requirement, CurrentRevisionRequirement)
    )
    return ResolvedDocumentBindings(
        references=tuple(resolution.reference for resolution in resolutions),
        binding_set=DocumentBindingSet(
            bindings=bindings, revision_requirement_refs=relations
        ),
    )
