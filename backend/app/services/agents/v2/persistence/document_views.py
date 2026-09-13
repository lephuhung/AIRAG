"""Revision-qualified artifact identity, revision-selected retrieval scope, and
the current-revision document viewer adapter — Phase 1C Task 5.

This module is the **single authoritative definition** of the v2
revision-qualified artifact identity scheme (controller ruling R9): object
keys, vector ids, the embedding-model/dimension-qualified collection
namespace, and the KG revision scope. Task 9's
``delete_revision_artifacts`` / ``delete_revision`` consume these helpers and
must never redefine them.

Identity rules
--------------
- **Object artifacts** are keyed per revision, never per document:
  ``revision_markdown_key`` / ``revision_structure_key``. A reindex uploads a
  *new* object; it never overwrites a published revision's markdown.
- **Vector ids** are ``rev_<revision_id>_chunk_<ordinal>`` — the revision and
  the stable chunk ordinal are both recoverable from the id alone
  (:func:`parse_revision_vector_id`), so a locator can be reconstructed from a
  stored id without consulting mutable document state.
- **The vector collection namespace** is
  ``ws_<workspace>_embed_<model_hash>_d<dimension>``
  (:func:`embedding_namespace`). The model hash and dimension are baked into
  the name, so a build under a different embedding model/dimension writes a
  *different* collection instead of deleting/recreating the one holding
  published vectors. A dimension mismatch on an existing collection raises
  :class:`EmbeddingMigrationRequired` (see ``vector_store.add_documents``).
- **KG provenance** is scoped per revision (:func:`revision_kg_scope`): a fact
  produced by R1 is queryable only for R1.

Retrieval policy
----------------
``Document.current_revision_id`` selects the revision. A document with **no**
current revision is a legacy v1 document: it stays on the unchanged v1
adapter (``Document.markdown_s3_key`` + ``doc_<document>_chunk_<index>``
vector ids). The v2 path is *explicit* — a caller that names revisions (or
calls :func:`resolve_retrieval_revisions`) requires a published,
artifact-verified revision and raises :class:`RevisionNotReady`
(``REVISION_NOT_READY``) instead of falling back to legacy chunks/KG.

Historical revisions always resolve their embedding namespace/model/dimension/
vector-artifact version from that revision's own build manifest
(:func:`load_revision_identity`), never from current configuration.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentImage
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
from app.models.document_revision_chunk import DocumentRevisionChunk


# ---------------------------------------------------------------------------
# Public constants / exceptions
# ---------------------------------------------------------------------------


#: Machine-readable code carried by :class:`RevisionNotReady`. The v2 port
#: surfaces this verbatim (API detail / tool error) instead of a generic 404.
REVISION_NOT_READY: str = "REVISION_NOT_READY"

#: Schema version of the persisted vector artifact. Stamped on every build
#: manifest so a consumer can detect an upgrade without guessing from config.
VECTOR_ARTIFACT_VERSION: str = "v1"

#: Schema version of the revision structure artifact (chunk payloads).
STRUCTURE_ARTIFACT_VERSION: str = "v1"

#: The only revision state a retrieval caller may select. ``published``
#: implies ``verified`` (the publish CAS is gated on it).
RETRIEVAL_READY_STATUS: str = "published"


class RevisionNotReady(Exception):
    """The v2 port was asked to use a revision that is not published and
    artifact-verified.

    Carries ``code = "REVISION_NOT_READY"``, the ``document_id``, and (when
    known) the ``revision_id`` so callers can render an actionable error.
    Never raised by the v1 adapter path.
    """

    code: str = REVISION_NOT_READY

    def __init__(
        self,
        document_id: uuid.UUID | str,
        reason: str,
        revision_id: uuid.UUID | str | None = None,
    ) -> None:
        self.document_id = str(document_id)
        self.revision_id = str(revision_id) if revision_id is not None else None
        self.reason = reason
        super().__init__(
            f"{REVISION_NOT_READY}: document {self.document_id} "
            f"(revision {self.revision_id or '-'}): {reason}"
        )


class EmbeddingMigrationRequired(Exception):
    """An existing collection was written with a different embedding dimension.

    Raised instead of deleting/recreating the collection: the collection may
    hold published revisions' vectors, and a dimension change must migrate to a
    new :func:`embedding_namespace` (which the revision allocator does
    automatically) rather than destroy them.
    """


class MixedRevisionMerge(Exception):
    """A merge was asked to combine chunks from revisions outside the selected
    scope.

    Revision-selected retrieval is exact: a vector or BM25 hit whose
    ``revision_id`` is not in the selected set must never enter the merged
    result, because that would let one revision's fact leak into another
    revision's answer.
    """


# ---------------------------------------------------------------------------
# Object keys
# ---------------------------------------------------------------------------


def revision_prefix(
    workspace_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
    revision_id: uuid.UUID | str,
) -> str:
    """Object-key prefix owned by one revision.

    ``kb_<workspace>/revisions/<document>/<revision>`` — the revision id is
    part of the key, so every artifact write is copy-on-write by construction.
    """
    return (
        f"kb_{workspace_id}/revisions/{document_id}/{revision_id}"
    )


def revision_markdown_key(
    workspace_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
    revision_id: uuid.UUID | str,
) -> str:
    """Revision-qualified markdown object key."""
    return f"{revision_prefix(workspace_id, document_id, revision_id)}/document.md"


def revision_structure_key(
    workspace_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
    revision_id: uuid.UUID | str,
) -> str:
    """Revision-qualified structure artifact key (chunk payloads + locators)."""
    return f"{revision_prefix(workspace_id, document_id, revision_id)}/structure.json"


# ---------------------------------------------------------------------------
# Vector identity
# ---------------------------------------------------------------------------


def embedding_namespace(
    workspace_id: uuid.UUID | str, model_hash: str, dimension: int
) -> str:
    """Collection namespace qualified by embedding model + dimension.

    ``ws_<workspace>_embed_<model_hash>_d<dimension>``. Two builds that differ
    in model or dimension land in different collections, so a migration never
    has to delete a collection holding published vectors.
    """
    if not model_hash:
        raise ValueError("model_hash is required to qualify an embedding namespace")
    if int(dimension) <= 0:
        raise ValueError("dimension must be positive")
    return f"ws_{workspace_id}_embed_{model_hash}_d{int(dimension)}"


def revision_vector_id(revision_id: uuid.UUID | str, ordinal: int) -> str:
    """Stable vector id owned by one revision.

    ``rev_<revision_id>_chunk_<ordinal>`` — both components are recoverable
    (:func:`parse_revision_vector_id`), which is what makes a stored locator
    reconstructible without reading mutable document state.
    """
    return f"rev_{revision_id}_chunk_{int(ordinal)}"


def legacy_vector_id(document_id: uuid.UUID | str, chunk_index: int) -> str:
    """The v1 (document-scoped) vector id — used ONLY by the legacy adapter."""
    return f"doc_{document_id}_chunk_{int(chunk_index)}"


def parse_revision_vector_id(
    vector_id: str,
) -> Optional[tuple[uuid.UUID, int]]:
    """Split ``rev_<uuid>_chunk_<ordinal>`` back into ``(revision_id, ordinal)``.

    Returns ``None`` for a legacy id (``doc_<uuid>_chunk_<n>``) or any string
    that is not a revision-qualified id.
    """
    if not isinstance(vector_id, str) or not vector_id.startswith("rev_"):
        return None
    body = vector_id[len("rev_") :]
    head, sep, tail = body.rpartition("_chunk_")
    if not sep:
        return None
    try:
        return uuid.UUID(head), int(tail)
    except (ValueError, AttributeError):
        return None


def revision_kg_scope(revision_id: uuid.UUID | str) -> str:
    """KG provenance scope for one revision (the string stored on KG rows)."""
    return str(revision_id)


# ---------------------------------------------------------------------------
# Resolved revision identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevisionArtifactIdentity:
    """Everything retrieval needs to read one revision's artifacts.

    Every field is read from the revision's **own** build manifest, so a
    historical revision keeps working after the current embedding config
    changes.
    """

    revision_id: uuid.UUID
    document_id: uuid.UUID
    generation: int
    build_profile: str
    markdown_artifact_key: Optional[str]
    structure_artifact_key: Optional[str]
    embedding_namespace: Optional[str]
    embedding_model_hash: Optional[str]
    embedding_dimension: Optional[int]
    vector_artifact_version: Optional[str]

    @property
    def vectors_available(self) -> bool:
        """True when the manifest pins a complete vector identity."""
        return (
            self.embedding_namespace is not None
            and self.embedding_model_hash is not None
            and self.embedding_dimension is not None
            and self.vector_artifact_version is not None
        )

    @property
    def kg_scope(self) -> str:
        return revision_kg_scope(self.revision_id)


async def _load_build(
    db: AsyncSession, revision_id: uuid.UUID
) -> Optional[DocumentRevisionBuild]:
    """The revision's build manifest (latest finished wins; one row per profile)."""
    return await db.scalar(
        select(DocumentRevisionBuild)
        .where(DocumentRevisionBuild.revision_id == revision_id)
        .order_by(DocumentRevisionBuild.finished_at.desc().nullslast())
        .limit(1)
    )


async def _identity_from_revision(
    db: AsyncSession,
    revision: DocumentRevision,
    *,
    require_vectors: bool,
) -> RevisionArtifactIdentity:
    """Build the artifact identity for an already-loaded revision row.

    Shared by :func:`load_revision_identity` (by id) and
    :func:`load_revision_identity_for_workspace` (workspace-scoped), so both
    enforce exactly the same published/artifact contract.
    """
    if revision.status != RETRIEVAL_READY_STATUS:
        raise RevisionNotReady(
            revision.document_id,
            f"revision status is {revision.status!r}, not published",
            revision_id=revision.revision_id,
        )
    build = await _load_build(db, revision.revision_id)
    if build is None:
        raise RevisionNotReady(
            revision.document_id,
            "published revision has no build manifest",
            revision_id=revision.revision_id,
        )
    identity = RevisionArtifactIdentity(
        revision_id=revision.revision_id,
        document_id=revision.document_id,
        generation=revision.generation,
        build_profile=build.build_profile,
        markdown_artifact_key=build.markdown_artifact_key,
        structure_artifact_key=build.structure_artifact_key,
        embedding_namespace=build.embedding_namespace,
        embedding_model_hash=build.embedding_model_hash,
        embedding_dimension=build.embedding_dimension,
        vector_artifact_version=build.vector_artifact_version,
    )
    if identity.markdown_artifact_key is None:
        raise RevisionNotReady(
            revision.document_id,
            "published revision has no markdown artifact",
            revision_id=revision.revision_id,
        )
    if require_vectors and not identity.vectors_available:
        raise RevisionNotReady(
            revision.document_id,
            "published revision has no complete vector manifest",
            revision_id=revision.revision_id,
        )
    return identity


async def load_revision_identity(
    db: AsyncSession,
    revision_id: uuid.UUID,
    *,
    require_vectors: bool = False,
) -> RevisionArtifactIdentity:
    """Resolve a revision's artifact identity from its own build manifest.

    :param require_vectors: when True, a manifest without a complete
        namespace/model-hash/dimension/version set raises
        :class:`RevisionNotReady` (the v2 retrieval contract needs vectors).

    :raises RevisionNotReady: unknown revision, not ``published``, no build
        manifest, or missing required artifacts.
    """
    revision = await db.get(DocumentRevision, revision_id)
    if revision is None:
        raise RevisionNotReady(
            "unknown", "revision does not exist", revision_id=revision_id
        )
    return await _identity_from_revision(
        db, revision, require_vectors=require_vectors
    )


async def load_revision_identity_for_workspace(
    db: AsyncSession,
    revision_id: uuid.UUID,
    workspace_id: uuid.UUID,
    *,
    require_vectors: bool = False,
) -> RevisionArtifactIdentity:
    """Resolve a revision only when it is owned by ``workspace_id``.

    This is the guard for **caller-supplied** revision ids. Chroma
    collections are global and the identity module derives a revision's
    collection namespace from the revision alone, so a revision id that is
    not joined to its document would let a caller authorized for workspace A
    read workspace B's chunk text. The join also rejects a tombstoned
    document (``source_deleted_at`` set), whose revisions must not be
    selectable for new retrieval.

    :raises RevisionNotReady: the revision does not exist, belongs to a
        different workspace, its document is tombstoned, or it fails the
        published/artifact contract.
    """
    row = (
        await db.execute(
            select(DocumentRevision, Document)
            .join(Document, Document.id == DocumentRevision.document_id)
            .where(
                DocumentRevision.revision_id == revision_id,
                Document.workspace_id == workspace_id,
                Document.source_deleted_at.is_(None),
            )
        )
    ).first()
    if row is None:
        raise RevisionNotReady(
            "unknown",
            "revision does not exist, is not owned by this workspace, or its "
            "document is tombstoned",
            revision_id=revision_id,
        )
    revision, _document = row
    return await _identity_from_revision(
        db, revision, require_vectors=require_vectors
    )


async def load_revision_vector_manifest(
    db: AsyncSession, revision_id: uuid.UUID
) -> Optional[tuple[str, str, int, str]]:
    """``(namespace, model_hash, dimension, vector_artifact_version)`` recorded
    on a revision's build manifest, or ``None`` when the embed stage has not
    run yet.

    Unlike :func:`load_revision_identity` this does NOT require the revision to
    be published: a worker mid-build (e.g. the caption re-embed) must write
    into exactly the collection the embed stage created.
    """
    build = await _load_build(db, revision_id)
    if build is None or not (
        build.embedding_namespace
        and build.embedding_model_hash
        and build.embedding_dimension
        and build.vector_artifact_version
    ):
        return None
    return (
        build.embedding_namespace,
        build.embedding_model_hash,
        build.embedding_dimension,
        build.vector_artifact_version,
    )


async def load_current_revision_identity_for_workspace(
    db: AsyncSession,
    document_id: uuid.UUID,
    workspace_id: uuid.UUID,
    *,
    require_vectors: bool = False,
) -> Optional[RevisionArtifactIdentity]:
    """Resolve the document's *current* revision only inside ``workspace_id``.

    Defense-in-depth for caller-supplied document ids: the row is joined to
    its owning document with a workspace match and a tombstone rejection
    (``source_deleted_at`` set), so a foreign-workspace or tombstoned
    document raises :class:`RevisionNotReady` instead of pinning. A document
    with no current revision returns ``None`` (legacy), exactly like
    :func:`load_current_revision_identity`.

    :raises RevisionNotReady: the document does not exist, is not owned by
        this workspace, is tombstoned, or its current revision fails the
        published/artifact contract.
    """
    row = (
        await db.execute(
            select(Document.current_revision_id).where(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
                Document.source_deleted_at.is_(None),
            )
        )
    ).first()
    if row is None:
        raise RevisionNotReady(
            document_id,
            "document does not exist, is not owned by this workspace, or "
            "is tombstoned",
        )
    current_revision_id = row[0]
    if current_revision_id is None:
        return None
    return await load_revision_identity(
        db, current_revision_id, require_vectors=require_vectors
    )


async def load_current_revision_identity(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    require_vectors: bool = False,
) -> Optional[RevisionArtifactIdentity]:
    """Resolve the document's *current* revision identity.

    Returns ``None`` when the document has no current revision (a legacy
    document, or a tombstoned one whose pointer was cleared) — the caller then
    uses the v1 adapter. A document whose pointer is set but whose revision is
    unreadable raises :class:`RevisionNotReady` rather than silently falling
    back to legacy artifacts.
    """
    current_revision_id = await db.scalar(
        select(Document.current_revision_id).where(Document.id == document_id)
    )
    if current_revision_id is None:
        return None
    return await load_revision_identity(
        db, current_revision_id, require_vectors=require_vectors
    )


@dataclass(frozen=True)
class DocumentRetrievalTarget:
    """One requested document's retrieval target.

    ``identity`` is ``None`` for a legacy document (no current revision): the
    caller must use the unchanged v1 adapter for it. ``eligible`` is ``False``
    for a document that must not be retrieved at all (tombstoned, or outside
    the requested workspace): the caller skips it instead of falling back to
    the legacy adapter, because the tombstone cleared ``current_revision_id``
    and a legacy fallback would serve deleted content.
    """

    document_id: uuid.UUID
    identity: Optional[RevisionArtifactIdentity]
    eligible: bool = True

    @property
    def is_legacy(self) -> bool:
        return self.eligible and self.identity is None


async def resolve_document_targets(
    db: AsyncSession,
    document_ids: Sequence[uuid.UUID],
    *,
    workspace_id: Optional[uuid.UUID] = None,
) -> list[DocumentRetrievalTarget]:
    """Resolve each document to its current revision, or to ``None`` (legacy).

    Order follows ``document_ids``; duplicates are preserved so a caller can
    zip the result back onto its request. When ``workspace_id`` is supplied, a
    document outside that workspace is marked ``eligible=False``; a tombstoned
    document is always marked ineligible.
    """
    targets: list[DocumentRetrievalTarget] = []
    for document_id in document_ids:
        conditions = [
            Document.id == document_id,
            Document.source_deleted_at.is_(None),
        ]
        if workspace_id is not None:
            conditions.append(Document.workspace_id == workspace_id)
        row = (
            await db.execute(
                select(Document.current_revision_id).where(*conditions)
            )
        ).first()
        if row is None:
            targets.append(
                DocumentRetrievalTarget(
                    document_id=document_id,
                    identity=None,
                    eligible=False,
                )
            )
            continue
        current_revision_id = row[0]
        identity = (
            None
            if current_revision_id is None
            else await load_revision_identity(db, current_revision_id)
        )
        targets.append(
            DocumentRetrievalTarget(document_id=document_id, identity=identity)
        )
    return targets


async def resolve_retrieval_revisions(
    db: AsyncSession, document_ids: Sequence[uuid.UUID]
) -> list[RevisionArtifactIdentity]:
    """The v2 binding: every requested document MUST have a published revision.

    :raises RevisionNotReady: the first document without a current published
        revision (the v2 port never falls back to legacy chunks/KG).
    """
    identities: list[RevisionArtifactIdentity] = []
    for document_id in document_ids:
        identity = await load_current_revision_identity(
            db, document_id, require_vectors=True
        )
        if identity is None:
            raise RevisionNotReady(
                document_id,
                "document has no current revision (legacy document); "
                "reindex it to publish a revision before using the v2 path",
            )
        identities.append(identity)
    return identities


# ---------------------------------------------------------------------------
# Structure artifact (chunk payloads + stable locators)
# ---------------------------------------------------------------------------


@dataclass
class ChunkRecord:
    """One revision-owned chunk payload with its stable locator."""

    chunk_id: str
    ordinal: int
    content: str
    page_no: int = 0
    heading_path: list[str] = field(default_factory=list)
    source_file: str = ""
    image_refs: list[str] = field(default_factory=list)
    table_refs: list[str] = field(default_factory=list)
    has_table: bool = False
    has_code: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "ordinal": self.ordinal,
            "content": self.content,
            "page_no": self.page_no,
            "heading_path": list(self.heading_path),
            "source_file": self.source_file,
            "image_refs": list(self.image_refs),
            "table_refs": list(self.table_refs),
            "has_table": self.has_table,
            "has_code": self.has_code,
        }


def build_structure_artifact(
    revision_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
    chunks: Iterable[ChunkRecord],
) -> str:
    """Serialize the revision structure artifact (chunk payloads) to JSON."""
    return json.dumps(
        {
            "artifact_version": STRUCTURE_ARTIFACT_VERSION,
            "revision_id": str(revision_id),
            "document_id": str(document_id),
            "chunks": [c.as_dict() for c in chunks],
        },
        ensure_ascii=False,
    )


def parse_structure_artifact(raw: str) -> list[ChunkRecord]:
    """Parse a structure artifact back into :class:`ChunkRecord` rows.

    Tolerates a missing/blank artifact (returns ``[]``) so a PARSE_ONLY
    revision whose structure write failed degrades to "no chunks" rather than
    a hard error at view time.
    """
    if not raw or not raw.strip():
        return []
    payload = json.loads(raw)
    records: list[ChunkRecord] = []
    for ordinal, entry in enumerate(payload.get("chunks") or []):
        records.append(
            ChunkRecord(
                chunk_id=str(entry.get("chunk_id") or ""),
                ordinal=int(entry.get("ordinal", ordinal)),
                content=entry.get("content") or "",
                page_no=int(entry.get("page_no") or 0),
                heading_path=list(entry.get("heading_path") or []),
                source_file=entry.get("source_file") or "",
                image_refs=list(entry.get("image_refs") or []),
                table_refs=list(entry.get("table_refs") or []),
                has_table=bool(entry.get("has_table")),
                has_code=bool(entry.get("has_code")),
            )
        )
    return records


async def load_revision_chunks(
    db: AsyncSession, identity: RevisionArtifactIdentity, *, storage=None
) -> list[ChunkRecord]:
    """Load a revision's chunk payloads from its structure artifact.

    The SQL ``document_revision_chunks`` rows are the stable locators; the
    payloads live in the revision's structure artifact (the ORM docstring
    documents this split). ``storage`` is injectable so tests can supply a fake
    artifact store; the real default is the MinIO ``StorageService`` singleton.
    """
    if identity.structure_artifact_key is None:
        return []
    if storage is None:
        from app.services.storage_service import get_storage_service

        storage = get_storage_service()
    raw = await storage.download_markdown(identity.structure_artifact_key)
    return parse_structure_artifact(raw)


# ---------------------------------------------------------------------------
# CurrentDocumentViewAdapter
# ---------------------------------------------------------------------------


class CurrentDocumentViewAdapter:
    """Serve the document viewer endpoints from the document's current revision.

    Contract (brief Step 3): resolve ``Document.current_revision_id`` and load
    that revision's markdown/images/chunks. It must **never** fall back to
    legacy ``Document.markdown_s3_key`` / ``Document.chunk_count`` /
    ``doc_<document>_chunk_<index>`` when a current revision exists — the
    fallback exists only for documents that have no revision at all (the v1
    documents that predate revision-aware ingestion).
    """

    def __init__(self, db: AsyncSession, *, storage=None) -> None:
        self.db = db
        self._storage = storage

    @property
    def storage(self):
        """The artifact store (lazy MinIO singleton unless one was injected)."""
        if self._storage is None:
            from app.services.storage_service import get_storage_service

            self._storage = get_storage_service()
        return self._storage

    async def current_identity(
        self, document_id: uuid.UUID
    ) -> Optional[RevisionArtifactIdentity]:
        """The current revision's identity, or ``None`` for a legacy document."""
        return await load_current_revision_identity(self.db, document_id)

    async def load_markdown(self, document_id: uuid.UUID) -> str:
        """Markdown for the current revision (legacy key only when no revision)."""
        identity = await self.current_identity(document_id)
        if identity is not None:
            if identity.markdown_artifact_key is None:
                raise RevisionNotReady(
                    document_id,
                    "current revision has no markdown artifact",
                    revision_id=identity.revision_id,
                )
            return await self.storage.download_markdown(
                identity.markdown_artifact_key
            )
        return await self._legacy_markdown(document_id)

    async def _legacy_markdown(self, document_id: uuid.UUID) -> str:
        key = await self.db.scalar(
            select(Document.markdown_s3_key).where(Document.id == document_id)
        )
        if not key:
            raise RevisionNotReady(
                document_id, "no current revision and no legacy markdown"
            )
        return await self.storage.download_markdown(key)

    async def load_chunks(self, document_id: uuid.UUID) -> list[ChunkRecord]:
        """Chunk payloads for the current revision.

        Returns ``[]`` for a legacy document: legacy chunk payloads live in
        ChromaDB under ``doc_<document>_chunk_<index>`` and are the v1
        adapter's business, not this adapter's.
        """
        identity = await self.current_identity(document_id)
        if identity is None:
            return []
        return await load_revision_chunks(self.db, identity, storage=self._storage)

    async def load_chunk_context(
        self,
        document_id: uuid.UUID,
        *,
        chunk_index: Optional[int] = None,
        page_no: Optional[int] = None,
        heading_path: Optional[str] = None,
        context_window: int = 2,
    ) -> dict[str, Any]:
        """Target chunk + neighbours, with stable locators.

        ``/document/{id}/chunk-context`` resolves the target ordinal from the
        revision's own chunk rows (never from ``Document.chunk_count``).
        """
        identity = await self.current_identity(document_id)
        if identity is None:
            return {
                "document_id": str(document_id),
                "revision_id": None,
                "legacy": True,
                "target_chunk_index": chunk_index or 0,
                "chunk_range": [0, 0],
                "total_chunks": 0,
                "chunks": [],
                "markdown": "",
            }
        chunks = await load_revision_chunks(
            self.db, identity, storage=self._storage
        )
        by_ordinal = {c.ordinal: c for c in chunks}
        total = len(chunks)

        target = chunk_index
        if target is None and heading_path is not None:
            target = _ordinal_for_heading(chunks, heading_path, page_no)
        if target is None and page_no is not None:
            target = _ordinal_for_page(chunks, page_no)
        if target is None:
            target = chunks[0].ordinal if chunks else 0

        start = max(0, target - context_window)
        end = min((total - 1) if total else 0, target + context_window)
        if start > end:
            start = end

        window = [by_ordinal[o] for o in range(start, end + 1) if o in by_ordinal]
        return {
            "document_id": str(document_id),
            "revision_id": str(identity.revision_id),
            "legacy": False,
            "target_chunk_index": target,
            "chunk_range": [start, end],
            "total_chunks": total,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "vector_id": revision_vector_id(identity.revision_id, c.ordinal),
                    "chunk_index": c.ordinal,
                    "content": c.content,
                    "page_no": c.page_no,
                    "heading_path": " > ".join(c.heading_path),
                    "source": c.source_file,
                }
                for c in window
            ],
            "markdown": _assemble_chunk_markdown(window),
        }

    async def load_images(self, document_id: uuid.UUID) -> list[DocumentImage]:
        """Images owned by the current revision (legacy rows only when none)."""
        identity = await self.current_identity(document_id)
        stmt = select(DocumentImage).where(DocumentImage.document_id == document_id)
        if identity is not None:
            stmt = stmt.where(DocumentImage.revision_id == identity.revision_id)
        stmt = stmt.order_by(DocumentImage.page_no)
        return list((await self.db.scalars(stmt)).all())


def _ordinal_for_heading(
    chunks: list[ChunkRecord], heading_path: str, page_no: Optional[int]
) -> Optional[int]:
    """Smallest ordinal whose heading_path/page matches the requested locator."""
    wanted = heading_path.strip()
    matches = [
        c
        for c in chunks
        if " > ".join(c.heading_path) == wanted
        and (page_no is None or c.page_no == page_no)
    ]
    if not matches:
        return None
    return min(c.ordinal for c in matches)


def _ordinal_for_page(chunks: list[ChunkRecord], page_no: int) -> Optional[int]:
    matches = [c for c in chunks if c.page_no == page_no]
    if not matches:
        return None
    return min(c.ordinal for c in matches)


def _assemble_chunk_markdown(chunks: Sequence[ChunkRecord]) -> str:
    """The chunk-context markdown body the frontend already expects."""
    parts: list[str] = []
    last_heading = ""
    last_page: Optional[int] = None
    for chunk in chunks:
        heading = " > ".join(chunk.heading_path)
        if heading and heading != last_heading:
            parts.append(f"\n### 📍 {heading}\n")
            last_heading = heading
        if chunk.page_no and chunk.page_no != last_page:
            parts.append(f"\n<!-- page {chunk.page_no} -->\n")
            last_page = chunk.page_no
        parts.append(chunk.content)
    return "\n\n".join(parts)


async def record_revision_chunk_rows(
    db: AsyncSession,
    revision_id: uuid.UUID,
    chunks: Sequence[ChunkRecord],
) -> list[ChunkRecord]:
    """Replace the revision's ``document_revision_chunks`` locator rows.

    Returns the chunks with ``chunk_id`` populated (a fresh UUID when the
    caller had none), so the structure artifact and the SQL rows share one
    identity per chunk. Scoped to ``revision_id``: a re-run replaces only its
    own revision's rows, never another revision's.
    """
    from sqlalchemy import delete

    await db.execute(
        delete(DocumentRevisionChunk).where(
            DocumentRevisionChunk.revision_id == revision_id
        )
    )
    for chunk in chunks:
        if not chunk.chunk_id:
            chunk.chunk_id = str(uuid.uuid4())
        db.add(
            DocumentRevisionChunk(
                chunk_id=uuid.UUID(chunk.chunk_id),
                revision_id=revision_id,
                ordinal=chunk.ordinal,
            )
        )
    await db.flush()
    return list(chunks)
