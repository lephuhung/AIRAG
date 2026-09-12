"""
Deep RAG Service
=================

Orchestrator for the HRAG pipeline:
  Document → Docling Parse → ChromaDB Index + LightRAG KG → Hybrid Retrieval

Backward-compatible: exposes the same `process_document()`, `query()`,
`delete_document()`, `get_chunk_count()` interface as legacy RAGService.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable, DocumentStatus
from app.services.parsing.deep_document_parser import DeepDocumentParser
from app.services.kg.knowledge_graph_service import get_kg_service
from app.services.retrieval.deep_retriever import DeepRetriever
from app.services.embedding.embedder import EmbeddingService, get_embedding_service
from app.services.parsing.heading_path import extract_article_nos
from app.services.embedding.vector_store import VectorStore, get_vector_store
from app.services.retrieval.reranker import get_reranker_service
from app.services.retrieval.rag_service import RAGQueryResult, RetrievedChunk
from app.services.models.parsed_document import DeepRetrievalResult
from app.services.agents.v2.persistence.document_views import (
    RevisionArtifactIdentity,
    RevisionNotReady,
)

logger = logging.getLogger(__name__)


class HRAGService:
    """
    Full HRAG pipeline orchestrator.

    Phases:
      1. PARSING  — Docling parse → markdown + chunks + images
      2. INDEXING — Embed chunks → ChromaDB + ingest markdown → LightRAG KG
      3. INDEXED  — Update document metadata in DB

    Query:
      - query()       — backward-compatible sync vector-only search
      - query_deep()  — full async hybrid retrieval (KG + vector + images)
    """

    def __init__(self, db: AsyncSession, workspace_id: uuid.UUID):
        self.db = db
        self.workspace_id = workspace_id

        # Services
        self.parser = DeepDocumentParser(workspace_id=workspace_id)
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)

        # KG service (optional, gated by config)
        self.kg_service = None
        if settings.HRAG_ENABLE_KG:
            self.kg_service = get_kg_service(workspace_id=workspace_id)

        # Retriever (with cross-encoder reranker)
        self.retriever = DeepRetriever(
            workspace_id=workspace_id,
            kg_service=self.kg_service,
            vector_store=self.vector_store,
            embedder=self.embedder,
            db=db,
            reranker=get_reranker_service(),
        )

    # ------------------------------------------------------------------
    # Document Processing
    # ------------------------------------------------------------------

    async def process_document(self, document_id: uuid.UUID, file_path: str) -> int:
        """
        Process a document through the full HRAG pipeline.

        Returns:
            Number of chunks created
        """
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        start_time = time.time()
        _cleanup_tmp: str | None = None

        # If file_path doesn't exist on disk but upload_s3_key is set,
        # download from MinIO and use a temp file
        from pathlib import Path as _P
        if not _P(file_path).exists() and document.upload_s3_key:
            import tempfile
            from app.services.storage_service import get_storage_service as _get_storage
            file_bytes = await _get_storage().download_file(document.upload_s3_key)
            ext = _P(file_path).suffix or _P(document.upload_s3_key).suffix
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(file_bytes)
            tmp.close()
            file_path = tmp.name
            _cleanup_tmp = file_path
            logger.info(
                f"[hrag] doc={document_id} downloaded from MinIO "
                f"({document.upload_s3_key}) → {file_path}"
            )

        try:
            # Phase 1: PARSING
            document.status = DocumentStatus.PARSING
            await self.db.commit()

            parsed = await self.parser.parse(
                file_path=file_path,
                document_id=document_id,
                original_filename=document.original_filename,
            )

            # Save markdown + images to DB
            from app.services.storage_service import get_storage_service
            storage = get_storage_service()
            s3_key = await storage.upload_markdown(
                workspace_id=self.workspace_id,
                document_id=document_id,
                content=parsed.markdown,
            )
            document.markdown_s3_key = s3_key
            document.page_count = parsed.page_count
            document.table_count = parsed.tables_count
            document.parser_version = (
                "docling" if DeepDocumentParser.is_docling_supported(file_path) else "legacy"
            )
            await self.db.commit()

            # Clean up old image records before saving new ones (handles re-processing)
            await self.db.execute(
                delete(DocumentImage).where(DocumentImage.document_id == document_id)
            )
            await self.db.commit()

            # Save extracted images to DB
            for img in parsed.images:
                db_image = DocumentImage(
                    document_id=document_id,
                    image_id=img.image_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                )
                self.db.add(db_image)
            if parsed.images:
                document.image_count = len(parsed.images)
                await self.db.commit()

            # Clean up old table records before saving new ones (handles re-processing)
            await self.db.execute(
                delete(DocumentTable).where(DocumentTable.document_id == document_id)
            )
            await self.db.commit()

            # Save extracted tables to DB
            for tbl in parsed.tables:
                db_table = DocumentTable(
                    document_id=document_id,
                    table_id=tbl.table_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption=tbl.caption,
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                )
                self.db.add(db_table)
            if parsed.tables:
                await self.db.commit()

            # Phase 2: INDEXING
            document.status = DocumentStatus.INDEXING
            await self.db.commit()

            chunk_count = 0
            if parsed.chunks:
                # Embed and store in ChromaDB
                chunk_texts = [c.content for c in parsed.chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                ids = [
                    f"doc_{document_id}_chunk_{i}"
                    for i in range(len(parsed.chunks))
                ]
                # Build image_id→URL lookup for metadata
                _img_url_map = {
                    img.image_id: f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                    for img in parsed.images
                }

                metadatas = [
                    {
                        "document_id": str(document_id),
                        "chunk_index": c.chunk_index,
                        "source": c.source_file,
                        "file_type": document.file_type,
                        "page_no": c.page_no,
                        "heading_path": " > ".join(c.heading_path) if c.heading_path else "",
                        # Số Điều dạng cấu trúc ("17|18") — tra cứu điều khoản
                        # chính xác, không regex trên chuỗi heading_path
                        "article_nos": "|".join(extract_article_nos(c.heading_path)),
                        "has_table": c.has_table,
                        "has_code": c.has_code,
                        # Image-aware metadata: pipe-separated IDs and URLs
                        "image_ids": "|".join(c.image_refs) if c.image_refs else "",
                        "table_ids": "|".join(c.table_refs) if c.table_refs else "",
                        "image_urls": "|".join(
                            _img_url_map.get(iid, "") for iid in c.image_refs
                        ) if c.image_refs else "",
                        # Recency boost: published date for date-aware scoring
                        "published_date": document.published_date or "",
                    }
                    for c in parsed.chunks
                ]

                self.vector_store.add_documents(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=metadatas,
                )
                chunk_count = len(parsed.chunks)

            # KG ingest (async, non-blocking failure)
            if self.kg_service and parsed.markdown:
                try:
                    await self.kg_service.ingest(parsed.markdown, document_id=document_id)
                except Exception as e:
                    logger.error(
                        f"KG ingest failed for document {document_id}, "
                        f"continuing without KG: {e}"
                    )

            # Phase 3: INDEXED
            elapsed_ms = int((time.time() - start_time) * 1000)
            document.status = DocumentStatus.INDEXED
            document.chunk_count = chunk_count
            document.processing_time_ms = elapsed_ms
            await self.db.commit()

            logger.info(
                f"HRAG processed document {document_id}: "
                f"{chunk_count} chunks, {len(parsed.images)} images, "
                f"{parsed.tables_count} tables in {elapsed_ms}ms"
            )
            return chunk_count

        except Exception as e:
            logger.error(f"HRAG failed for document {document_id}: {e}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await self.db.commit()
            raise
        finally:
            # Clean up temp file if we downloaded from MinIO
            if _cleanup_tmp:
                import os as _os
                try:
                    _os.unlink(_cleanup_tmp)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[list[uuid.UUID]] = None,
        revision_identities: Optional[list[RevisionArtifactIdentity]] = None,
    ) -> RAGQueryResult:
        """
        Backward-compatible sync query (vector-only).
        Returns same RAGQueryResult as legacy RAGService.

        With ``revision_identities`` the search is revision-selected: each
        revision's own recorded namespace is queried with an exact
        ``revision_id`` filter, so a current revision's answer can never be
        built from another revision's vectors.
        """
        query_embedding = self.embedder.embed_query(question)

        if revision_identities:
            chunks = []
            seen: set[str] = set()
            for identity in revision_identities:
                if not identity.embedding_namespace:
                    raise RevisionNotReady(
                        identity.document_id,
                        "revision has no vector manifest for retrieval",
                        revision_id=identity.revision_id,
                    )
                store = get_vector_store(
                    self.workspace_id, namespace=identity.embedding_namespace
                )
                where: dict = {"revision_id": str(identity.revision_id)}
                if document_ids:
                    where = {
                        "$and": [
                            where,
                            {
                                "document_id": {
                                    "$in": [str(d) for d in document_ids]
                                }
                            },
                        ]
                    }
                results = store.query(
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where=where,
                )
                for i, doc in enumerate(results.get("documents", [])):
                    chunk_id = results["ids"][i] if results.get("ids") else ""
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    chunks.append(RetrievedChunk(
                        content=doc,
                        metadata=(
                            results["metadatas"][i]
                            if results.get("metadatas")
                            else {}
                        ),
                        score=(
                            results["distances"][i]
                            if results.get("distances")
                            else 0.0
                        ),
                        chunk_id=chunk_id,
                    ))
            chunks.sort(key=lambda x: x.score)
            chunks = chunks[:top_k]
            return RAGQueryResult(
                chunks=chunks,
                context=self._assemble_sync_context(chunks),
                query=question,
            )

        where = None
        if document_ids:
            where = {"document_id": {"$in": [str(doc_id) for doc_id in document_ids]}}

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        chunks = []
        for i, doc in enumerate(results.get("documents", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            chunks.append(RetrievedChunk(
                content=doc,
                metadata=meta,
                score=results["distances"][i] if results.get("distances") else 0.0,
                chunk_id=results["ids"][i] if results.get("ids") else "",
            ))

        chunks.sort(key=lambda x: x.score)
        return RAGQueryResult(
            chunks=chunks,
            context=self._assemble_sync_context(chunks),
            query=question,
        )

    @staticmethod
    def _assemble_sync_context(chunks: list[RetrievedChunk]) -> str:
        """Citation-prefixed context body shared by both sync query paths."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page_no", 0)
            heading = chunk.metadata.get("heading_path", "")
            citation = source
            if page:
                citation += f" | p.{page}"
            if heading:
                citation += f" | {heading}"
            context_parts.append(f"[{i + 1}] {citation}\n{chunk.content}")
        return "\n\n---\n\n".join(context_parts)

    async def query_deep(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[list[uuid.UUID]] = None,
        mode: str = "hybrid",
        include_images: bool = True,
        revision_identities: Optional[list[RevisionArtifactIdentity]] = None,
    ) -> DeepRetrievalResult:
        """
        Full async hybrid retrieval with KG + vector + images + citations.

        ``revision_identities`` selects exact revisions (the v2 path). Without
        it the unchanged v1 document-scoped adapter path runs.
        """
        return await self.retriever.query(
            question=question,
            mode=mode,
            top_k=top_k,
            document_ids=document_ids,
            include_images=include_images,
            revision_identities=revision_identities,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Physical purge of a document's vector + KG data (v1/legacy only).

        This is the **legacy** purge used by chat-temp cleanup and the admin
        cancel path. It is deliberately NOT called by the tombstone-first
        ``DELETE /documents/{id}`` endpoint or by reindex: tombstoning keeps
        every revision and artifact, and only Task 9's GC reclaims them. A
        revision-aware caller must use
        ``vector_store.delete_by_document_id(..., revision_id=...)`` per
        revision instead of this document-scoped purge.
        """
        # Delete from ChromaDB
        self.vector_store.delete_by_document_id(document_id)

        # Delete from KG (Neo4j / LightRAG) — LegalKGService and KnowledgeGraphService
        # both implement delete_document(document_id)
        try:
            kg_service = get_kg_service(workspace_id=self.workspace_id)
            await kg_service.delete_document(document_id)
        except Exception as e:
            logger.warning(
                f"[hrag_service] KG deletion failed for doc={document_id}: {e}"
            )
            # Non-fatal: vector data is already deleted; continue

        # Delete images from DB (cascade handles it, but clean up files)
        result = await self.db.execute(
            select(DocumentImage).where(DocumentImage.document_id == document_id)
        )
        for img in result.scalars().all():
            from pathlib import Path
            img_path = Path(img.file_path)
            if img_path.exists():
                img_path.unlink()

        logger.info(f"Deleted document {document_id} from HRAG stores")

    def get_chunk_count(self) -> int:
        """Return total number of chunks in the knowledge base's vector store."""
        return self.vector_store.count()
