"""
RAG (Retrieval-Augmented Generation) Service
Main service that orchestrates document processing, indexing, and retrieval.
"""
from __future__ import annotations

import logging
import uuid as uuid_lib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import Document, DocumentStatus
from app.services.parsing.document_loader import load_document, LoadedDocument
from app.services.embedding.chunker import DocumentChunker, TextChunk
from app.services.embedding.embedder import EmbeddingService, get_embedding_service
from app.services.embedding.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with its relevance score."""
    content: str
    metadata: dict
    score: float  # Lower is more similar (distance)
    chunk_id: str


@dataclass
class RAGQueryResult:
    """Result of a RAG query."""
    chunks: list[RetrievedChunk]
    context: str  # Concatenated chunks for LLM context
    query: str


class RAGService:
    """
    Main RAG service that handles document processing and retrieval.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_id: uuid_lib.UUID,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG service.

        Args:
            db: Database session
            workspace_id: Knowledge base ID for isolation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.db = db
        self.workspace_id = workspace_id
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)

    async def process_document(self, document_id: uuid_lib.UUID, file_path: str) -> int:
        """
        Process a document: load, chunk, embed, and store.

        Args:
            document_id: Database document ID
            file_path: Path to the document file

        Returns:
            Number of chunks created

        Raises:
            ValueError: If document processing fails
        """
        # Get document from DB
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if document is None:
            raise ValueError(f"Document {document_id} not found")

        try:
            # Update status to processing
            document.status = DocumentStatus.PROCESSING
            await self.db.commit()

            # Load document
            logger.info(f"Loading document {document_id} from {file_path}")
            loaded = load_document(file_path)

            # Chunk text
            logger.info(f"Chunking document {document_id}")
            chunks = self.chunker.split_text(
                text=loaded.content,
                source=document.original_filename,
                extra_metadata={
                    "document_id": document_id,
                    "file_type": loaded.file_type,
                    "page_count": loaded.page_count
                }
            )

            if not chunks:
                document.status = DocumentStatus.INDEXED
                document.chunk_count = 0
                await self.db.commit()
                logger.warning(f"Document {document_id} produced no chunks (empty content)")
                return 0

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_texts(chunk_texts)

            # Prepare data for vector store
            ids = [f"doc_{document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": str(document_id),
                    "chunk_index": c.chunk_index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "source": c.metadata.get("source", ""),
                    "file_type": c.metadata.get("file_type", "")
                }
                for c in chunks
            ]

            # Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in vector store")
            self.vector_store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )

            # Update document status
            document.status = DocumentStatus.INDEXED
            document.chunk_count = len(chunks)
            await self.db.commit()

            logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await self.db.commit()
            raise

    async def delete_document(self, document_id: uuid_lib.UUID) -> None:
        """
        Delete a document's chunks from the vector store.

        Args:
            document_id: Database document ID
        """
        self.vector_store.delete_by_document_id(document_id)
        logger.info(f"Deleted document {document_id} from vector store")

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[uuid_lib.UUID] | None = None
    ) -> RAGQueryResult:
        """
        Query the vector store for relevant chunks.

        Args:
            question: The query question
            top_k: Number of chunks to retrieve
            document_ids: Optional filter to specific documents

        Returns:
            RAGQueryResult with retrieved chunks and assembled context
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(question)

        # Build filter
        where = None
        if document_ids:
            where = {"document_id": {"$in": [str(doc_id) for doc_id in document_ids]}}

        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where
        )

        # Build retrieved chunks
        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append(RetrievedChunk(
                content=doc,
                metadata=results["metadatas"][i] if results["metadatas"] else {},
                score=results["distances"][i] if results["distances"] else 0.0,
                chunk_id=results["ids"][i] if results["ids"] else ""
            ))

        # Sort by score (lower distance = more similar)
        chunks.sort(key=lambda x: x.score)

        # Assemble context
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            context_parts.append(f"[Source: {source}, Chunk {i+1}]\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        return RAGQueryResult(
            chunks=chunks,
            context=context,
            query=question
        )

    def get_chunk_count(self) -> int:
        """Return total number of chunks in the knowledge base's vector store."""
        return self.vector_store.count()


# --- Knowledge Graph Service Factory ---
# Re-export the central factory from knowledge_graph_service so existing callers
# that import get_kg_service from rag_service continue to work.
_kg_service_cache: dict = {}

async def get_kg_service(workspace_id: uuid_lib.UUID):
    """Get KG service for a workspace — cached, routed via HRAG_KG_MODE."""
    from app.services.kg.knowledge_graph_service import get_kg_service as _factory
    if workspace_id not in _kg_service_cache:
        _kg_service_cache[workspace_id] = _factory(workspace_id)
    return _kg_service_cache[workspace_id]


# --- RAG Service Factory ---

def get_rag_service(db: AsyncSession, workspace_id: uuid_lib.UUID) -> "RAGService | HRAGService":
    """Factory function: routes to HRAGService or legacy RAGService based on config."""
    from app.core.config import settings

    if settings.HRAG_ENABLED:
        from app.services.retrieval.hrag_service import HRAGService
        from app.services.kg.knowledge_graph_service import get_kg_service as _kg_factory

        # Build a FRESH HRAGService per request so its db / retriever.db are
        # request-PRIVATE. The old module-level cache reused one instance per
        # workspace and rebound .db on every call — two concurrent requests to
        # the SAME workspace then raced on a shared AsyncSession (which is not
        # safe to use from multiple tasks concurrently). Everything the service
        # wires is a process-wide singleton (embedder, reranker, Chroma client,
        # Docling converter) so construction is cheap — EXCEPT the KG service,
        # which lazily opens a Neo4j driver, so THAT stays cached per workspace
        # (reuse the existing _kg_service_cache) to avoid a driver per request.
        svc = HRAGService(db=db, workspace_id=workspace_id)
        if svc.kg_service is not None:
            if workspace_id not in _kg_service_cache:
                _kg_service_cache[workspace_id] = _kg_factory(workspace_id)
            svc.kg_service = _kg_service_cache[workspace_id]
            svc.retriever.kg_service = _kg_service_cache[workspace_id]
        return svc

    return RAGService(db=db, workspace_id=workspace_id)
