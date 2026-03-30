"""
Embed Worker
============
Consumes hrag.embed queue.

Responsibilities:
  1. Load raw_chunks_json from DB
  2. (Optional) Contextual Embeddings: enrich each chunk with LLM-generated
     situating context before embedding (see HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS)
  3. Embed all chunks with bge-m3
  4. Store in ChromaDB
  5. Set embed_done=True  →  status = EMBEDDING (searchable now)
  6. Clear raw_chunks_json to free DB space
  7. Check if fully INDEXED via check_and_finalize
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401
from app.models.document import Document, DocumentStatus
from app.queue.messages import EmbedMessage
from app.services.embedder import get_embedding_service
from app.services.vector_store import get_vector_store
from app.workers.utils import check_and_finalize

logger = logging.getLogger(__name__)


async def handle_embed(payload: dict) -> None:
    msg = EmbedMessage(**payload)
    logger.info(f"[embed_worker] doc={msg.document_id}")

    async with async_session_maker() as db:
        result = await db.execute(select(Document).where(Document.id == msg.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[embed_worker] doc={msg.document_id} not found")
            return

        try:
            # Set EMBEDDING at start
            document.status = DocumentStatus.EMBEDDING
            await db.commit()

            raw = document.raw_chunks_json
            if not raw:
                logger.warning(f"[embed_worker] doc={msg.document_id} has no raw_chunks_json — skipping embed")
                document.embed_done = True
                await db.commit()
                await check_and_finalize(document, db)
                return

            chunks_data: list[dict] = json.loads(raw)
            if not chunks_data:
                document.embed_done = True
                document.chunk_count = 0
                await db.commit()
                await check_and_finalize(document, db)
                return

            # ── Contextual Embeddings (optional) ────────────────────────────
            # When enabled, each chunk is enriched with a short LLM-generated
            # sentence that situates it within the full document before embedding.
            # The original content is kept for display; only embed_texts changes.
            # See: https://www.anthropic.com/engineering/contextual-retrieval
            embed_texts = [c["content"] for c in chunks_data]
            if settings.HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS:
                try:
                    from app.services.contextual_embedder import enrich_chunks_with_context
                    from app.services.storage_service import get_storage_service

                    document_markdown = ""
                    if document.markdown_s3_key:
                        try:
                            storage = get_storage_service()
                            document_markdown = await storage.download_markdown(document.markdown_s3_key)
                        except Exception as _md_err:
                            logger.warning(
                                f"[embed_worker] doc={msg.document_id} "
                                f"could not load markdown for contextual enrichment: {_md_err}"
                            )

                    if document_markdown:
                        embed_texts = await enrich_chunks_with_context(
                            document_markdown=document_markdown,
                            chunks=chunks_data,
                        )
                        logger.info(
                            f"[embed_worker] doc={msg.document_id} "
                            f"contextual enrichment done for {len(chunks_data)} chunks"
                        )
                    else:
                        logger.warning(
                            f"[embed_worker] doc={msg.document_id} "
                            f"skipping contextual enrichment — no markdown available"
                        )
                except Exception as _ctx_err:
                    logger.warning(
                        f"[embed_worker] doc={msg.document_id} "
                        f"contextual enrichment failed (falling back to plain content): {_ctx_err}"
                    )

            # ── Embed ───────────────────────────────────────────────────────
            embedder     = get_embedding_service()
            vector_store = get_vector_store(msg.workspace_id)

            embeddings = embedder.embed_texts(embed_texts)

            ids = [
                f"doc_{msg.document_id}_chunk_{c['chunk_index']}"
                for c in chunks_data
            ]
            img_url_prefix = f"/static/doc-images/kb_{msg.workspace_id}/images"
            metadatas = [
                {
                    "document_id":     msg.document_id,
                    "chunk_index":     c["chunk_index"],
                    "source":          c["source_file"],
                    "file_type":       document.file_type,
                    "page_no":         c["page_no"],
                    "heading_path":    " > ".join(c["heading_path"]) if c["heading_path"] else "",
                    "has_table":       c["has_table"],
                    "has_code":        c["has_code"],
                    "image_ids":       "|".join(c["image_refs"]) if c["image_refs"] else "",
                    "table_ids":       "|".join(c["table_refs"]) if c["table_refs"] else "",
                    "image_urls":      "|".join(
                        f"{img_url_prefix}/{iid}.png" for iid in c["image_refs"]
                    ) if c["image_refs"] else "",
                    "document_number": c.get("document_number", ""),
                }
                for c in chunks_data
            ]

            try:
                vector_store.add_documents(
                    ids=ids,
                    embeddings=embeddings,
                    documents=[c["content"] for c in chunks_data],   # store original for display
                    metadatas=metadatas,
                )
                logger.info(
                    f"[embed_worker] doc={msg.document_id} added "
                    f"{len(ids)} chunks to ChromaDB collection={vector_store.collection_name}"
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                logger.error(
                    f"[embed_worker] doc={msg.document_id} CHROMADB ADD FAILED "
                    f"({type(e).__name__}): collection={vector_store.collection_name} "
                    f"chunk_count={len(ids)} — {e}",
                    exc_info=True,
                )
                raise

            # ── Mark searchable ─────────────────────────────────────────────
            document.embed_done      = True
            document.chunk_count     = len(chunks_data)
            document.raw_chunks_json = None   # free space
            await db.commit()
            logger.info(
                f"[embed_worker] doc={msg.document_id} embedded "
                f"{len(chunks_data)} chunks → embed_done"
            )
            await check_and_finalize(document, db)

        except Exception as e:
            logger.error(f"[embed_worker] doc={msg.document_id} FAILED: {e}", exc_info=True)
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await db.commit()
            raise
        finally:
            # Return cached GPU memory to PyTorch's allocator so the next
            # worker (KG) can use the freed blocks.  This is a best-effort
            # hint — PyTorch may still hold the CUDA context until process exit.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
