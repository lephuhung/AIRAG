"""
Embed Worker
============
Consumes nexusrag.embed queue.

Responsibilities:
  1. Load raw_chunks_json from DB
  2. Embed all chunks with bge-m3
  3. Store in ChromaDB
  4. Set embed_done=True  →  status = INDEXED_PARTIAL (searchable now)
  5. Clear raw_chunks_json to free DB space
  6. Check if fully INDEXED
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import select

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

            # ── Embed ───────────────────────────────────────────────────────
            embedder     = get_embedding_service()
            vector_store = get_vector_store(msg.workspace_id)

            texts = [c["content"] for c in chunks_data]
            embeddings = embedder.embed_texts(texts)

            ids = [
                f"doc_{msg.document_id}_chunk_{c['chunk_index']}"
                for c in chunks_data
            ]
            img_url_prefix = f"/static/doc-images/kb_{msg.workspace_id}/images"
            metadatas = [
                {
                    "document_id":  msg.document_id,
                    "chunk_index":  c["chunk_index"],
                    "source":       c["source_file"],
                    "file_type":    document.file_type,
                    "page_no":      c["page_no"],
                    "heading_path": " > ".join(c["heading_path"]) if c["heading_path"] else "",
                    "has_table":    c["has_table"],
                    "has_code":     c["has_code"],
                    "image_ids":    "|".join(c["image_refs"]) if c["image_refs"] else "",
                    "table_ids":    "|".join(c["table_refs"]) if c["table_refs"] else "",
                    "image_urls":   "|".join(
                        f"{img_url_prefix}/{iid}.png" for iid in c["image_refs"]
                    ) if c["image_refs"] else "",
                }
                for c in chunks_data
            ]

            vector_store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            # ── Mark searchable ─────────────────────────────────────────────
            document.embed_done      = True
            document.chunk_count     = len(chunks_data)
            document.raw_chunks_json = None   # free space
            document.status          = DocumentStatus.INDEXED_PARTIAL
            await db.commit()
            logger.info(
                f"[embed_worker] doc={msg.document_id} embedded "
                f"{len(chunks_data)} chunks → INDEXED_PARTIAL"
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
