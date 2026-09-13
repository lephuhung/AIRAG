"""
Embed Worker
============
Consumes hrag.embed queue.

Responsibilities:
  1. Load raw_chunks_json from DB
  2. (Optional) Contextual Embeddings: enrich each chunk with LLM-generated
     situating context before embedding (see HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS)
  3. Embed all chunks with the HRAG_EMBEDDING_MODEL (EmbeddingService)
  4. Store in ChromaDB
  5. Set embed_done=True  →  status = EMBEDDING (searchable now)
  6. Clear raw_chunks_json to free DB space
  7. Check if fully INDEXED via check_and_finalize
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401
from app.models.document import Document, DocumentStatus
from app.models.document_revision_build import DocumentRevisionBuild
from app.queue.messages import EmbedMessage
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
)
from app.services.agents.v2.persistence.document_views import (
    VECTOR_ARTIFACT_VERSION,
    embedding_namespace,
    revision_vector_id,
)
from app.services.embedding.embedder import get_embedding_service
from app.services.parsing.heading_path import extract_article_nos
from app.services.embedding.vector_store import get_vector_store
from app.workers.utils import (
    check_and_finalize,
    load_revision_chunk_payloads,
    load_revision_execution,
    mark_revision_failed,
    record_embed_artifacts,
)

logger = logging.getLogger(__name__)

#: Schema version of the persisted vector artifact — owned by the revision
#: artifact identity module so every producer/consumer agrees on one spelling.
_VECTOR_ARTIFACT_VERSION = VECTOR_ARTIFACT_VERSION


async def handle_embed(payload: dict) -> None:
    # Config watch first: pick up WebUI LLM changes before resolving any
    # provider — in particular BEFORE contextual enrichment below, where
    # contextual_embedder resolves the memory_agent LLM. Fail-open.
    from app.workers.config_watch import ensure_fresh_config
    await ensure_fresh_config()

    msg = EmbedMessage(**payload)
    logger.info(
        f"[embed_worker] doc={msg.document_id} rev={msg.revision_id} "
        f"profile={msg.build_profile}"
    )
    profile = RevisionBuildProfile(msg.build_profile)

    async with async_session_maker() as db:
        result = await db.execute(
            select(Document)
            .where(Document.id == msg.document_id)
            .with_for_update()
        )
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[embed_worker] doc={msg.document_id} not found")
            return

        # Revision-owned execution decision: terminal revision or tombstoned
        # source is a no-op dead-letter.
        execution = await load_revision_execution(
            db, revision_id=msg.revision_id, document_id=msg.document_id
        )
        if not execution.run:
            logger.info(
                f"[embed_worker] doc={msg.document_id} rev={msg.revision_id} "
                f"no-op ({execution.reason})"
            )
            return

        # Idempotency is revision-scoped: a redelivered message re-runs only
        # its incomplete stage. The build manifest, not Document.embed_done,
        # proves whether THIS revision's vectors already exist.
        already = await db.scalar(
            select(DocumentRevisionBuild.build_id).where(
                DocumentRevisionBuild.revision_id == msg.revision_id,
                DocumentRevisionBuild.build_profile == profile.value,
                DocumentRevisionBuild.embedding_namespace.is_not(None),
                DocumentRevisionBuild.vector_artifact_version.is_not(None),
            )
        )
        if already is not None:
            logger.info(
                f"[embed_worker] doc={msg.document_id} rev={msg.revision_id} "
                f"already has a vector manifest — skipping"
            )
            await check_and_finalize(
                document, db, revision_id=msg.revision_id
            )
            return

        try:
            # Set EMBEDDING at start
            document.status = DocumentStatus.EMBEDDING
            await db.commit()

            raw = document.raw_chunks_json
            # Prefer the revision's own structure artifact: ``raw_chunks_json``
            # is a document-level mirror that two overlapping revisions would
            # clobber, while the artifact is owned by this revision alone.
            chunks_data: list[dict] | None = await load_revision_chunk_payloads(
                db, msg.revision_id
            )
            if chunks_data is None:
                chunks_data = json.loads(raw) if raw else None
            if not chunks_data:
                logger.warning(f"[embed_worker] doc={msg.document_id} has no raw_chunks_json — skipping embed")
                document.embed_done = True
                await db.commit()
                await check_and_finalize(
                    document, db, revision_id=msg.revision_id
                )
                return

            # ── Strip OCR layout markup before vectorising ──────────────────
            # Scanned docs store administrative-layout HTML (alignment, 2-column
            # header, data-bbox) in their markdown/chunks so DocumentViewer can
            # render the original format. That markup is pure noise for the
            # embeddings + citation snippets, so remove it here. No-op for
            # non-OCR (Docling) documents — strip_ocr_layout only touches text
            # that actually carries the layout markers.
            from app.services.parsing.ocr_service import strip_ocr_layout
            for c in chunks_data:
                c["content"] = strip_ocr_layout(c["content"])

            # ── Contextual Embeddings (optional) ────────────────────────────
            # When enabled, each chunk is enriched with a short LLM-generated
            # sentence that situates it within the full document before embedding.
            # The original content is kept for display; only embed_texts changes.
            # See: https://www.anthropic.com/engineering/contextual-retrieval
            embed_texts = [c["content"] for c in chunks_data]
            if settings.HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS:
                try:
                    from app.services.embedding.contextual_embedder import enrich_chunks_with_context
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

                    # The stored markdown may be OCR layout HTML — feed the
                    # contextual LLM clean text, not coordinate markup.
                    document_markdown = strip_ocr_layout(document_markdown)

                    if document_markdown:
                        embed_texts = await enrich_chunks_with_context(
                            document_markdown=document_markdown,
                            chunks=chunks_data,
                            document_title=getattr(document, "document_title", "") or "",
                            document_type=getattr(document.document_type, "name", "") if document.document_type else "",
                            document_number=getattr(document, "document_number", "") or "",
                            issuing_agency=getattr(document, "issuing_agency", "") or "",
                            published_date=getattr(document, "published_date", "") or "",
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
            model_hash = hashlib.sha256(
                embedder.model_name.encode("utf-8")
            ).hexdigest()[:16]
            dimension = int(embedder.dimension)
            # The collection namespace is qualified by the embedding model hash
            # and dimension: a build under a new model/dimension writes a NEW
            # collection instead of deleting/recreating the one holding
            # published revisions' vectors.
            namespace = embedding_namespace(
                msg.workspace_id, model_hash, dimension
            )
            vector_store = get_vector_store(msg.workspace_id, namespace=namespace)

            # Pass ONLY non-empty texts to the embedder so the returned list is
            # aligned with valid_indices by construction (the embedder silently
            # drops empty strings, which would desync ids/embeddings otherwise).
            valid_indices = [i for i, t in enumerate(embed_texts) if t.strip()]
            embeddings = await asyncio.to_thread(
                embedder.embed_texts, [embed_texts[i] for i in valid_indices]
            )

            # ids, metadatas, documents must be aligned with the embeddings list
            ws_id = str(msg.workspace_id)
            img_url_prefix = f"/static/doc-images/kb_{ws_id}/images"
            ids = []
            documents = []
            metadatas = []
            for i in valid_indices:
                c = chunks_data[i]
                ordinal = c["chunk_index"]
                ids.append(revision_vector_id(msg.revision_id, ordinal))
                documents.append(c["content"])
                metadatas.append({
                    "document_id":     str(msg.document_id),
                    "workspace_id":   ws_id,
                    "revision_id":    str(msg.revision_id),
                    "chunk_id":       str(c.get("chunk_id") or ""),
                    "ordinal":        ordinal,
                    "chunk_index":     ordinal,
                    "source":          c["source_file"],
                    "file_type":       document.file_type,
                    "page_no":         c["page_no"],
                    "heading_path":    " > ".join(c["heading_path"]) if c["heading_path"] else "",
                    # Số Điều cấu trúc ("17|18") — tra cứu điều khoản chính xác
                    "article_nos":     "|".join(extract_article_nos(c["heading_path"])),
                    "has_table":       c["has_table"],
                    "has_code":        c["has_code"],
                    "image_ids":       "|".join(c["image_refs"]) if c["image_refs"] else "",
                    "table_ids":       "|".join(c["table_refs"]) if c["table_refs"] else "",
                    "image_urls":      "|".join(
                        f"{img_url_prefix}/{iid}.png" for iid in c["image_refs"]
                    ) if c["image_refs"] else "",
                    "document_number": c.get("document_number", document.document_number or ""),
                })

            try:
                vector_store.add_documents(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
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

            # ── Record the revision's vector manifest ────────────────────────
            await record_embed_artifacts(
                db,
                msg.revision_id,
                profile,
                embedding_namespace=vector_store.collection_name,
                embedding_model_hash=model_hash,
                embedding_dimension=dimension,
                vector_artifact_version=_VECTOR_ARTIFACT_VERSION,
            )
            await db.commit()
            logger.info(
                f"[embed_worker] doc={msg.document_id} rev={msg.revision_id} "
                f"recorded vector manifest for {vector_store.collection_name}"
            )

            # ── Mark searchable (v1/UI mirror only) ────────────────────────
            document.embed_done  = True
            document.chunk_count = len(chunks_data)
            if document.is_chat_upload:
                # No caption worker runs for chat uploads — safe to free now.
                document.raw_chunks_json = None
            else:
                # Keep the (OCR-stripped) chunks AND the contextual sentence for
                # the caption worker: its re-embed must rebuild
                # "context + content + captions", otherwise it would overwrite
                # the contextual vectors with context-less text.
                # check_and_finalize frees this column once embed+captions done.
                for i, c in enumerate(chunks_data):
                    t = embed_texts[i]
                    base = c["content"]
                    c["context"] = (
                        t[: -len(base)].rstrip("\n")
                        if base and t != base and t.endswith(base)
                        else ""
                    )
                document.raw_chunks_json = json.dumps(chunks_data)
            await db.commit()
            logger.info(
                f"[embed_worker] doc={msg.document_id} embedded "
                f"{len(chunks_data)} chunks → embed_done"
            )
            await check_and_finalize(
                document, db, revision_id=msg.revision_id
            )

        except Exception as e:
            logger.error(f"[embed_worker] doc={msg.document_id} FAILED: {e}", exc_info=True)
            # Session may be aborted if the failure came from a DB flush —
            # roll back first so the FAILED status can be written.
            await db.rollback()
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await db.commit()
            # Terminalize the revision too: a failed revision is immutable and
            # never resumed, so parallel caption/kg messages dead-letter.
            await mark_revision_failed(
                db, msg.revision_id, stage="embed", error_class=type(e).__name__
            )
            raise
        finally:
            # Return cached GPU memory to PyTorch's allocator so the next
            # worker (KG) can use the freed blocks.  This is a best-effort
            # hint — PyTorch may still hold the CUDA context until process exit.
            try:
                import os, torch
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                device_count = torch.cuda.device_count()
                is_available = torch.cuda.is_available()
                logger.debug(
                    f"[embed_worker] CUDA state: CUDA_VISIBLE_DEVICES={cuda_visible!r}, "
                    f"device_count={device_count}, is_available={is_available}"
                )
                if is_available and device_count > 0:
                    torch.cuda.empty_cache()
                    logger.debug(f"[embed_worker] GPU cache cleared")
                else:
                    logger.debug(f"[embed_worker] No CUDA GPU available for cache clear — skipping")
            except Exception as e:
                logger.debug(f"[embed_worker] GPU cache clear failed (non-fatal): {e}")
