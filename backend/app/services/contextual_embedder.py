"""
Contextual Embedder
====================
Implements Anthropic's "Contextual Retrieval" technique:
  https://www.anthropic.com/engineering/contextual-retrieval

For each chunk, a small LLM (Qwen3-4B via memory agent) generates a 1-2 sentence
context that situates the chunk within the full document.  The generated context is
prepended to the chunk content before embedding, giving the vector model richer
signal about *where* and *why* the chunk matters.

Result per chunk:
    "<generated context>\n<original chunk content>"

The original content is preserved separately so it can still be displayed verbatim
in the UI — only the embedding text changes.

Performance (per Anthropic benchmarks):
  - Contextual Embeddings alone   → -35% retrieval failure rate
  - + BM25 hybrid                 → -49%
  - + reranking                   → -67%

Usage:
    from app.services.contextual_embedder import enrich_chunks_with_context

    enriched_texts = await enrich_chunks_with_context(
        document_markdown=full_markdown,
        chunks=[{"content": "...", "chunk_index": 0, ...}, ...],
    )
    # enriched_texts[i] is the text to embed for chunks[i]
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Prompt adapted for Vietnamese administrative / legal documents.
# The <document> block is the *same* for every chunk in one document → on
# remote APIs this would be cached; with local Qwen3-4B the KV-cache reuse
# within a batch helps too, though it is not guaranteed.
_CONTEXT_SYSTEM_PROMPT = (
    "Bạn là trợ lý hỗ trợ tìm kiếm tài liệu. "
    "Nhiệm vụ: viết 1-2 câu ngắn gọn mô tả đoạn văn bản dưới đây nằm ở phần nào "
    "của tài liệu, nhằm giúp tìm kiếm chính xác hơn. "
    "Chỉ trả lời phần mô tả, không giải thích thêm, không lặp lại nội dung chunk."
)

_CONTEXT_USER_TEMPLATE = (
    "<document>\n"
    "<title>{document_title}</title>\n"
    "<type>{document_type}</type>\n"
    "<number>{document_number}</number>\n"
    "<issued_by>{issuing_agency}</issued_by>\n"
    "<date>{published_date}</date>\n"
    "{document_preview}\n"
    "</document>\n\n"
    "Đây là đoạn văn bản cần được định vị:\n"
    "<chunk>\n{chunk_content}\n</chunk>\n\n"
    "Viết mô tả ngắn (1-2 câu) định vị đoạn này trong tài liệu."
)

# How much of the document to send as context — enough to capture the
# title / header / issuing authority, but cheap to tokenise.
_DOCUMENT_PREVIEW_CHARS = 6000


async def _generate_context_for_chunk(
    llm,
    document_title: str,
    document_type: str,
    document_number: str,
    issuing_agency: str,
    published_date: str,
    document_preview: str,
    chunk_content: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Call the memory-agent LLM to generate a situating sentence for one chunk.
    Returns the context string, or empty string on failure (caller uses fallback).
    """
    from app.services.llm.types import LLMMessage

    user_content = _CONTEXT_USER_TEMPLATE.format(
        document_title=document_title,
        document_type=document_type,
        document_number=document_number,
        issuing_agency=issuing_agency,
        published_date=published_date,
        document_preview=document_preview,
        chunk_content=chunk_content[:800],   # cap chunk to keep prompt small
    )

    async with semaphore:
        try:
            result = await llm.acomplete(
                [LLMMessage(role="user", content=user_content)],
                system_prompt=_CONTEXT_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            context = result if isinstance(result, str) else result.content
            context = (context or "").strip()
            # Sanity-check: reject if the model hallucinated something too long
            if len(context) > max_tokens * 6:   # ~6 chars/token heuristic
                logger.debug("[contextual_embedder] context too long, discarding")
                return ""
            return context
        except Exception as err:
            logger.warning(f"[contextual_embedder] LLM call failed (non-fatal): {err}")
            return ""


async def enrich_chunks_with_context(
    document_markdown: str,
    chunks: Sequence[dict],
    document_title: str = "",
    document_type: str = "",
    document_number: str = "",
    issuing_agency: str = "",
    published_date: str = "",
    max_tokens: int | None = None,
    concurrency: int | None = None,
) -> list[str]:
    """
    Generate contextual text for every chunk and return the enriched embed texts.

    Args:
        document_markdown: Full parsed markdown of the document.
        chunks:            List of chunk dicts (each has at least a "content" key).
        document_title:    Title of the document for additional context.
        document_type:     Type of the document (e.g. "Công văn", "Nghị định").
        document_number:   Document number (e.g. "123/BCD-TTHT").
        issuing_agency:   Agency that issued the document.
        published_date:    Publication/issue date of the document.
        max_tokens:        Max tokens for generated context.  Defaults to settings value.
        concurrency:       Max parallel LLM calls.  Defaults to settings value.

    Returns:
        List of enriched strings (same length as chunks).
        enriched[i] = "<context sentence>\n<original chunk content>"
        Falls back to the original content when LLM fails for that chunk.
    """
    from app.core.config import settings
    from app.services.llm import get_memory_agent

    _max_tokens  = max_tokens  or settings.HRAG_CONTEXTUAL_MAX_TOKENS
    _concurrency = concurrency or settings.HRAG_CONTEXTUAL_CONCURRENCY

    if not chunks:
        return []

    llm = get_memory_agent()
    document_preview = document_markdown[:_DOCUMENT_PREVIEW_CHARS].strip()
    semaphore = asyncio.Semaphore(_concurrency)

    tasks = [
        _generate_context_for_chunk(
            llm=llm,
            document_title=document_title,
            document_type=document_type,
            document_number=document_number,
            issuing_agency=issuing_agency,
            published_date=published_date,
            document_preview=document_preview,
            chunk_content=c["content"],
            max_tokens=_max_tokens,
            semaphore=semaphore,
        )
        for c in chunks
    ]

    contexts: list[str] = await asyncio.gather(*tasks)

    enriched: list[str] = []
    success = 0
    for chunk, ctx in zip(chunks, contexts):
        original = chunk["content"]
        if ctx:
            enriched.append(f"{ctx}\n{original}")
            success += 1
        else:
            enriched.append(original)   # fallback: embed original content

    logger.info(
        f"[contextual_embedder] {success}/{len(chunks)} chunks enriched with context"
    )
    return enriched
