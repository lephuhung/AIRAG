"""
SourcesSnapshotAccumulator — cumulative deduplicated sources snapshot.

Per Phase 0 B4 contract: sources events MUST be cumulative deduplicated.
Identity = (document_id, page_or_chunk, content_hash).
Multi-source same content is PRESERVED (provenance).

This module is used by streaming.py to accumulate sources across multiple
push_event calls and deduplicate at terminal complete emission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Source:
    """Source chunk from a document section."""
    doc: str
    chunk: str
    content_hash: str
    source_id: Optional[str] = None
    document_id: Optional[str] = None

    @classmethod
    def from_chat_source_chunk(cls, chunk: Any) -> "Source":
        """Build a dedup identity from a ``ChatSourceChunk``-shaped object.

        ``ChatSourceChunk`` exposes ``document_id``/``chunk_id``/``content``/
        ``source_file``/``document_number``/``index`` but NOT ``chunk``/
        ``content_hash``/``source_id``; reading it directly was the v1 A/B
        ``AttributeError``. ``content`` owns the dedup text, ``chunk_id`` is the
        stable identity, and ``index`` discriminates provenance.
        """
        import hashlib

        content = getattr(chunk, "content", "") or ""
        content_hash = getattr(chunk, "chunk_id", "") or hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:16]
        document_id = str(getattr(chunk, "document_id", "") or "")
        doc = (
            getattr(chunk, "source_file", None)
            or getattr(chunk, "document_number", None)
            or document_id
        )
        return cls(
            doc=doc,
            chunk=content,
            content_hash=content_hash,
            source_id=getattr(chunk, "index", None),
            document_id=document_id,
        )


class SourcesSnapshotAccumulator:
    """Cumulative deduplicated sources snapshot per B4 contract.

    Identity = (document_id, page_or_chunk, content_hash).
    Multi-source same content PRESERVED (different source_id).
    """

    def __init__(self):
        self._by_id: dict[tuple, Source] = {}

    def add(self, sources: list[Source]) -> None:
        """Add sources to the accumulator, deduplicating by identity."""
        for s in sources:
            # Key INCLUDES source_id so multi-source same content is preserved.
            # If source_id is None (e.g., simple lookup), uses "" as discriminator.
            key = (s.document_id or s.doc, s.chunk, s.content_hash, s.source_id or "")
            if key not in self._by_id:
                self._by_id[key] = s

    def deduplicated(self) -> list[Source]:
        """Return the deduplicated list of sources."""
        return list(self._by_id.values())

    def clear(self) -> None:
        """Clear all accumulated sources (for rollback)."""
        self._by_id.clear()
