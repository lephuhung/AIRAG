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
from typing import Optional


@dataclass
class Source:
    """Source chunk from a document section."""
    doc: str
    chunk: str
    content_hash: str
    source_id: Optional[str] = None
    document_id: Optional[str] = None


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
