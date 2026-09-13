"""
Phase 0 / B4 — source snapshot deduplication regression test.

Per F.3/O69: sources event MUST be cumulative deduplicated snapshot.
Identity = (document_id, page_or_chunk, content_hash).
Multi-source same content PRESERVED (provenance).

The SourcesSnapshotAccumulator class accumulates sources across multiple
push_event calls and deduplicates at terminal complete emission.
"""
from __future__ import annotations

from app.services.agent.sources_accumulator import SourcesSnapshotAccumulator, Source


def test_multiple_rounds_accumulate_without_loss():
    """Sources from multiple rounds should accumulate without loss."""
    acc = SourcesSnapshotAccumulator()
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", document_id="doc_A"),
    ])
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", document_id="doc_A"),
        Source(doc="B", chunk="p.2", content_hash="h2", document_id="doc_B"),
    ])
    sources = acc.deduplicated()
    assert len(sources) == 2
    doc_ids = {s.document_id or s.doc for s in sources}
    assert "doc_A" in doc_ids
    assert "doc_B" in doc_ids


def test_duplicate_identity_dedup():
    """Same source appearing twice should be deduplicated."""
    acc = SourcesSnapshotAccumulator()
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", document_id="doc_A"),
    ])
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", document_id="doc_A"),
    ])
    sources = acc.deduplicated()
    assert len(sources) == 1


def test_multi_source_same_content_preserved():
    """Same content from different sources (different source_id) should be preserved."""
    acc = SourcesSnapshotAccumulator()
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", source_id="src1", document_id="doc_A"),
        Source(doc="A", chunk="p.1", content_hash="h1", source_id="src2", document_id="doc_A"),
    ])
    sources = acc.deduplicated()
    assert len(sources) == 2  # provenance preserved


def test_clear_resets_accumulator():
    """Clear should reset the accumulator for rollback."""
    acc = SourcesSnapshotAccumulator()
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", document_id="doc_A"),
    ])
    assert len(acc.deduplicated()) == 1
    acc.clear()
    assert len(acc.deduplicated()) == 0


def test_empty_accumulator():
    """Empty accumulator should return empty list."""
    acc = SourcesSnapshotAccumulator()
    assert acc.deduplicated() == []


def test_accumulator_accepts_a_chat_source_chunk_object():
    """The v1 SSE path emits ChatSourceChunk objects; the accumulator must not read .chunk."""
    from app.schemas.rag import ChatSourceChunk

    chunk = ChatSourceChunk(
        index="id12",
        chunk_id="chunk_1",
        content="nội dung điều 17",
        document_id="11111111-1111-1111-1111-111111111111",
        source_file="luat.pdf",
        document_number="12/2020/NĐ-CP",
    )
    acc = SourcesSnapshotAccumulator()
    acc.add([Source.from_chat_source_chunk(chunk)])
    sources = acc.deduplicated()
    assert len(sources) == 1
    assert sources[0].chunk == "nội dung điều 17"
    assert sources[0].document_id == "11111111-1111-1111-1111-111111111111"
    assert sources[0].source_id == "id12"
