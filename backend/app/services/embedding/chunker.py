"""
Text Chunking Service
Handles splitting documents into smaller chunks for embedding and retrieval.
"""
from __future__ import annotations

from typing import NamedTuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunk(NamedTuple):
    """Represents a chunk of text with metadata."""
    content: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict


class DocumentChunker:
    """
    Splits documents into chunks using recursive character-based splitting.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            separators: Custom separators for splitting (optional)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators,
        )

    def split_text(
        self,
        text: str,
        source: str = "",
        extra_metadata: dict | None = None
    ) -> list[TextChunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: The text content to split
            source: Source identifier (e.g., filename)
            extra_metadata: Additional metadata to include with each chunk

        Returns:
            List of TextChunk objects with content and metadata
        """
        if not text.strip():
            return []

        # Use LangChain splitter
        chunks = self._splitter.split_text(text)

        result = []
        current_pos = 0

        for i, chunk_content in enumerate(chunks):
            # Find the actual position in the original text
            # This is approximate due to overlap handling
            start_pos = text.find(chunk_content[:50], current_pos)
            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(chunk_content)

            metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(extra_metadata or {})
            }

            result.append(TextChunk(
                content=chunk_content,
                chunk_index=i,
                char_start=start_pos,
                char_end=end_pos,
                metadata=metadata
            ))

            # Update position for next search (accounting for overlap)
            current_pos = max(start_pos + len(chunk_content) - self.chunk_overlap, current_pos + 1)

        return result

    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate the number of chunks without actually splitting.

        Args:
            text: The text to estimate

        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0

        text_length = len(text)
        effective_chunk = self.chunk_size - self.chunk_overlap

        if effective_chunk <= 0:
            return 1

        return max(1, (text_length + effective_chunk - 1) // effective_chunk)


# Default chunker instance
default_chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)


def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[TextChunk]:
    """
    Convenience function to chunk text with default or custom settings.

    Args:
        text: Text to chunk
        source: Source identifier
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        List of TextChunk objects
    """
    if chunk_size == 500 and chunk_overlap == 50:
        return default_chunker.split_text(text, source)

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_text(text, source)


class LegalDocumentChunker:
    """Chunker theo CẤU TRÚC văn bản pháp luật VN (Phần/Chương/Mục/Điều).

    ``DocumentChunker`` cắt theo kích thước nên một chunk thường trộn cuối điều
    này với đầu điều kia — retrieval theo điều/khoản trả nửa nội dung lạc đề.
    Chunker này cắt tại RANH GIỚI heading cấu trúc trước (mỗi Điều là một đơn
    vị trọn vẹn, giữ nguyên phần mở đầu/phụ lục), điều quá dài mới size-split
    tiếp bằng RecursiveCharacterTextSplitter bên trong đúng điều đó.

    Cùng contract ``split_text() -> list[TextChunk]`` với DocumentChunker để
    dùng thay thế trực tiếp trong ``_parse_legacy``. ``char_start``/``char_end``
    chính xác tuyệt đối (slice trực tiếp, không dò find()) — phép gán page_no
    theo page marker giữ nguyên độ tin cậy.
    """

    # Dưới ngưỡng này coi như văn bản KHÔNG có cấu trúc luật (công văn, tờ
    # trình...) → caller nên dùng DocumentChunker thường.
    MIN_DIEU_HEADINGS = 3

    def __init__(self, max_chars: int = 1800, sub_overlap: int = 150):
        self.max_chars = max_chars
        self._sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=sub_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @classmethod
    def has_legal_structure(cls, text: str) -> bool:
        from app.services.parsing.heading_path import find_headings

        dieu = [h for h in find_headings(text) if h.title.lower().startswith("điều")]
        return len(dieu) >= cls.MIN_DIEU_HEADINGS

    def _sub_split(
        self, section: str, base: int, start: int, end: int
    ) -> list[tuple[str, int]]:
        """Size-split một khoảng [start, end) của ``section`` — không bao giờ
        tràn sang khoản/điều kế tiếp. Trả (content, absolute_char_start)."""
        seg = section[start:end]
        pieces: list[tuple[str, int]] = []
        pos = 0
        for sub in self._sub_splitter.split_text(seg):
            at = seg.find(sub[:50], pos)
            if at == -1:
                at = pos
            pieces.append((sub, base + start + at))
            pos = max(at + 1, pos + 1)
        return pieces

    def _split_oversized_section(self, section: str, base: int) -> list[tuple[str, int]]:
        """Cắt section quá dài theo KHOẢN trước, size-split chỉ khi một khoản
        đơn lẻ vượt max_chars. Không có marker khoản nào → fallback cũ."""
        from app.services.parsing.heading_path import find_subdivisions

        khoans = [sd for sd in find_subdivisions(section) if sd.kind == "khoan"]
        if not khoans:
            return self._sub_split(section, base, 0, len(section))

        segments: list[tuple[int, int]] = []
        if section[: khoans[0].start].strip():
            segments.append((0, khoans[0].start))
        for i, k in enumerate(khoans):
            end = khoans[i + 1].start if i + 1 < len(khoans) else len(section)
            segments.append((k.start, end))

        pieces: list[tuple[str, int]] = []
        buf: tuple[int, int] | None = None

        def flush() -> None:
            nonlocal buf
            if buf is not None:
                pieces.append((section[buf[0] : buf[1]], base + buf[0]))
                buf = None

        for st, en in segments:
            if en - st > self.max_chars:
                flush()
                pieces.extend(self._sub_split(section, base, st, en))
            elif buf is None:
                buf = (st, en)
            elif en - buf[0] <= self.max_chars:
                buf = (buf[0], en)
            else:
                flush()
                buf = (st, en)
        flush()
        return pieces

    def split_text(
        self,
        text: str,
        source: str = "",
        extra_metadata: dict | None = None,
    ) -> list[TextChunk]:
        from app.services.parsing.heading_path import (
            derive_heading_paths,
            derive_subdivision_metadata,
            find_headings,
        )

        if not text.strip():
            return []

        headings = find_headings(text)
        # Ranh giới section: đầu văn bản + vị trí từng heading. Section cuối
        # chạy tới hết văn bản. Preamble (trước heading đầu) là section riêng.
        bounds = [0] + [h.start for h in headings if h.start > 0] + [len(text)]
        bounds = sorted(set(bounds))

        result: list[TextChunk] = []
        for s, e in zip(bounds, bounds[1:]):
            section = text[s:e]
            if not section.strip():
                continue
            if len(section) <= self.max_chars:
                pieces = [(section, s)]
            else:
                pieces = self._split_oversized_section(section, s)
            for content, at in pieces:
                result.append(TextChunk(
                    content=content,
                    chunk_index=len(result),
                    char_start=at,
                    char_end=at + len(content),
                    metadata={
                        "source": source,
                        "chunk_index": len(result),
                        **(extra_metadata or {}),
                    },
                ))
        contents = [c.content for c in result]
        sub_metas = derive_subdivision_metadata(
            contents, derive_heading_paths(contents)
        )
        # total_chunks chỉ biết sau khi duyệt xong
        return [
            c._replace(metadata={
                **c.metadata,
                "total_chunks": len(result),
                "khoan_nos": list(m.khoan_nos),
                "diem_labels": list(m.diem_labels),
                "subdivision_refs": list(m.subdivision_refs),
                "subdivision_schema_version": 1,
            })
            for c, m in zip(result, sub_metas)
        ]
