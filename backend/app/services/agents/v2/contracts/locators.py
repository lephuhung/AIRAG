"""Structured content locators (spec §10).

Stable ingestion structure IDs are the canonical coordinate system; human
readable headings are presentation data and never appear here. The union is
discriminated on ``kind`` and has no generic metadata escape hatch, so planning,
reading, coverage, evidence, and grounding all share one coordinate system.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import ContractModel


class DocumentLocator(ContractModel):
    """Whole-document coordinate."""

    kind: Literal["document"]


class SectionLocator(ContractModel):
    """A section addressed by its stable ingestion structure node ID."""

    kind: Literal["section"]
    structure_node_id: str


class ArticleLocator(ContractModel):
    """An article inside a structure node, addressed by stable IDs only."""

    kind: Literal["article"]
    structure_node_id: str
    article_id: str


class PageRangeLocator(ContractModel):
    """Inclusive page range."""

    kind: Literal["page_range"]
    start: int
    end: int


class ChunkRangeLocator(ContractModel):
    """Range over stable chunk identities."""

    kind: Literal["chunk_range"]
    start: str
    end: str


ContentLocator = Annotated[
    DocumentLocator | SectionLocator | ArticleLocator | PageRangeLocator | ChunkRangeLocator,
    Field(discriminator="kind"),
]
