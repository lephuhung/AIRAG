"""Server-owned Markdown rendering for grounded claims (spec §11.5).

Each grounded claim renders exactly once; citation markers are inserted from
the claim's resolved public citations immediately before terminal sentence
punctuation — ``[a3z9][b2m7]``, never ``[a3z9, b2m7]``, never with a leading
space, and never followed by a references list. Because rendering happens
after grounding, bullet prefixes and Markdown structure cannot break claim
identity.

Presentation structure: summary claims form the opening paragraph(s),
detail claims render as bullets, and caveat claims render as bounded
``Lưu ý`` items — in that order regardless of claim order.
"""
from __future__ import annotations

import re

from ..contracts.synthesis import GroundedClaim
from .citations import CitationProjection

__all__ = ["RenderError", "render_grounded_claims"]

#: Terminal sentence punctuation the markers sit immediately before.
_TERMINAL_PUNCT_RE = re.compile(r"[.!?…。？！]+$")

#: Bounded caveat prefix (Vietnamese product language).
_CAVEAT_PREFIX = "**Lưu ý:** "


class RenderError(ValueError):
    """The grounded claims cannot be rendered with their projection."""


def _insert_markers(text: str, indexes: tuple[str, ...]) -> str:
    """Insert ``[idx]`` markers immediately before terminal punctuation."""
    markers = "".join(f"[{index}]" for index in indexes)
    stripped = text.rstrip()
    match = _TERMINAL_PUNCT_RE.search(stripped)
    if match is None:
        return f"{stripped}{markers}"
    return f"{stripped[: match.start()]}{markers}{match.group(0)}"


def render_grounded_claims(
    claims: tuple[GroundedClaim, ...], projection: CitationProjection
) -> str:
    """Render the grounded claims once, with inline citation markers.

    :raises RenderError: a claim has no projected indexes — rendering can
        never fabricate or drop citation identity.
    """
    summaries: list[str] = []
    details: list[str] = []
    caveats: list[str] = []
    for claim in claims:
        indexes = projection.claim_indexes.get(claim.claim_id)
        if not indexes:
            raise RenderError(
                f"grounded claim {claim.claim_id!r} has no projected citations"
            )
        line = _insert_markers(claim.text, indexes)
        if claim.presentation == "summary":
            summaries.append(line)
        elif claim.presentation == "caveat":
            caveats.append(f"- {_CAVEAT_PREFIX}{line}")
        else:
            details.append(f"- {line}")

    blocks: list[str] = []
    if summaries:
        blocks.append("\n\n".join(summaries))
    if details:
        blocks.append("\n".join(details))
    if caveats:
        blocks.append("\n".join(caveats))
    return "\n\n".join(blocks)
