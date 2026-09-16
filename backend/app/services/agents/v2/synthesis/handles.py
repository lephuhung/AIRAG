"""Stable ``E``-handle manifest for grounded LLM synthesis (spec §8.4).

Before the first provider call the server binds opaque ``E1..En`` handles to
exact ``EvidenceUseRef`` values in selected-evidence order. The manifest is
checkpointed inside ``SynthesisCheckpoint`` and carries no evidence plaintext;
the model only ever sees the opaque ``E``-labels.

Identity invariant: repair reuses the same manifest and numbering, and resume
never rebinds a handle to a different use — a denied ``E2`` fails rehydration
upstream rather than sliding onto the next surviving use. Resolution therefore
goes strictly by handle string through the checkpointed manifest, never by
position in a rebuilt selection.
"""
from __future__ import annotations

from typing import Iterable

from ..contracts.evidence import EvidenceUseRef
from ..contracts.synthesis import HandleManifestEntry

__all__ = [
    "HandleManifestError",
    "build_handle_manifest",
    "resolve_handle",
    "resolve_handles",
]

#: Closed failure codes (spec §16) — content-free by construction.
UNKNOWN_HANDLE = "unknown_evidence_handle"
MANIFEST_MISMATCH = "handle_manifest_mismatch"


class HandleManifestError(ValueError):
    """Closed manifest failure; ``code`` is safe to checkpoint/emit."""

    def __init__(self, code: str) -> None:
        if code not in (UNKNOWN_HANDLE, MANIFEST_MISMATCH):
            raise ValueError(f"not a closed manifest code: {code!r}")
        super().__init__(code)
        self.code = code


def _use_ref_of(item) -> EvidenceUseRef:
    """Accept an ``EvidenceUseRef`` or any hydrated item exposing ``use_id``."""
    if isinstance(item, EvidenceUseRef):
        return item
    use_id = getattr(item, "use_id", None)
    if use_id is None:
        raise HandleManifestError(MANIFEST_MISMATCH)
    return EvidenceUseRef(use_id=use_id)


def build_handle_manifest(items: Iterable) -> tuple[HandleManifestEntry, ...]:
    """Bind ``E1..En`` to the exact uses in deterministic selection order.

    ``items`` may be ``EvidenceUseRef`` values or hydrated evidence objects
    exposing ``use_id``; only the reference is stored — never content.
    """
    entries: list[HandleManifestEntry] = []
    seen: set = set()
    for item in items:
        use = _use_ref_of(item)
        if use.use_id in seen:
            raise HandleManifestError(MANIFEST_MISMATCH)
        seen.add(use.use_id)
        entries.append(
            HandleManifestEntry(handle=f"E{len(entries) + 1}", use=use)
        )
    return tuple(entries)


def _manifest_map(manifest: tuple[HandleManifestEntry, ...]) -> dict:
    """Validate manifest structure and return the handle -> entry map.

    A checkpointed manifest must be exactly ``E1..En`` in order with no gaps,
    duplicates, or foreign labels; anything else is a structural mismatch.
    """
    mapping: dict = {}
    for index, entry in enumerate(manifest):
        expected = f"E{index + 1}"
        if entry.handle != expected or entry.handle in mapping:
            raise HandleManifestError(MANIFEST_MISMATCH)
        mapping[entry.handle] = entry
    return mapping


def resolve_handle(
    handle: str, manifest: tuple[HandleManifestEntry, ...]
) -> EvidenceUseRef:
    """Resolve one opaque handle to its exact use through the manifest only."""
    entry = _manifest_map(manifest).get(handle)
    if entry is None:
        raise HandleManifestError(UNKNOWN_HANDLE)
    return entry.use


def resolve_handles(
    handles: Iterable[str], manifest: tuple[HandleManifestEntry, ...]
) -> tuple[EvidenceUseRef, ...]:
    """Resolve handles to exact uses, deduplicating while preserving order."""
    mapping = _manifest_map(manifest)
    resolved: list[EvidenceUseRef] = []
    seen: set = set()
    for handle in handles:
        if handle in seen:
            continue
        seen.add(handle)
        entry = mapping.get(handle)
        if entry is None:
            raise HandleManifestError(UNKNOWN_HANDLE)
        resolved.append(entry.use)
    return tuple(resolved)
