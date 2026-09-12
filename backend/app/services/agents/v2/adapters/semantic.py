"""Semantic port: legacy query meaning → v2 draft/finalizer contracts (spec §8.3, §8.4).

This is a **server-internal** port, not the agent-facing tool gateway. It
translates two legacy shapes into canonical v2 contracts:

- a validated Phase-1 ``PreprocessingResult`` produced by the deterministic
  preprocessor;
- the compact persisted payload stored on ``chat_messages.semantic_context``
  (``to_persisted_dict`` output).

Ownership rule (spec §8.3, §26): ``RequestContext.original_query`` is the sole
authoritative raw query. ``SemanticDraft`` and ``SemanticContext`` have no
``original_query`` field, and this adapter never copies the raw query into
either: when the legacy result has no normalized form the adapter refuses and
asks its caller for the contextualized query instead of falling back.

Loading rule (spec §25): the legacy ``from_persisted_dict`` uses
``model_construct()`` to bypass validation because the compact payload drops
span offsets. This adapter does not: it validates the persisted payload
explicitly (version, key set, field types, enum membership) and translates it
directly into typed v2 contracts, so an incompatible pre-release payload is
rejected rather than construct-bypassed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from app.services.agents.semantic_preprocessor import (
    AbbreviationEntry,
    BlockingAmbiguity as LegacyBlockingAmbiguity,
    DocumentRefEntry,
    PreprocessingResult,
)

from ..contracts.base import CONTRACT_VERSION
from ..contracts.binding import DocumentBindingSet
from ..contracts.semantic import (
    AbbreviationResolution,
    BlockingAmbiguity,
    DocumentReference,
    SemanticContext,
    SemanticDraft,
    SemanticSnapshot,
)
from ..contracts.validation import validate_binding_set, validate_semantic_context
from .document import ResolvedDocumentBindings

#: The only persisted legacy semantic-context schema this adapter understands.
PERSISTED_SEMANTIC_VERSION = "1.0"

_LEGACY_PERSISTED_KEYS = frozenset(
    {
        "version",
        "original_query",
        "normalized_query",
        "preprocessing_status",
        "abbreviations",
        "document_refs",
        "blocking_ambiguities",
    }
)
_LEGACY_PREPROCESSING_STATUSES = frozenset({"ok", "partial", "complete", "error"})
_LEGACY_ABBREVIATION_STATUSES = frozenset(
    {"resolved", "ambiguous", "unknown", "not_in_db"}
)
_LEGACY_RESOLUTION_STATUSES = frozenset(
    {"resolved", "ambiguous", "not_found", "deferred", "error"}
)


class SemanticAdapterError(ValueError):
    """A legacy semantic payload cannot be translated into v2 contracts."""


@dataclass(frozen=True)
class FinalizedSemantics:
    """The Semantic Finalizer's output.

    ``snapshot`` is the persisted semantic boundary; ``bindings`` is the Binding
    Resolver's checkpoint projection. Both are validated together because the
    revision-requirement relation can only be checked against the canonical
    document references.
    """

    snapshot: SemanticSnapshot
    bindings: DocumentBindingSet


# ---------------------------------------------------------------------------
# Shared translation helpers
# ---------------------------------------------------------------------------


def _map_resolution_status(status: str) -> str:
    """Legacy resolution status → v2 ``DocumentResolutionStatus``.

    ``deferred`` means "lookup has not completed", which is exactly the v2
    ``unresolved`` state; no new v2 status is invented.
    """
    if status in ("resolved", "ambiguous", "not_found", "error"):
        return status
    if status == "deferred":
        return "unresolved"
    raise SemanticAdapterError(f"unsupported legacy reference status {status!r}")


def _abbreviations(
    entries: Sequence[AbbreviationEntry],
) -> tuple[AbbreviationResolution, ...]:
    return tuple(
        AbbreviationResolution(
            abbreviation=entry.span, expansion=entry.chosen
        )
        for entry in entries
    )


def _blocking_ambiguities(
    ambiguities: Sequence[LegacyBlockingAmbiguity],
) -> tuple[BlockingAmbiguity, ...]:
    translated: list[BlockingAmbiguity] = []
    for index, ambiguity in enumerate(ambiguities):
        if not ambiguity.essential:
            continue
        translated.append(
            BlockingAmbiguity(
                ambiguity_id=ambiguity.source_ref or f"ambiguity-{index + 1}",
                description=ambiguity.description,
            )
        )
    return tuple(translated)


def _legacy_reference(reference: DocumentRefEntry) -> DocumentReference:
    status = _map_resolution_status(reference.resolution_status)
    resolved_document_id = reference.document_handle
    if status == "resolved" and resolved_document_id is None:
        raise SemanticAdapterError(
            f"legacy reference {reference.ref_id!r} is resolved but carries no "
            "server-validated document handle"
        )
    if status != "resolved":
        resolved_document_id = None
    candidates = (
        tuple(
            candidate.document_id
            for candidate in reference.candidates
            if candidate.document_id is not None
        )
        if status == "ambiguous"
        else ()
    )
    return DocumentReference(
        ref_id=reference.ref_id,
        original_span=reference.original_span,
        normalized_reference=reference.reference,
        requested_role=None,
        revision_requirement=None,
        resolution_status=status,
        resolved_document_id=resolved_document_id,
        candidate_document_ids=candidates,
    )


# ---------------------------------------------------------------------------
# Validated PreprocessingResult → draft
# ---------------------------------------------------------------------------


def draft_from_preprocessing(
    result: PreprocessingResult,
    *,
    contextualized_query: str | None = None,
) -> SemanticDraft:
    """Translate a validated legacy preprocessor result into a semantic draft.

    ``contextualized_query`` lets the caller supply the discourse-contextualized
    form (the legacy preprocessor does not resolve coreference). It is never
    defaulted to ``result.original_query``: if no normalized/contextualized form
    exists the adapter refuses rather than copying the raw query.
    """
    base = contextualized_query if contextualized_query is not None else result.normalized_query
    if base is None or not base.strip():
        raise SemanticAdapterError(
            "legacy PreprocessingResult carries no normalized query; supply "
            "contextualized_query instead of copying the raw query"
        )
    return SemanticDraft(
        provisional_contextualized_query=base,
        abbreviations=_abbreviations(result.abbreviations),
        coreferences=(),
        document_refs=tuple(_legacy_reference(ref) for ref in result.document_refs),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=_blocking_ambiguities(result.blocking_ambiguities),
    )


# ---------------------------------------------------------------------------
# Persisted legacy semantic payload → draft (explicit validation, no bypass)
# ---------------------------------------------------------------------------


def _record_list(value: object, field: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SemanticAdapterError(f"persisted semantic {field} must be a list")
    records: list[Mapping[str, object]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise SemanticAdapterError(
                f"persisted semantic {field} entries must be objects"
            )
        records.append(entry)
    return records


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticAdapterError(f"persisted semantic {field} must be a non-blank string")
    return value


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticAdapterError(f"persisted semantic {field} must be a string or null")
    return value


def _require_enum(value: object, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SemanticAdapterError(
            f"persisted semantic {field} must be one of {sorted(allowed)}"
        )
    return value


def _optional_uuid(value: object, field: str) -> UUID | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticAdapterError(f"persisted semantic {field} must be a uuid string or null")
    try:
        return UUID(value)
    except ValueError as exc:
        raise SemanticAdapterError(
            f"persisted semantic {field} is not a valid uuid: {value!r}"
        ) from exc


def _persisted_reference(entry: Mapping[str, object]) -> DocumentReference:
    reference = _require_str(entry.get("reference"), "document_refs.reference")
    status = _map_resolution_status(
        _require_enum(
            entry.get("resolution_status"),
            "document_refs.resolution_status",
            _LEGACY_RESOLUTION_STATUSES,
        )
    )
    document_id = _optional_uuid(entry.get("document_handle"), "document_refs.document_handle")
    if status == "resolved" and document_id is None:
        raise SemanticAdapterError(
            "persisted semantic resolved document_refs entry has no document handle"
        )
    if status != "resolved":
        document_id = None
    return DocumentReference(
        ref_id=_require_str(entry.get("ref_id"), "document_refs.ref_id"),
        original_span=reference,
        normalized_reference=reference,
        requested_role=None,
        revision_requirement=None,
        resolution_status=status,
        resolved_document_id=document_id,
        candidate_document_ids=(),
    )


def draft_from_persisted_semantic(payload: Mapping[str, object]) -> SemanticDraft:
    """Validate and translate the compact persisted legacy semantic payload.

    The raw ``original_query`` stored alongside the payload is deliberately
    ignored (``RequestContext`` owns it). Only the normalized form feeds the
    draft, so it is required.
    """
    if not isinstance(payload, Mapping):
        raise SemanticAdapterError("persisted semantic payload must be a mapping")
    unknown = set(payload) - _LEGACY_PERSISTED_KEYS
    if unknown:
        raise SemanticAdapterError(
            f"persisted semantic payload has unsupported keys: {sorted(unknown)}"
        )
    version = payload.get("version")
    if version != PERSISTED_SEMANTIC_VERSION:
        raise SemanticAdapterError(
            f"persisted semantic payload declares version {version!r}; only "
            f"{PERSISTED_SEMANTIC_VERSION!r} is supported and incompatible "
            "payloads are rejected rather than migrated"
        )
    _optional_str(payload.get("original_query"), "original_query")
    normalized_query = _require_str(payload.get("normalized_query"), "normalized_query")
    _require_enum(
        payload.get("preprocessing_status"),
        "preprocessing_status",
        _LEGACY_PREPROCESSING_STATUSES,
    )

    abbreviations: list[AbbreviationResolution] = []
    for entry in _record_list(payload.get("abbreviations"), "abbreviations"):
        _optional_str(entry.get("short_form"), "abbreviations.short_form")
        _require_enum(
            entry.get("status"),
            "abbreviations.status",
            _LEGACY_ABBREVIATION_STATUSES,
        )
        abbreviations.append(
            AbbreviationResolution(
                abbreviation=_require_str(entry.get("span"), "abbreviations.span"),
                expansion=_optional_str(entry.get("chosen"), "abbreviations.chosen"),
            )
        )

    ambiguities: list[BlockingAmbiguity] = []
    for index, entry in enumerate(
        _record_list(payload.get("blocking_ambiguities"), "blocking_ambiguities")
    ):
        source_ref = _optional_str(entry.get("source_ref"), "blocking_ambiguities.source_ref")
        if not isinstance(entry.get("essential"), bool):
            raise SemanticAdapterError(
                "persisted semantic blocking_ambiguities.essential must be a boolean"
            )
        if not entry["essential"]:
            continue
        ambiguities.append(
            BlockingAmbiguity(
                ambiguity_id=source_ref or f"ambiguity-{index + 1}",
                description=_require_str(
                    entry.get("description"), "blocking_ambiguities.description"
                ),
            )
        )

    return SemanticDraft(
        provisional_contextualized_query=normalized_query,
        abbreviations=tuple(abbreviations),
        coreferences=(),
        document_refs=tuple(
            _persisted_reference(entry)
            for entry in _record_list(payload.get("document_refs"), "document_refs")
        ),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=tuple(ambiguities),
    )


# ---------------------------------------------------------------------------
# Draft + bindings → semantic finalizer
# ---------------------------------------------------------------------------


def finalize_semantic_context(
    draft: SemanticDraft,
    *,
    contextualized_query: str | None = None,
    normalized_query: str | None = None,
    document_refs: tuple[DocumentReference, ...] | None = None,
) -> SemanticContext:
    """Produce the finalized, validated query meaning from a draft."""
    contextualized = (
        contextualized_query
        if contextualized_query is not None
        else draft.provisional_contextualized_query
    )
    normalized = normalized_query if normalized_query is not None else contextualized
    context = SemanticContext(
        contextualized_query=contextualized,
        normalized_query=normalized,
        abbreviations=draft.abbreviations,
        coreferences=draft.coreferences,
        document_refs=document_refs if document_refs is not None else draft.document_refs,
        person_refs=draft.person_refs,
        section_refs=draft.section_refs,
        blocking_ambiguities=draft.preliminary_ambiguities,
    )
    validate_semantic_context(context)
    return context


def finalize_semantics(
    draft: SemanticDraft,
    resolved: ResolvedDocumentBindings,
    *,
    contextualized_query: str | None = None,
    normalized_query: str | None = None,
) -> FinalizedSemantics:
    """Finalize the canonical semantics and validate the paired binding set."""
    semantic = finalize_semantic_context(
        draft,
        contextualized_query=contextualized_query,
        normalized_query=normalized_query,
        document_refs=resolved.references,
    )
    validate_binding_set(resolved.binding_set, semantic)
    return FinalizedSemantics(
        snapshot=SemanticSnapshot(contract_version=CONTRACT_VERSION, semantic=semantic),
        bindings=resolved.binding_set,
    )


# ``AbbreviationEntry`` is referenced for the typed signature above; re-exported
# so Phase 2 can type its preprocessor boundary without importing legacy modules
# directly.
__all__ = [
    "FinalizedSemantics",
    "PERSISTED_SEMANTIC_VERSION",
    "SemanticAdapterError",
    "AbbreviationEntry",
    "draft_from_persisted_semantic",
    "draft_from_preprocessing",
    "finalize_semantic_context",
    "finalize_semantics",
]
