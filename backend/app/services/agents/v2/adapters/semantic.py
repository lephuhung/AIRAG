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

from sqlalchemy.ext.asyncio import AsyncSession

from ..semantic.document_identity import (
    DocumentIdentityResolver,
    normalize_section_label,
)

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
    SectionReference,
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


def _section_refs_from_labels(labels: Sequence[str | None]) -> tuple[SectionReference, ...]:
    """Project authoritative one-turn section labels onto v2 section refs.

    Label-only by design (Phase 4B, Task 7): ``structure_node_id`` stays
    ``None`` because no revision is pinned here. Non-authoritative spans
    (multi-locator, discourse anaphora) are dropped for Phase 4C.
    """
    refs: list[SectionReference] = []
    for label in labels:
        normalized = normalize_section_label(label)
        if normalized is None:
            continue
        refs.append(
            SectionReference(
                ref_id=f"s{len(refs) + 1}",
                label=normalized,
                structure_node_id=None,
            )
        )
    return tuple(refs)


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
    from ..semantic.discourse import extract_person_refs

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
        # Phase 4C (Task 8): stable typed person identities from the
        # current query (advisory only; people lookup stays gated on
        # can_read_people + capability authorization downstream).
        person_refs=extract_person_refs(base),
        section_refs=_section_refs_from_labels(
            [ref.section_reference for ref in result.document_refs]
        ),
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


def _persisted_candidates(value: object) -> tuple[UUID, ...]:
    """Project persisted candidate IDs onto valid document UUIDs.

    Absent/invalid entries are dropped (never fabricated into identity);
    order is preserved and duplicates collapsed. A non-list payload is
    rejected rather than migrated.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SemanticAdapterError("persisted semantic document_refs.candidates must be a list or null")
    seen: set[UUID] = set()
    ordered: list[UUID] = []
    for entry in value:
        # Writer shape is a {document_id, match_basis, confidence} record;
        # a bare uuid string is accepted for forward compatibility.
        raw = entry.get("document_id") if isinstance(entry, Mapping) else entry
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = UUID(raw)
        except ValueError:
            continue
        if parsed not in seen:
            seen.add(parsed)
            ordered.append(parsed)
    return tuple(ordered)


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
    # Candidates survive only on ambiguity (the frozen validator forbids
    # them on unresolved/not-found/error and they are noise on resolved).
    candidates = (
        _persisted_candidates(entry.get("candidates"))
        if status == "ambiguous"
        else ()
    )
    return DocumentReference(
        ref_id=_require_str(entry.get("ref_id"), "document_refs.ref_id"),
        original_span=reference,
        normalized_reference=reference,
        requested_role=None,
        revision_requirement=None,
        resolution_status=status,
        resolved_document_id=document_id,
        candidate_document_ids=candidates,
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

    from ..semantic.discourse import extract_person_refs

    persisted_refs = _record_list(payload.get("document_refs"), "document_refs")
    return SemanticDraft(
        provisional_contextualized_query=normalized_query,
        abbreviations=tuple(abbreviations),
        coreferences=(),
        document_refs=tuple(
            _persisted_reference(entry) for entry in persisted_refs
        ),
        person_refs=extract_person_refs(normalized_query),
        section_refs=_section_refs_from_labels(
            [
                entry.get("section_reference")
                if isinstance(entry.get("section_reference"), str)
                else None
                for entry in persisted_refs
            ]
        ),
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
async def resolve_draft_identities(
    draft: SemanticDraft,
    *,
    question: str,
    identity_resolver: DocumentIdentityResolver | None,
    workspace_ids: Sequence[UUID],
    db: AsyncSession,
    use_llm_fallback: bool = True,
    can_read_people: bool = False,
) -> SemanticDraft:
    """Enrich unresolved draft refs with v1 identity facts (Phase 4B, Task 6).

    Already-resolved refs (preprocessor/ui_selection/api_explicit) pass
    through untouched; each remaining ref is resolved through the shared
    request-scoped ``DocumentIdentityResolver`` with the full
    contextualized question as ``topic``. Only identity fields
    (``resolution_status``/``resolved_document_id``/
    ``candidate_document_ids``) change: revision pinning stays with the
    existing v2 binding resolver, and the resolver's per-request cache
    makes repeated builds free. When the draft carries no section refs,
    the resolver's preserved authoritative one-turn locator (if any) is
    projected as a label-only ``SectionReference`` (no revision
    coordinate). Errors fail closed (propagate) rather than fabricating
    an identity. Discourse mentions in ``question`` then resolve only
    against the just-resolved document IDs (never workspace IDs, never
    history identity); true ambiguity appends a ``BlockingAmbiguity``.
    """
    if identity_resolver is None:
        raise SemanticAdapterError(
            "identity resolution requires a DocumentIdentityResolver; "
            "refusing to fabricate document identity"
        )
    resolved_refs: list[DocumentReference] = []
    # Refs resolved below may carry a v1 one-turn section locator; capture
    # the pre-resolution refs so their preserved labels can be projected.
    pending: list[DocumentReference] = []
    for reference in draft.document_refs:
        if reference.resolution_status == "resolved":
            resolved_refs.append(reference)
            continue
        pending.append(reference)
        resolved_refs.append(
            await identity_resolver.resolve_reference(
                reference,
                question=question,
                workspace_ids=workspace_ids,
                db=db,
                use_llm_fallback=use_llm_fallback,
            )
        )
    from ..semantic.discourse import (
        merge_ambiguities,
        merge_coreferences,
        resolve_coreferences,
    )

    # Task 8 fix round 1 (Critical-2): the allowed scope is resolved
    # DOCUMENT identity, never the workspace/tenant scope. ``workspace_ids``
    # authorize the resolver search; the resolved IDs below are the only
    # targets a mention may link to (a workspace UUID never equals a
    # document UUID, so passing it here yielded false clarifications).
    allowed_ids = tuple(
        reference.resolved_document_id
        for reference in resolved_refs
        if reference.resolution_status == "resolved"
        and reference.resolved_document_id is not None
    )
    corefs, coref_ambiguities = resolve_coreferences(
        question,
        document_refs=tuple(resolved_refs),
        person_refs=draft.person_refs,
        section_refs=draft.section_refs,
        allowed_document_ids=allowed_ids,
        can_read_people=can_read_people,
    )
    section_refs = draft.section_refs
    if not section_refs and pending:
        # Task 7 bridge: preserve the resolver's authoritative one-turn
        # locator as a label-only section ref (no revision coordinate, so
        # routing must use the bounded document fallback, never
        # section.read). Drafts that already carry section refs win.
        bridged: list[SectionReference] = []
        for reference in pending:
            label = normalize_section_label(
                identity_resolver.cached_section_label(
                    reference,
                    question=question,
                    workspace_ids=workspace_ids,
                    use_llm_fallback=use_llm_fallback,
                )
            )
            if label is None:
                continue
            bridged.append(
                SectionReference(
                    ref_id=f"s{len(bridged) + 1}",
                    label=label,
                    structure_node_id=None,
                )
            )
        section_refs = tuple(bridged)
    return draft.model_copy(
        update={
            "document_refs": tuple(resolved_refs),
            "section_refs": section_refs,
            # Merge (never blind-concat): chained with the draft-build
            # seam on the same turn, output stays idempotent under the
            # frozen uniqueness rules.
            "coreferences": merge_coreferences(draft.coreferences, corefs),
            "preliminary_ambiguities": merge_ambiguities(
                draft.preliminary_ambiguities, coref_ambiguities
            ),
        }
    )


__all__ = [
    "FinalizedSemantics",
    "PERSISTED_SEMANTIC_VERSION",
    "SemanticAdapterError",
    "AbbreviationEntry",
    "draft_from_persisted_semantic",
    "draft_from_preprocessing",
    "finalize_semantic_context",
    "finalize_semantics",
    "resolve_draft_identities",
]
