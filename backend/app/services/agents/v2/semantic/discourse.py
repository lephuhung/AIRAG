"""Deterministic discourse/coreference helpers (Phase 4C, Task 8).

Pure functions only: no LLM, no capability dispatch, no supervisor/graph
state, no DB. Advisory semantic facts for the deterministic v2 route
policy — v2 alone owns ACL, binding, routing, and scheduling.

Confinement rule enforced at every boundary here: history-derived
references NEVER reauthorize resources. ``resolve_coreferences()`` only
resolves to references whose identity is inside the caller-supplied
``allowed_document_ids`` (the current runtime scope); out-of-scope pins
are invisible (never resolved, never named in ambiguity text). Person
mentions only resolve when ``can_read_people`` holds for this turn.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID

from ..contracts.conversation import ActiveEntity, EntityKind, EntityReference
from ..contracts.semantic import (
    BlockingAmbiguity,
    CoreferenceResolution,
    DocumentReference,
    SectionReference,
)

__all__ = [
    "classify_entity_kind",
    "derive_last_focus",
    "extract_person_refs",
    "resolve_coreferences",
    "typed_active_entities",
]

#: Vietnamese phone: exactly 10 digits, 0-leading (mirrors the v1
#: ``supervisor_scope._VN_PHONE_RE`` shape; document numbers never start
#: with 0, so this cannot collide with doc identity).
_PHONE_RE = re.compile(r"\b0\d{9}\b")

#: CCCD / citizen ID: 9-12 digits, only a person identity when a
#: person keyword is adjacent (mirrors v1 scope gating).
_CCCD_RE = re.compile(r"\b\d{9,12}\b")
_PERSON_ID_KEYWORDS = (
    "cccd",
    "căn cước",
    "cancuoc",
    "id card",
    "định danh",
)

#: BHXH: 10 digits with a social-insurance keyword (v1 scope parity).
_BHXH_RE = re.compile(r"\b\d{10}\b")
_BHXH_KEYWORDS = ("bhxh", "bảo hiểm xã hội", "bảo hiểm", "sổ bảo hiểm")

#: Phone keyword gate for masked/redacted numbers (v1 scope parity).
_PHONE_KEYWORDS = ("sđt", "sdt", "số điện thoại", "điện thoại", "phone", "liên lạc")
_MASKED_ID_RE = re.compile(r"\*{2,}\s*\d{3,4}")

#: Honorific + capitalized Vietnamese name ("ông Nguyễn Văn A").
#: Requires an explicit person cue so ordinary capitalized nouns
#: ("Luật An ninh mạng") never become person refs.
_HONORIFIC_NAME_RE = re.compile(
    r"(?<!\w)(ông|bà|anh|chị|em|cô|chú|bác)\s+"
    r"([A-ZĐ][a-zà-ỹđ]*(?:\s+[A-ZĐ][a-zà-ỹđ]*)+)",
    re.UNICODE,
)
_PERSON_NAME_CUE_RE = re.compile(
    r"(?:là\s+ai|tìm\s+(?:ông|bà|anh|chị|em|cô|chú|bác|người)|"
    r"tra\s+cứu\s+(?:người|thông\s*tin)|phone\s+of|"
    r"số\s+điện\s+thoại\s+của)",
    re.IGNORECASE | re.UNICODE,
)

#: Bare multi-word capitalized name ("Nguyễn Văn A") — only a person
#: identity together with a person cue (``là ai`` et al.).
_CAPITALIZED_NAME_RE = re.compile(
    r"(?<!\w)([A-ZĐ][a-zà-ỹđ]*(?:\s+[A-ZĐ][a-zà-ỹđ]*){1,4})(?!\w)",
    re.UNICODE,
)

#: Document-type keywords for entity typing (mirrors the v1 named-doc
#: vocabulary, logic-free).
_DOC_TYPE_RE = re.compile(
    r"(luật|nghị\s*định|thông\s*tư|quyết\s*định|nghị\s*quyết|"
    r"pháp\s*lệnh|bộ\s*luật|hiến\s*pháp|chỉ\s*thị|công\s*văn|"
    r"\bnđ\b|\bqđ\b|\bnq\b)",
    re.IGNORECASE | re.UNICODE,
)
_DOC_NUM_RE = re.compile(
    r"\b\d{1,4}\s*/\s*(?:19|20)\d{2}\s*/",
    re.UNICODE,
)

#: Section labels for entity typing.
_SECTION_LABEL_RE = re.compile(
    r"(?:điều|khoản|chương|mục|phần|phụ\s*lục)\s+[\dIVXivx]+",
    re.IGNORECASE | re.UNICODE,
)

#: Discourse anaphora mentions (surface spans, matched case-insensitively).
_DOC_MENTION_RE = re.compile(
    r"(?<!\w)(văn\s+bản\s+(?:này|đó|ấy|kia|trên)|"
    r"tài\s+liệu\s+(?:này|đó|ấy|kia|trên)|"
    r"nghị\s*định\s+(?:này|đó|trên|ấy)|"
    r"thông\s*tư\s+(?:này|đó|trên)|"
    r"luật\s+(?:này|đó)|"
    r"file\s+(?:này|đó)|"
    r"quy\s+định\s+(?:này|đó))(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_SECTION_MENTION_RE = re.compile(
    r"(?<!\w)(điều|khoản|chương|mục)\s+(?:này|đó|trên|ấy)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_PERSON_MENTION_RE = re.compile(
    r"(?<!\w)(ông|bà|anh|chị|em|cô|chú|bác|người)\s+ấy(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_ORDINAL_DOC_RE = re.compile(
    r"(?<!\w)(?:file|văn\s+bản|tài\s+liệu)\s+thứ\s+"
    r"(nhất|hai|ba|bốn|năm|một|1|2|3|4|5)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_ORDINAL_VALUE = {
    "nhất": 1,
    "một": 1,
    "1": 1,
    "hai": 2,
    "2": 2,
    "ba": 3,
    "3": 3,
    "bốn": 4,
    "4": 4,
    "năm": 5,
    "5": 5,
}


def _lower(text: str) -> str:
    return (text or "").lower()


def extract_person_refs(query: str) -> tuple[EntityReference, ...]:
    """Extract stable typed person identities from the current query.

    Only identifier-grade evidence becomes a ref: VN phone, CCCD/BHXH
    with their keyword gates, masked IDs with a phone keyword, or an
    honorific/cued capitalized name. Anything weaker yields no ref
    (never guess a person identity).
    """
    text = (query or "").strip()
    if not text:
        return ()
    lowered = _lower(text)
    refs: list[EntityReference] = []
    seen: set[str] = set()

    def _add(label: str) -> None:
        key = label.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        refs.append(
            EntityReference(
                ref_id=f"p{len(refs) + 1}", kind="person", label=label.strip()
            )
        )

    for match in _PHONE_RE.finditer(text):
        _add(match.group(0))
    if any(keyword in lowered for keyword in _PERSON_ID_KEYWORDS):
        for match in _CCCD_RE.finditer(text):
            # A bare 10-digit 0-leading hit is a phone, already added.
            if _PHONE_RE.fullmatch(match.group(0)):
                continue
            _add(match.group(0))
    if any(keyword in lowered for keyword in _BHXH_KEYWORDS):
        for match in _BHXH_RE.finditer(text):
            if _PHONE_RE.fullmatch(match.group(0)):
                continue
            _add(match.group(0))
    if any(keyword in lowered for keyword in _PHONE_KEYWORDS):
        for match in _MASKED_ID_RE.finditer(text):
            _add(match.group(0))
    for match in _HONORIFIC_NAME_RE.finditer(text):
        _add(f"{match.group(1)} {match.group(2)}")
    if _PERSON_NAME_CUE_RE.search(text):
        for match in _CAPITALIZED_NAME_RE.finditer(text):
            candidate = match.group(1)
            # Skip spans already covered by the honorific match and
            # document-type phrases ("Luật An ninh mạng" is a document).
            if _DOC_TYPE_RE.search(candidate):
                continue
            _add(candidate)
    return tuple(refs)


def classify_entity_kind(label: str) -> EntityKind:
    """Project a discourse label onto a typed entity kind (deterministic).

    Document and section signals win over person (a "Nghị định" phrase
    naming no one is a document); person identifiers/names win over the
    ``concept`` fallback. Pure fallback is ``concept`` — never invented
    identity.
    """
    text = (label or "").strip()
    if not text:
        return "concept"
    if _SECTION_LABEL_RE.search(text) and not _DOC_TYPE_RE.search(text):
        return "section"
    if _DOC_TYPE_RE.search(text) or _DOC_NUM_RE.search(text):
        return "document"
    lowered = _lower(text)
    if (
        _PHONE_RE.search(text)
        or _MASKED_ID_RE.search(text)
        or (
            _CCCD_RE.search(text)
            and any(keyword in lowered for keyword in _PERSON_ID_KEYWORDS)
        )
        or _HONORIFIC_NAME_RE.search(text)
        or (
            _CAPITALIZED_NAME_RE.fullmatch(text)
            and not _DOC_TYPE_RE.search(text)
        )
    ):
        return "person"
    return "concept"


def typed_active_entities(labels: Sequence[str]) -> tuple[ActiveEntity, ...]:
    """Build typed discourse entities from summary labels (order-preserving).

    Blank labels are dropped; duplicates collapse. ``ref_id`` stays the
    label (existing conversation-adapter convention).
    """
    entities: list[ActiveEntity] = []
    seen: set[str] = set()
    for label in labels or ():
        if not isinstance(label, str) or not label.strip():
            continue
        clean = label.strip()
        if clean in seen:
            continue
        seen.add(clean)
        entities.append(
            ActiveEntity(ref_id=clean, kind=classify_entity_kind(clean), label=clean)
        )
    return tuple(entities)


def derive_last_focus(
    entities: Sequence[EntityReference | ActiveEntity],
) -> EntityReference | None:
    """Derive discourse focus from validated outcomes: the most recent entity.

    The caller passes already-validated entities (contract validation runs
    on the assembled context); the last one is the current focus. Empty
    input yields ``None`` — focus is never invented.
    """
    ordered = [entity for entity in entities or () if entity is not None]
    if not ordered:
        return None
    last = ordered[-1]
    return EntityReference(ref_id=last.ref_id, kind=last.kind, label=last.label)


def _in_scope_ids(allowed: Sequence[UUID | str]) -> set[str]:
    return {str(item) for item in allowed or ()}


def resolve_coreferences(
    query: str,
    *,
    document_refs: Sequence[DocumentReference] = (),
    person_refs: Sequence[EntityReference] = (),
    section_refs: Sequence[SectionReference] = (),
    active_entities: Sequence[ActiveEntity] = (),
    allowed_document_ids: Sequence[UUID | str] = (),
    can_read_people: bool = False,
) -> tuple[tuple[CoreferenceResolution, ...], tuple[BlockingAmbiguity, ...]]:
    """Resolve unambiguous discourse mentions; true ambiguity clarifies.

    ``active_entities`` are discourse labels only (no identity) — they
    confirm a mention KIND but never supply a resolution target. Only
    validated current-turn references (``document_refs``/``person_refs``/
    ``section_refs``) resolve, and document targets must sit inside
    ``allowed_document_ids``: history pins outside the current scope are
    invisible here, so history can never reauthorize a resource. Person
    targets additionally require ``can_read_people``. Ambiguity text is
    generic (counts only) — no out-of-scope IDs, titles, or scores leak.
    """
    _ = active_entities  # kind confirmation only; never a resolution target.
    text = (query or "").strip()
    if not text:
        return (), ()
    scope = _in_scope_ids(allowed_document_ids)
    scoped_docs = [
        reference
        for reference in document_refs or ()
        if reference.resolution_status == "resolved"
        and reference.resolved_document_id is not None
        and str(reference.resolved_document_id) in scope
    ]
    people = (
        [reference for reference in person_refs or ()]
        if can_read_people
        else []
    )
    sections = [reference for reference in section_refs or ()]

    corefs: list[CoreferenceResolution] = []
    ambiguities: list[BlockingAmbiguity] = []

    def _clarify(mention: str, kind: str, count: int) -> None:
        ambiguities.append(
            BlockingAmbiguity(
                ambiguity_id=f"coref-{kind}-{len(ambiguities) + 1}",
                description=(
                    f"Không rõ {mention!r} chỉ tới đâu "
                    f"({count} ứng viên {kind}); "
                    "vui lòng nêu rõ tên/tài liệu cụ thể."
                ),
            )
        )

    def _resolve_one(mention: str, kind: str, candidates: Sequence[str]) -> None:
        if len(candidates) == 1:
            corefs.append(
                CoreferenceResolution(
                    mention=mention, resolved_ref_id=candidates[0]
                )
            )
        else:
            _clarify(mention, kind, len(candidates))

    ordinal = _ORDINAL_DOC_RE.search(text)
    if ordinal is not None:
        mention = ordinal.group(0)
        index = _ORDINAL_VALUE.get(ordinal.group(1).lower(), 0)
        if 1 <= index <= len(scoped_docs):
            corefs.append(
                CoreferenceResolution(
                    mention=mention,
                    resolved_ref_id=scoped_docs[index - 1].ref_id,
                )
            )
        else:
            _clarify(mention, "tài liệu", len(scoped_docs))
    doc_mention = _DOC_MENTION_RE.search(text)
    if doc_mention is not None and ordinal is None:
        mention = doc_mention.group(0)
        _resolve_one(
            mention, "tài liệu", [reference.ref_id for reference in scoped_docs]
        )
    section_mention = _SECTION_MENTION_RE.search(text)
    if section_mention is not None:
        mention = section_mention.group(0)
        _resolve_one(
            mention, "điều khoản", [reference.ref_id for reference in sections]
        )
    person_mention = _PERSON_MENTION_RE.search(text)
    if person_mention is not None:
        mention = person_mention.group(0)
        if not can_read_people:
            _clarify(mention, "người", 0)
        else:
            _resolve_one(
                mention, "người", [reference.ref_id for reference in people]
            )
    return tuple(corefs), tuple(ambiguities)
