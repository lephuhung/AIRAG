"""High-risk anchor extraction, canonicalization, and support checks.

Spec §10.3: deterministic guards cover factual anchors where hallucination is
especially damaging and canonical literal matching is feasible — monetary and
other numeric quantities, percentages, dates, explicit durations/deadlines,
``Điều``/``Khoản``/``Điểm``/``Chương``/``Mục`` locators, official document
numbers, and configured closed literals.

Canonicalization is deliberately conservative: Unicode-normalize + case-fold,
grouping separators only where unambiguous, Vietnamese scale words as numeric
multipliers, currency/unit as part of the anchor, ``%`` preserved, zero-padded
dates/durations normalized only when semantically equivalent, qualifiers never
collapsed, hierarchical locators never weakened, and document numbers exact.
An anchor that cannot be canonicalized safely (``canonical is None``) is never
proven — it enters the repair/failure path.

Support rule (§10.3): a canonical claim anchor must occur in at least one
*individually cited* evidence item — never assembled across sources.

Spec §10.4 honesty: anchor presence is literal support, not entailment.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal

__all__ = [
    "AnchorKind",
    "ClaimAnchor",
    "extract_anchors",
    "anchor_supported",
    "unsupported_anchors",
]

AnchorKind = Literal[
    "amount", "percent", "date", "duration", "locator", "doc_number", "literal"
]

#: Canonical anchor forms per kind:
#:   amount    -> ("amount", value:int|float, unit:str)   unit "" = bare number
#:   percent   -> ("percent", value:int|float)
#:   date      -> ("date", year|None, month|None, day|None)
#:   duration  -> ("duration", value:int|float, unit:str) unit keeps qualifiers
#:   locator   -> ("locator", frozenset[(level, number)])
#:   doc_number-> ("doc_number", casefolded_symbol:str)
#:   literal   -> ("literal", normalized_text:str)
#: ``canonical=None`` marks an anchor that failed conservative
#: canonicalization; it is never supported.


@dataclass(frozen=True)
class ClaimAnchor:
    """One extracted high-risk literal with its canonical form."""

    kind: AnchorKind
    surface: str
    canonical: tuple | None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """NFC + casefold + collapsed whitespace — the shared comparison form."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _norm_number(token: str) -> int | float | None:
    """Canonicalize one numeric token or return ``None`` when ambiguous.

    Rules (spec §10.3): grouping separators only where unambiguous
    (``20.000.000`` -> ``20000000``); a single separator with a three-digit
    tail is ambiguous (``20.000`` could be twenty-thousand or twenty) and is
    never guessed; a non-three-digit tail is a decimal (``20,5``); with both
    separators the last one is the decimal only when its tail is ≤2 digits.
    Leading zeros normalize (``05`` -> ``5``).
    """
    token = token.strip()
    if not token or not token[0].isdigit():
        return None
    if token.isdigit():
        value = int(token)
        return value
    if " " in token:
        # Space-separated groups: grouping only when every tail group is 3.
        parts = token.split(" ")
        if all(p.isdigit() for p in parts) and all(
            len(p) == 3 for p in parts[1:]
        ):
            return int("".join(parts))
        return None
    dot = token.rfind(".")
    comma = token.rfind(",")
    if dot >= 0 and comma >= 0:
        last = max(dot, comma)
        head, tail = token[:last], token[last + 1 :]
        if not tail.isdigit() or len(tail) > 2:
            return None
        head_digits = head.replace(".", "").replace(",", "")
        if not head_digits.isdigit():
            return None
        # Every separator in the head must be a clean 3-digit grouping.
        groups = re.split(r"[.,]", head)
        if not all(len(g) == 3 for g in groups[1:]):
            return None
        return float(f"{int(head_digits)}.{tail}")
    for sep in (".", ","):
        if sep not in token:
            continue
        parts = token.split(sep)
        if not all(p.isdigit() for p in parts):
            return None
        if len(parts) > 2:
            # Repeated separator: grouping only when all tail groups are 3.
            if all(len(p) == 3 for p in parts[1:]):
                return int("".join(parts))
            return None
        head, tail = parts
        if len(tail) == 3:
            return None  # ambiguous: "20.000" / "20,000" — never guessed
        if len(tail) == 0:
            return None
        return float(f"{int(head)}.{tail}")
    return None


def _fmt_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Extraction patterns (applied to the normalized text, longest-kind first)
# ---------------------------------------------------------------------------

_VN_LETTER = r"[^\W\d_]"
_NUM = r"\d(?:[\d., ]*\d)?"

# Official document numbers/symbols: "12/2020/NĐ-CP", "123/QĐ-UBND",
# "50-LP/TN". Exact match after casefold — never fuzzy.
_DOC_NUMBER_RES = (
    re.compile(r"(?<![\w/])\d{1,4}/\d{4}/[^\s,;()]+"),
    re.compile(
        rf"(?<![\d/\w])\d{{1,4}}-{_VN_LETTER}{{1,8}}/{_VN_LETTER}[\w-]*"
    ),
)

_DATE_RES = (
    # ngày 15 tháng 9 năm 2026 / ngày 15 tháng 9
    re.compile(
        rf"ngày\s+(\d{{1,2}})\s+tháng\s+(\d{{1,2}})(?:\s+năm\s+(\d{{4}}))?"
    ),
    # tháng 9 năm 2026 / tháng 9
    re.compile(rf"tháng\s+(\d{{1,2}})(?:\s+năm\s+(\d{{4}}))?"),
    # năm 2026
    re.compile(rf"năm\s+(\d{{4}})"),
    # ISO 2026-09-15
    re.compile(r"(?<![\d/-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d/-])"),
    # 15/09/2026, 15-09-2026, 15.09.2026
    re.compile(
        r"(?<![\w/])(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?![\d/])"
    ),
    # 12/2020 (month/year)
    re.compile(r"(?<![\w/])(\d{1,2})/(\d{4})(?![\d/])"),
    # 15/09 (day/month)
    re.compile(r"(?<![\w/.])(\d{1,2})[-/.](\d{1,2})(?![\d/.])"),
)

_DURATION_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s+"
    rf"(giây|phút|giờ|ngày|tuần|tháng|quý|năm)"
    rf"(?:\s+({_VN_LETTER}+(?:\s+{_VN_LETTER}+)?))?"
)

_PERCENT_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s*(%|phần\s+trăm)(?![\d])"
)

_LOCATOR_RE = re.compile(
    rf"(?<![\w.])(điều|khoản|điểm|chương|mục)\s*"
    rf"(\d{{1,3}}[a-zđ]?|[ivxlcdm]{{1,7}})(?![\w])"
)

_SCALE_WORDS = {
    "nghìn": 1_000,
    "ngàn": 1_000,
    "vạn": 10_000,
    "triệu": 1_000_000,
    "tỷ": 1_000_000_000,
    "tỉ": 1_000_000_000,
}
_SCALE_PATTERN = "|".join(sorted(_SCALE_WORDS, key=len, reverse=True))

_AMOUNT_RE = re.compile(
    rf"(?<![\w.,])({_NUM})"
    rf"(?:\s+({_SCALE_PATTERN}))?"
    rf"(?:\s+({_NUM})\s+({_SCALE_PATTERN}))?"
    rf"(?:\s+({_VN_LETTER}+))?"
)

#: Words that never count as an amount unit (conjunctions, prepositions).
_UNIT_STOPWORDS = frozenset(
    {
        "và", "hoặc", "hay", "của", "cho", "với", "từ", "đến", "trong",
        "theo", "tại", "về", "trên", "dưới", "sau", "trước", "khi", "nếu",
        "thì", "là", "được", "bị", "có", "không", "đã", "sẽ", "đang",
        "một", "các", "những", "mỗi", "từng", "bao", "gồm", "đó", "này",
        "kể", "kể từ", "tính", "tính từ", "đối", "đối với", "như", "ra",
        "vào", "lên", "xuống", "còn", "cũng", "rất", "đều", "phải",
    }
)

_LOCATOR_LEVELS = ("chương", "mục", "điều", "khoản", "điểm")

_ROMAN = {
    "i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000,
}


def _roman_to_int(token: str) -> int | None:
    if not token or any(c not in _ROMAN for c in token):
        return None
    total = 0
    prev = 0
    for char in reversed(token):
        value = _ROMAN[char]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total if total > 0 else None


def _locator_number(token: str) -> str | None:
    token = token.strip()
    if token.isdigit() or (token[:-1].isdigit() and token[-1:].isalpha()):
        head = token[:-1] if not token.isdigit() else token
        suffix = "" if token.isdigit() else token[-1]
        return f"{int(head)}{suffix}"
    roman = _roman_to_int(token)
    return str(roman) if roman is not None else None


def _valid_date(year: int | None, month: int | None, day: int | None) -> bool:
    if month is not None and not 1 <= month <= 12:
        return False
    if day is not None and not 1 <= day <= 31:
        return False
    return True


def _mask(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Blank matched spans so later passes cannot re-read consumed digits."""
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = "\x00"
    return "".join(chars)


def _merge_locator_matches(
    matches: list[re.Match[str]],
) -> list[tuple[tuple[int, int], frozenset]]:
    """Merge adjacent locator components into one hierarchical anchor.

    ``Khoản 2 Điều 5`` is one anchor; ``Điều 5 và Điều 6`` stays two because
    the conjunction breaks adjacency. Only whitespace/commas may separate
    merged components.
    """
    groups: list[tuple[tuple[int, int], frozenset]] = []
    for match in matches:
        level = match.group(1)
        number = _locator_number(match.group(2))
        if number is None:
            continue
        pair = (level, number)
        if groups:
            (start, end), pairs = groups[-1]
            gap = match.string[end : match.start()]
            # Merge only across hierarchy levels: "Khoản 2, Điều 5" is one
            # locator, but "Điều 5, Điều 6" is a list of two distinct ones.
            same_level = any(existing == level for existing, _ in pairs)
            if re.fullmatch(r"[\s,]*", gap) and not same_level:
                groups[-1] = ((start, match.end()), pairs | {pair})
                continue
        groups.append(((match.start(), match.end()), frozenset({pair})))
    return groups


def extract_anchors(
    text: str, *, literals: tuple[str, ...] = ()
) -> tuple[ClaimAnchor, ...]:
    """Extract high-risk anchors from claim (or evidence) text.

    Extraction is kind-priority: document numbers, dates, durations,
    percentages, and locators mask their spans before numeric amounts are
    read, so ``Điều 5`` never also yields a bare ``5`` and ``5 ngày`` never
    yields a bare ``5``. Anchors that fail conservative canonicalization are
    returned with ``canonical=None`` (unproven by construction).
    """
    normalized = _normalize(text)
    anchors: list[ClaimAnchor] = []
    consumed: list[tuple[int, int]] = []

    def scan(patterns, build) -> None:
        nonlocal normalized
        for pattern in patterns:
            for match in pattern.finditer(normalized):
                anchor = build(match)
                if anchor is not None:
                    anchors.append(anchor)
                    consumed.append(match.span())
            # Mask after EACH pattern so a later sub-pattern cannot re-read
            # spans an earlier one already consumed (e.g. "ngày 15 tháng 9"
            # must not also yield a bare "tháng 9" date).
            normalized = _mask(normalized, consumed)
            consumed.clear()

    # 1. Official document numbers (before dates: "12/2020/NĐ-CP" masks the
    #    "12/2020" month/year prefix).
    scan(
        _DOC_NUMBER_RES,
        lambda m: ClaimAnchor(
            "doc_number",
            m.group(0),
            ("doc_number", m.group(0).rstrip(".,;")),
        ),
    )

    # 2. Dates.
    def _date(match: re.Match[str]) -> ClaimAnchor | None:
        pattern = match.re
        groups = match.groups()
        if pattern is _DATE_RES[0]:
            day, month, year = int(groups[0]), int(groups[1]), (
                int(groups[2]) if groups[2] else None
            )
        elif pattern is _DATE_RES[1]:
            day, month, year = None, int(groups[0]), (
                int(groups[1]) if groups[1] else None
            )
        elif pattern is _DATE_RES[2]:
            day, month, year = None, None, int(groups[0])
        elif pattern is _DATE_RES[3]:
            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
        elif pattern is _DATE_RES[4]:
            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        elif pattern is _DATE_RES[5]:
            day, month, year = None, int(groups[0]), int(groups[1])
        else:
            day, month, year = int(groups[0]), int(groups[1]), None
        canonical = (
            ("date", year, month, day)
            if _valid_date(year, month, day)
            else None
        )
        return ClaimAnchor("date", match.group(0), canonical)

    scan(_DATE_RES, _date)

    # 3. Durations/deadlines (before amounts: "5 ngày" masks the "5").
    def _duration(match: re.Match[str]) -> ClaimAnchor:
        value = _norm_number(match.group(1))
        unit = match.group(2)
        qualifier = match.group(3)
        if qualifier and qualifier in _UNIT_STOPWORDS:
            qualifier = None
        canonical = None
        if value is not None:
            full_unit = f"{unit} {qualifier}" if qualifier else unit
            canonical = ("duration", _fmt_number(value), full_unit)
        return ClaimAnchor("duration", match.group(0), canonical)

    scan((_DURATION_RE,), _duration)

    # 4. Percentages (before amounts: "20%" masks the "20").
    def _percent(match: re.Match[str]) -> ClaimAnchor:
        value = _norm_number(match.group(1))
        canonical = (
            ("percent", _fmt_number(value)) if value is not None else None
        )
        return ClaimAnchor("percent", match.group(0), canonical)

    scan((_PERCENT_RE,), _percent)

    # 5. Legal locators (before amounts: "Điều 5" masks the "5"). Adjacent
    #    components merge into one hierarchical anchor.
    locator_matches = list(_LOCATOR_RE.finditer(normalized))
    for (start, end), pairs in _merge_locator_matches(locator_matches):
        anchors.append(
            ClaimAnchor("locator", normalized[start:end], ("locator", pairs))
        )
        consumed.append((start, end))
    normalized = _mask(normalized, consumed)
    consumed.clear()

    # 6. Numeric amounts/quantities over whatever remains.
    def _amount(match: re.Match[str]) -> ClaimAnchor | None:
        first = _norm_number(match.group(1))
        if first is None:
            return ClaimAnchor("amount", match.group(0), None)
        value: int | float = first
        scale1 = match.group(2)
        if scale1:
            value = value * _SCALE_WORDS[scale1]
        second = match.group(3)
        scale2 = match.group(4)
        if second is not None and scale2:
            second_value = _norm_number(second)
            if second_value is None:
                return ClaimAnchor("amount", match.group(0), None)
            value = value + second_value * _SCALE_WORDS[scale2]
        unit = match.group(5) or ""
        if unit in _UNIT_STOPWORDS:
            unit = ""
        return ClaimAnchor(
            "amount", match.group(0), ("amount", _fmt_number(value), unit)
        )

    scan((_AMOUNT_RE,), _amount)

    # 7. Configured closed literals: verbatim (normalized) presence.
    normalized_text = _normalize(text)
    for literal in literals:
        needle = _normalize(literal)
        if needle and needle in normalized_text:
            anchors.append(ClaimAnchor("literal", literal, ("literal", needle)))

    # Dedupe by canonical form, preserving first-seen order.
    seen: set[tuple] = set()
    unique: list[ClaimAnchor] = []
    for anchor in anchors:
        key = (anchor.kind, anchor.canonical)
        if key in seen:
            continue
        seen.add(key)
        unique.append(anchor)
    return tuple(unique)


# ---------------------------------------------------------------------------
# Support checks
# ---------------------------------------------------------------------------


def _amount_supported(claim: tuple, evidence_anchors: tuple[ClaimAnchor, ...]) -> bool:
    _, value, unit = claim
    for anchor in evidence_anchors:
        if anchor.kind != "amount" or anchor.canonical is None:
            continue
        _, e_value, e_unit = anchor.canonical
        if e_value != value:
            continue
        # Unit is part of the anchor: a unit-bearing claim needs the exact
        # unit; a bare quantity is literally present inside a unit-bearing
        # evidence anchor ("2 tỷ" occurs in "2 tỷ đồng").
        if unit and e_unit != unit:
            continue
        return True
    return False


def _locator_supported(claim: tuple, evidence_anchors: tuple[ClaimAnchor, ...]) -> bool:
    _, pairs = claim
    for anchor in evidence_anchors:
        if anchor.kind != "locator" or anchor.canonical is None:
            continue
        # Hierarchy is preserved: every claimed level must occur inside ONE
        # evidence locator; "Khoản 2 Điều 5" is never weakened to "Điều 5".
        if pairs <= anchor.canonical[1]:
            return True
    return False


def _anchor_in_item(anchor: ClaimAnchor, item_anchors: tuple[ClaimAnchor, ...]) -> bool:
    """True when the canonical claim anchor occurs in ONE evidence item."""
    if anchor.canonical is None:
        return False
    if anchor.kind == "amount":
        return _amount_supported(anchor.canonical, item_anchors)
    if anchor.kind == "locator":
        return _locator_supported(anchor.canonical, item_anchors)
    return any(other.canonical == anchor.canonical for other in item_anchors)


def anchor_supported(
    anchor: ClaimAnchor, evidence_items: Iterable[tuple[ClaimAnchor, ...]]
) -> bool:
    """True when the anchor occurs in at least one individual evidence item."""
    return any(_anchor_in_item(anchor, item) for item in evidence_items)


def unsupported_anchors(
    claim_anchors: tuple[ClaimAnchor, ...],
    evidence_items: Iterable[tuple[ClaimAnchor, ...]],
) -> tuple[ClaimAnchor, ...]:
    """The claim anchors no single cited evidence item supports.

    ``evidence_items`` is one extracted-anchor tuple per *individually cited*
    evidence item; anchors are never assembled across items (spec §10.3).
    """
    items = tuple(evidence_items)
    return tuple(
        anchor for anchor in claim_anchors if not anchor_supported(anchor, items)
    )
