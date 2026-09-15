"""Conversation port: legacy chat rows → v2 discourse contracts (spec §8.2).

This is a **server-internal** port, not the agent-facing tool gateway. The chat
database stays the authoritative conversation history; this adapter only
translates already-loaded legacy rows into the minimal v2 shapes:

- ``ChatMessage`` rows → ``ConversationTurn`` / ``ConversationContext``;
- ``ExchangeSummary`` rows → the rolling ``summary``, the monotonic
  ``summary_version`` (``exchange_index``) and ``built_through_message_id`` of
  ``ConversationSnapshot``, plus ``ActiveEntity`` labels from ``key_entities``.

``last_focus`` is derived from the validated typed entities (Phase 4C,
Task 8): the most recent active entity is the current discourse focus.
It is never invented — empty entity windows yield ``None``.
Persistence/optimistic locking stays with
``persistence.snapshots.ConversationSnapshotRepository``.

Confinement: history rows carry labels only, never identity. This port
consumes ``key_entities`` labels and recent-turn text; document IDs from
``document_ids``/``sources``/``people_data`` columns are never projected
into v2 contracts, so history can never reauthorize an out-of-scope
resource (resolution always re-checks the current runtime scope).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ..contracts.base import CONTRACT_VERSION
from ..contracts.conversation import (
    ActiveEntity,
    ConversationContext,
    ConversationSnapshot,
    ConversationTurn,
)
from ..contracts.request import KnownDocumentResource
from ..contracts.validation import (
    validate_conversation_context,
    validate_conversation_snapshot,
)

#: How many recent turns the discourse window carries by default.
DEFAULT_RECENT_TURN_LIMIT = 20

_LEGACY_ROLES = frozenset({"user", "assistant", "system"})


class ConversationAdapterError(ValueError):
    """A legacy chat row cannot be translated into a v2 conversation contract."""


class LegacyChatMessage(Protocol):
    """The legacy ``ChatMessage`` attributes this port consumes."""

    role: str
    content: str


class LegacyExchangeSummary(Protocol):
    """The legacy ``ExchangeSummary`` attributes this port consumes."""

    exchange_index: int
    user_message_id: str
    assistant_message_id: str | None
    summary: str
    key_entities: list[str] | None


def _ordered_summaries(
    summaries: Sequence[LegacyExchangeSummary],
) -> tuple[LegacyExchangeSummary, ...]:
    return tuple(sorted(summaries, key=lambda summary: summary.exchange_index))


def _recent_turns(
    messages: Sequence[LegacyChatMessage], limit: int
) -> tuple[ConversationTurn, ...]:
    if limit < 0:
        raise ConversationAdapterError("max_recent_turns must be non-negative")
    turns: list[ConversationTurn] = []
    for message in messages[-limit:] if limit else ():
        if message.role not in _LEGACY_ROLES:
            raise ConversationAdapterError(
                f"unsupported legacy chat role {message.role!r}"
            )
        if not isinstance(message.content, str):
            raise ConversationAdapterError("legacy chat message content must be a string")
        # Blank/whitespace-only rows (e.g. attachment- or citation-only
        # history rows with empty content) carry no discourse text: skip
        # them here so the frozen non-blank validation downstream keeps a
        # valid window. Identity extraction is unaffected — it reads every
        # loaded row's server-issued columns, never turn text.
        if not message.content.strip():
            continue
        turns.append(ConversationTurn(role=message.role, content=message.content))
    return tuple(turns)


def _active_entities(
    summaries: Sequence[LegacyExchangeSummary],
) -> tuple[ActiveEntity, ...]:
    from ..semantic.discourse import typed_active_entities

    labels: list[str] = []
    for summary in summaries:
        for label in summary.key_entities or ():
            if not isinstance(label, str) or not label.strip():
                raise ConversationAdapterError(
                    "legacy exchange key_entities must contain non-blank strings"
                )
            labels.append(label)
    # Typed kinds (document/person/section/concept) from the shared
    # deterministic discourse layer — never flattened to ``concept``.
    return typed_active_entities(labels)


def _summary_text(summaries: Sequence[LegacyExchangeSummary]) -> str:
    parts: list[str] = []
    for summary in summaries:
        if not isinstance(summary.summary, str):
            raise ConversationAdapterError("legacy exchange summary must be a string")
        if summary.summary.strip():
            parts.append(summary.summary)
    return "\n\n".join(parts)


def context_from_legacy(
    *,
    messages: Sequence[LegacyChatMessage],
    exchange_summaries: Sequence[LegacyExchangeSummary] = (),
    max_recent_turns: int = DEFAULT_RECENT_TURN_LIMIT,
) -> ConversationContext:
    """Translate legacy chat rows into the minimal v2 discourse context."""
    from ..semantic.discourse import derive_last_focus

    summaries = _ordered_summaries(exchange_summaries)
    entities = _active_entities(summaries)
    focus = derive_last_focus(entities)
    context = ConversationContext(
        summary=_summary_text(summaries),
        active_entities=entities,
        last_focus=focus,
        recent_turns=_recent_turns(messages, max_recent_turns),
    )
    validate_conversation_context(context)
    return context


#: Bound on server-issued conversation resources projected per turn (Task
#: 1B): the projection reads only the already-bounded history window, and
#: this cap guards against a single message carrying a huge identity list.
MAX_CONVERSATION_RESOURCES = 20


@dataclass(frozen=True)
class ConversationHistory:
    """Typed loader result: labels/text context plus identity resources.

    ``context`` is the existing label/text-only ``ConversationContext``
    (document UUIDs never enter it); ``resources`` carries the
    server-issued document identities (``source="conversation"``) the
    ingress merges into the current request's known documents. Both are
    empty on standalone threads, missing history, malformed rows, or any
    load failure (fail-open).
    """

    context: ConversationContext
    resources: tuple[KnownDocumentResource, ...] = ()

    @staticmethod
    def empty() -> "ConversationHistory":
        """Fail-open bundle: no discourse, no resources."""
        return ConversationHistory(
            context=ConversationContext(
                summary="", active_entities=(), last_focus=None, recent_turns=()
            ),
            resources=(),
        )


def _coerce_history_uuid(raw: object) -> UUID | None:
    """Project one historical identity value onto a UUID, or ``None``.

    Malformed values (non-UUID strings, blanks, wrong types) are ignored —
    history must never fabricate identity and must never fail the turn.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, UUID):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return UUID(raw.strip())
    except (ValueError, AttributeError):
        return None


def _message_identity_values(message: object) -> list[object]:
    """Raw identity values of one legacy message, in deterministic order.

    Server-issued only: the ``document_ids`` column, then public citation
    metadata (``citations[].document_id``), then served sources
    (``sources[].document_id``). Message text is never parsed for UUIDs.
    Every access is defensive: unexpected shapes yield nothing.
    """
    values: list[object] = []
    try:
        document_ids = getattr(message, "document_ids", None)
    except Exception:  # noqa: BLE001 — malformed row, ignore
        document_ids = None
    if isinstance(document_ids, (list, tuple)):
        values.extend(document_ids)
    for column in ("citations", "sources"):
        try:
            entries = getattr(message, column, None)
        except Exception:  # noqa: BLE001 — malformed row, ignore
            continue
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                values.append(entry.get("document_id"))
            else:
                try:
                    values.append(getattr(entry, "document_id", None))
                except Exception:  # noqa: BLE001 — malformed entry, ignore
                    continue
    return values


def conversation_resources_from_legacy(
    messages: Sequence[object],
) -> tuple[KnownDocumentResource, ...]:
    """Project server-issued history identities onto conversation resources.

    Chronological order is preserved (callers pass oldest-first), duplicates
    collapse keeping the first occurrence (stable ordinals: ``file thứ hai``
    always names the same document), and the projection is capped at
    ``MAX_CONVERSATION_RESOURCES``. Malformed values are ignored. Resources
    are candidates only: ``resource_id`` is a deterministic ``conv-N``
    handle (never a UUID/title leak vector), and binding/ACL stays with
    the existing request-scoped resolver/binder.
    """
    ordered: list[UUID] = []
    seen: set[UUID] = set()
    for message in messages or ():
        for raw in _message_identity_values(message):
            document_id = _coerce_history_uuid(raw)
            if document_id is None or document_id in seen:
                continue
            seen.add(document_id)
            ordered.append(document_id)
            if len(ordered) >= MAX_CONVERSATION_RESOURCES:
                break
        if len(ordered) >= MAX_CONVERSATION_RESOURCES:
            break
    return tuple(
        KnownDocumentResource(
            resource_id=f"conv-{index + 1}",
            document_id=document_id,
            source="conversation",
        )
        for index, document_id in enumerate(ordered)
    )


def snapshot_from_legacy(
    *,
    thread_id: str,
    messages: Sequence[LegacyChatMessage],
    exchange_summaries: Sequence[LegacyExchangeSummary] = (),
    max_recent_turns: int = DEFAULT_RECENT_TURN_LIMIT,
) -> ConversationSnapshot:
    """Build the persisted conversation projection from legacy chat rows.

    ``summary_version`` is the highest ``exchange_index`` (monotonic), and
    ``built_through_message_id`` is the assistant message id of that exchange —
    the same monotonic pointer the rolling-summary persistence owns.
    """
    summaries = _ordered_summaries(exchange_summaries)
    summary_version = 0
    built_through: str | None = None
    if summaries:
        last = summaries[-1]
        if last.exchange_index < 0:
            raise ConversationAdapterError("legacy exchange_index must be non-negative")
        summary_version = last.exchange_index
        built_through = last.assistant_message_id or last.user_message_id
    snapshot = ConversationSnapshot(
        contract_version=CONTRACT_VERSION,
        thread_id=thread_id,
        summary_version=summary_version,
        built_through_message_id=built_through,
        context=context_from_legacy(
            messages=messages,
            exchange_summaries=summaries,
            max_recent_turns=max_recent_turns,
        ),
    )
    validate_conversation_snapshot(snapshot)
    return snapshot
