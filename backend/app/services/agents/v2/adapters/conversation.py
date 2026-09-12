"""Conversation port: legacy chat rows → v2 discourse contracts (spec §8.2).

This is a **server-internal** port, not the agent-facing tool gateway. The chat
database stays the authoritative conversation history; this adapter only
translates already-loaded legacy rows into the minimal v2 shapes:

- ``ChatMessage`` rows → ``ConversationTurn`` / ``ConversationContext``;
- ``ExchangeSummary`` rows → the rolling ``summary``, the monotonic
  ``summary_version`` (``exchange_index``) and ``built_through_message_id`` of
  ``ConversationSnapshot``, plus ``ActiveEntity`` labels from ``key_entities``.

``last_focus`` is left ``None``: the legacy rows carry no last-focus owner, and
the v2 conversation builder derives it from the turns, so this adapter does not
invent one. Persistence/optimistic locking stays with
``persistence.snapshots.ConversationSnapshotRepository``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..contracts.base import CONTRACT_VERSION
from ..contracts.conversation import (
    ActiveEntity,
    ConversationContext,
    ConversationSnapshot,
    ConversationTurn,
)
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
        turns.append(ConversationTurn(role=message.role, content=message.content))
    return tuple(turns)


def _active_entities(
    summaries: Sequence[LegacyExchangeSummary],
) -> tuple[ActiveEntity, ...]:
    entities: list[ActiveEntity] = []
    seen: set[str] = set()
    for summary in summaries:
        for label in summary.key_entities or ():
            if not isinstance(label, str) or not label.strip():
                raise ConversationAdapterError(
                    "legacy exchange key_entities must contain non-blank strings"
                )
            if label in seen:
                continue
            seen.add(label)
            entities.append(ActiveEntity(ref_id=label, kind="concept", label=label))
    return tuple(entities)


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
    summaries = _ordered_summaries(exchange_summaries)
    context = ConversationContext(
        summary=_summary_text(summaries),
        active_entities=_active_entities(summaries),
        last_focus=None,
        recent_turns=_recent_turns(messages, max_recent_turns),
    )
    validate_conversation_context(context)
    return context


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
