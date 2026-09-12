"""Short-term discourse state and its persisted snapshot (spec §8.2).

``ConversationContext`` is discourse state ("nghị định này", "file thứ hai").
Rolling-summary persistence — not semantic context — owns optimistic locking, so
``summary_version``/``built_through_message_id``/``thread_id`` appear only on the
snapshot. Pending unresolved user questions are owned by ``ClarificationRequest``;
there is deliberately no duplicate ``open_questions`` collection.
"""
from __future__ import annotations

from typing import Literal

from .base import ContractModel, ContractVersion

EntityKind = Literal["document", "person", "section", "concept"]


class EntityReference(ContractModel):
    """Spec §8.2: minimal pointer to a named entity in the discourse."""

    ref_id: str
    kind: EntityKind
    label: str


class ActiveEntity(EntityReference):
    """Spec §8.2: an entity currently active in the discourse window.

    The spec references ``ActiveEntity`` without defining it; the minimal shape a
    coreference resolver needs is the entity's reference identity, so it currently
    adds no field of its own.
    """


class ConversationTurn(ContractModel):
    """Spec §8.2: one recent turn supplied to the context builder.

    The shape is not defined by the spec; the minimal shape is the turn role and
    its text, because chat persistence remains authoritative for full history.
    """

    role: Literal["user", "assistant", "system"]
    content: str


class ConversationContext(ContractModel):
    """Spec §8.2: short-term discourse state."""

    summary: str
    active_entities: tuple[ActiveEntity, ...]
    last_focus: EntityReference | None
    recent_turns: tuple[ConversationTurn, ...]


class ConversationSnapshot(ContractModel):
    """Spec §3/§8.2: the persisted conversation boundary with CAS metadata."""

    contract_version: ContractVersion
    thread_id: str
    summary_version: int
    built_through_message_id: str | None
    context: ConversationContext
