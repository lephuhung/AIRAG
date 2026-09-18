"""Discovery plan Task 5 — exact checkpoint revision and root-slot wiring."""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.base import ContractModel
from app.services.agents.v2.contracts.state import (
    CHECKPOINT_SCHEMA_REVISION,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.validation import (
    IncompatibleCheckpointError,
    migrate_checkpoint_payload,
    validate_checkpoint_payload,
)
from app.services.agents.v2.discovery_bootstrap.contracts import (
    DiscoveryNeed,
    ResearchTargetSelection,
)

from . import factories

_REQUIRED_REVISION_TWO_KEYS = (
    "checkpoint_schema_revision",
    "discovery_need",
    "discovery",
    "document_selection_clarification",
    "research_target_selection",
)


def _legacy_payload() -> dict:
    state = _fresh_state()
    payload = {key: value for key, value in state.items() if key != "synthesis"}
    for key in _REQUIRED_REVISION_TWO_KEYS:
        payload.pop(key, None)
    return payload


def _fresh_state() -> dict:
    from app.services.agents.supervisor_v2 import build_initial_v2_state

    return dict(build_initial_v2_state(request=factories.request_context()))


def test_missing_discriminator_and_no_new_keys_migrates() -> None:
    migrated = migrate_checkpoint_payload(_legacy_payload())
    assert migrated["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION
    assert migrated["synthesis"] is None
    for key in _REQUIRED_REVISION_TWO_KEYS[1:]:
        assert migrated[key] is None
    validate_checkpoint_payload(migrated)


def test_revision_one_without_new_keys_migrates() -> None:
    payload = {**_legacy_payload(), "checkpoint_schema_revision": 1}
    migrated = migrate_checkpoint_payload(payload)
    assert migrated["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION
    validate_checkpoint_payload(migrated)


def test_missing_discriminator_with_one_new_key_rejected() -> None:
    payload = {**_legacy_payload(), "discovery": None}
    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


def test_missing_discriminator_with_all_new_keys_rejected() -> None:
    payload = {
        **_legacy_payload(),
        "discovery_need": None,
        "discovery": None,
        "document_selection_clarification": None,
        "research_target_selection": None,
    }
    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


def test_revision_one_with_new_key_rejected() -> None:
    payload = {
        **_legacy_payload(),
        "checkpoint_schema_revision": 1,
        "research_target_selection": None,
    }
    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


@pytest.mark.parametrize("key", _REQUIRED_REVISION_TWO_KEYS)
def test_revision_two_missing_each_required_key_rejected(key: str) -> None:
    payload = _fresh_state()
    payload.pop(key)
    with pytest.raises(IncompatibleCheckpointError):
        validate_checkpoint_payload(migrate_checkpoint_payload(payload))


def test_unknown_revision_rejected() -> None:
    payload = {**_fresh_state(), "checkpoint_schema_revision": 7}
    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


def test_revision_two_round_trips_unchanged() -> None:
    payload = _fresh_state()
    migrated = migrate_checkpoint_payload(payload)
    assert migrated == payload
    validate_checkpoint_payload(migrated)


def test_fresh_state_writes_revision_and_slots() -> None:
    state = _fresh_state()
    assert state["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION
    for key in _REQUIRED_REVISION_TWO_KEYS[1:]:
        assert key in state
        assert state[key] is None


def test_fresh_turn_hygiene_clears_new_slots() -> None:
    import asyncio

    from app.services.agents import supervisor_v2 as sv2

    state = {
        **_fresh_state(),
        "checkpoint_schema_revision": CHECKPOINT_SCHEMA_REVISION,
        "discovery_need": DiscoveryNeed(required=False, reason=None),
        "research_target_selection": factories.target_selection(),
    }
    update = asyncio.run(
        sv2.SUPERVISOR_V2_NODES["context"](state, None)  # type: ignore[arg-type]
    )
    assert update["final_response"] is None
    assert update["synthesis"] is None
    assert update["discovery_need"] is None
    assert update["discovery"] is None
    assert update["document_selection_clarification"] is None
    assert update["research_target_selection"] is None


def test_slot_coercion_returns_concrete_models_after_json_roundtrip() -> None:
    from app.services.agents.supervisor_v2 import normalize_checkpoint_state

    state = {
        **_fresh_state(),
        "discovery_need": DiscoveryNeed(
            required=True, reason="unresolved_document_slot"
        ),
        "research_target_selection": factories.target_selection(),
    }
    payload = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, ContractModel)
            else value
        )
        for key, value in state.items()
    }
    resumed = normalize_checkpoint_state(payload)
    assert isinstance(resumed["discovery_need"], DiscoveryNeed)
    assert isinstance(
        resumed["research_target_selection"], ResearchTargetSelection
    )
    assert resumed["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION


@pytest.mark.parametrize(
    "revision", [1, True, 2.0, "2", 3, None]
)
def test_payload_discriminator_must_be_exact_int_two(revision: object) -> None:
    payload = _fresh_state()
    if revision is None:
        payload.pop("checkpoint_schema_revision")
    else:
        payload["checkpoint_schema_revision"] = revision
    with pytest.raises(IncompatibleCheckpointError):
        validate_checkpoint_payload(payload)


@pytest.mark.parametrize("revision", [True, 1.0, "1", "2", 2.0, None])
def test_migration_rejects_non_integer_discriminator(revision: object) -> None:
    payload = {**_legacy_payload(), "checkpoint_schema_revision": revision}
    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


@pytest.mark.parametrize("revision", [1, True, "2", 2.0, 3, None])
def test_aggregate_state_requires_exact_revision(revision: object) -> None:
    from app.services.agents.v2.contracts.validation import (
        validate_supervisor_state,
    )

    state = _fresh_state()
    if revision is None:
        state.pop("checkpoint_schema_revision")
    else:
        state["checkpoint_schema_revision"] = revision
    with pytest.raises(IncompatibleCheckpointError):
        validate_supervisor_state(state)  # type: ignore[arg-type]


def test_typed_state_declares_new_keys() -> None:
    keys = set(SupervisorV2State.__required_keys__)
    for key in _REQUIRED_REVISION_TWO_KEYS:
        assert key in keys
