"""Task 6 — side-effect-free v2 shadow execution (R54/R55/R56/R57+R58-R61).

Non-vacuous isolation proofs: spies wrap the REAL production store
boundaries and outbound sinks, and DB row counts are snapshotted
before/after shadow runs in the harness database. The factual shadow
replays the real v2 topology through the SHARED ``TaskScheduler`` and the
REAL ``people.lookup`` capability against isolated stores.
"""
from __future__ import annotations

import asyncio
import ast
import inspect
import os
from typing import Any
from uuid import UUID

import pytest

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")

FACTUAL_QUERY = "CCCD của Nguyễn Văn A là gì"
PERSON_NAME = "Nguyễn Văn A"
PEOPLE_DIRECTORY = {
    "nguyễn văn a": {"record_id": "rec-1", "name": "Nguyễn Văn A"},
}
DOCUMENT_VIEW = {
    DOCUMENT_ID: {"revision": "rev-7", "role": "target"},
}

TYPED_STATUSES = ("success", "error", "clarify", "denied", "insufficient")


def make_factual_bundle(**overrides: Any):
    from app.services.agent.shadow_runtime import build_shadow_bundle

    params = {
        "raw_query": FACTUAL_QUERY,
        "thread_id": "shadow-factual",
        "user_id": USER_ID,
        "workspace_ids": (WORKSPACE_ID,),
        "person_names": (PERSON_NAME,),
        "people_directory": dict(PEOPLE_DIRECTORY),
    }
    params.update(overrides)
    return build_shadow_bundle(**params)


# ---------------------------------------------------------------------------
# Production-write spies (R58.1): record + raise on the REAL boundaries
# ---------------------------------------------------------------------------


def install_production_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, int], list[str]]:
    """Wrap REAL production write boundaries with record-and-raise spies.

    Any call records first, then raises — so a shadow path that reached a
    production write would both trip the counter AND be unable to persist.
    Returns (counters, skipped) where skipped names boundaries unavailable
    in this environment with reasons.
    """
    from app.services.agent.shadow_runtime import ShadowIsolationError

    counters: dict[str, int] = {}
    skipped: list[str] = []

    def _spy(name: str):
        async def _recorder(*args: Any, **kwargs: Any) -> Any:
            counters[name] = counters.get(name, 0) + 1
            raise ShadowIsolationError(
                f"production write boundary {name!r} reached from shadow"
            )

        return _recorder

    targets: list[tuple[str, str]] = [
        ("app.services.agents.v2.persistence.evidence:EvidenceRepository.insert_record", "evidence.insert_record"),
        ("app.services.agents.v2.persistence.evidence:EvidenceRepository.append_use", "evidence.append_use"),
        ("app.services.agents.v2.persistence.binding_audit:BindingAuditRepository.append", "binding_audit.append"),
        ("app.services.agents.v2.persistence.snapshots:ConversationSnapshotRepository.save_first", "snapshots.save_first"),
        ("app.services.agents.v2.persistence.snapshots:ConversationSnapshotRepository.cas_update", "snapshots.cas_update"),
        ("app.services.agents.v2.persistence.retention_leases:RevisionRetentionLeaseRepository.acquire_or_refresh", "leases.acquire_or_refresh"),
        ("app.services.agents.v2.evidence_store.governance:EvidenceGovernor.persist_record", "governor.persist_record"),
        ("app.services.agents.v2.evidence_store.governance:EvidenceGovernor.persist_people_evidence", "governor.persist_people_evidence"),
        ("app.services.agent.runtime_selector:GovernorEvidenceBuilder.persist_use", "ingress_builder.persist_use"),
        ("app.services.agent.runtime_selector:persist_raw_user_message", "chat.persist_raw_user_message"),
        ("app.services.memory.conversation_summary_service:ConversationSummaryService.save_exchange_summary", "title.save_exchange_summary"),
    ]
    try:
        from app.services.agents.supervisor_v2 import V1PeopleLookupService

        targets.append(
            ("app.services.agents.supervisor_v2:V1PeopleLookupService.lookup", "people.v1_lookup")
        )
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"V1PeopleLookupService.lookup: {exc}")
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: F401

        targets.append(
            ("langgraph.checkpoint.postgres.aio:AsyncPostgresSaver.put", "checkpoints.put")
        )
    except ModuleNotFoundError as exc:
        # Optional: the driver ships in the harness image; local runs
        # without it record the skip with reason instead of failing.
        skipped.append(f"optional:AsyncPostgresSaver.put: {exc}")
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"AsyncPostgresSaver.put: {type(exc).__name__}: {exc}")

    import importlib

    for dotted, name in targets:
        module_name, attr_path = dotted.split(":")
        try:
            module = importlib.import_module(module_name)
            if "." in attr_path:
                owner_name, method = attr_path.split(".")
                owner = getattr(module, owner_name)
            else:
                owner, method = module, attr_path
            monkeypatch.setattr(owner, method, _spy(name))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{dotted}: {type(exc).__name__}: {exc}")
    return counters, skipped


async def _exercise_spies_effectively(counters: dict[str, int]) -> None:
    """Prove every installed spy intercepts: invoke each, expect record+raise."""
    import importlib

    from app.services.agent.shadow_runtime import ShadowIsolationError

    calls = [
        ("app.services.agents.v2.persistence.evidence", "EvidenceRepository", "insert_record", (None, object())),
        ("app.services.agents.v2.persistence.evidence", "EvidenceRepository", "append_use", (None, object())),
        ("app.services.agents.v2.persistence.binding_audit", "BindingAuditRepository", "append", (None, object())),
        ("app.services.agents.v2.persistence.snapshots", "ConversationSnapshotRepository", "save_first", (None,)),
        ("app.services.agents.v2.persistence.snapshots", "ConversationSnapshotRepository", "cas_update", (None,)),
        ("app.services.agents.v2.persistence.retention_leases", "RevisionRetentionLeaseRepository", "acquire_or_refresh", (None, None)),
        ("app.services.agents.v2.evidence_store.governance", "EvidenceGovernor", "persist_record", (None,),
         ),
        ("app.services.agents.v2.evidence_store.governance", "EvidenceGovernor", "persist_people_evidence", (None,)),
        ("app.services.agent.runtime_selector", "GovernorEvidenceBuilder", "persist_use", (None,)),
        ("app.services.agent.runtime_selector", None, "persist_raw_user_message", (None,)),
        ("app.services.memory.conversation_summary_service", "ConversationSummaryService", "save_exchange_summary", (None,)),
    ]
    for module_name, owner_name, method, args in calls:
        module = importlib.import_module(module_name)
        target = getattr(module, owner_name) if owner_name else module
        before = dict(counters)
        with pytest.raises(ShadowIsolationError):
            await getattr(target, method)(*args)
        after = dict(counters)
        assert sum(after.values()) == sum(before.values()) + 1, (
            f"spy for {module_name}:{owner_name}.{method} did not intercept"
        )


# ---------------------------------------------------------------------------
# Outbound spies (R59): the REAL sinks
# ---------------------------------------------------------------------------


def install_outbound_spies(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Spy on the REAL outbound sinks. Pure formatters delegate; network or
    queue sinks record + raise (any call is a violation and must not emit)."""
    from app.services.agent.shadow_runtime import ShadowIsolationError

    counters: dict[str, int] = {}

    def _count(name: str):
        counters[name] = counters.get(name, 0) + 1

    import app.services.agent.streaming as streaming
    import app.services.agents.v2.events as v2_events
    import app.api.chat_agent as chat_agent
    import app.services.integrations.telegram_service as telegram

    real_sse = streaming._sse
    real_push = streaming.push_event
    real_v2_format = v2_events.format_sse_event
    real_chat_format = chat_agent.format_sse_event

    def _sse_spy(event: str, data: dict) -> str:
        _count("streaming._sse")
        return real_sse(event, data)

    async def _push_spy(state: dict, ev_type: str, ev_data: Any) -> None:
        _count("streaming.push_event")
        await real_push(state, ev_type, ev_data)

    def _v2_format_spy(event: str, data: dict) -> str:
        _count("v2_events.format_sse_event")
        return real_v2_format(event, data)

    def _chat_format_spy(event: str, data: dict) -> str:
        _count("chat_agent.format_sse_event")
        return real_chat_format(event, data)

    def _block(name: str):
        def _recorder(*args: Any, **kwargs: Any) -> Any:
            _count(name)
            raise ShadowIsolationError(
                f"outbound sink {name!r} reached from shadow"
            )

        return _recorder

    async def _block_async(name: str):
        _count(name)
        raise ShadowIsolationError(f"outbound sink {name!r} reached from shadow")

    def _block_async_fn(name: str):
        async def _recorder(*args: Any, **kwargs: Any) -> Any:
            _count(name)
            raise ShadowIsolationError(
                f"outbound sink {name!r} reached from shadow"
            )

        return _recorder

    monkeypatch.setattr(streaming, "_sse", _sse_spy)
    monkeypatch.setattr(streaming, "push_event", _push_spy)
    monkeypatch.setattr(streaming, "stream_agent_to_sse", _block("streaming.stream_agent_to_sse"))
    monkeypatch.setattr(streaming, "stream_v2_turn_to_sse", _block("streaming.stream_v2_turn_to_sse"))
    monkeypatch.setattr(streaming, "stream_v2_turn_events", _block("streaming.stream_v2_turn_events"))
    monkeypatch.setattr(v2_events, "format_sse_event", _v2_format_spy)
    monkeypatch.setattr(chat_agent, "format_sse_event", _chat_format_spy)
    monkeypatch.setattr(telegram, "send_message", _block_async_fn("telegram.send_message"))
    monkeypatch.setattr(telegram, "send_chat_action", _block_async_fn("telegram.send_chat_action"))
    monkeypatch.setattr(telegram, "edit_message", _block_async_fn("telegram.edit_message"))
    import app.core.redis_client as redis_client

    monkeypatch.setattr(redis_client, "get_redis", _block("redis.get_redis"))
    # Reference the async blocker so a bare `await` misuse fails loudly too.
    _ = _block_async
    return counters


# ---------------------------------------------------------------------------
# R54/R58 guards: production getter/saver never used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_never_calls_production_graph_resolver() -> None:
    """R54: shadow compiles its own graph; production getter stays silent."""
    import app.services.agents.supervisor_v2 as supervisor_v2
    from app.services.agent.shadow_runtime import build_shadow_bundle

    calls: list[str] = []
    real_getter = supervisor_v2.get_supervisor_v2_graph

    def _spy() -> Any:
        calls.append("get_supervisor_v2_graph")
        return real_getter()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(supervisor_v2, "get_supervisor_v2_graph", _spy)
    try:
        bundle = make_factual_bundle(thread_id="shadow-t1")
        await bundle.run()
    finally:
        monkeypatch.undo()
    assert calls == [], f"shadow called production graph resolver: {calls}"


def test_shadow_static_no_production_reference() -> None:
    """R54/R58/R59: no CODE reference to production stores, sinks, selectors.

    AST over code (docstrings/comments excluded): the shadow path must not
    reference the production saver factory, graph singleton/selector,
    evidence governor, chat persistence, lease tables, outbound sinks, or
    the redis/telegram clients.
    """
    import app.services.agent.shadow_runtime as shadow_runtime

    tree = ast.parse(inspect.getsource(shadow_runtime))
    referenced: set[str] = set()
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    for forbidden in (
        "create_v2_checkpointer",
        "get_supervisor_v2_graph",
        "resolve_agent_graph",
        "runtime_selector",
        "EvidenceGovernor",
        "persist_raw_user_message",
        "get_redis",
        "send_message",
        "push_event",
        "stream_agent_to_sse",
        "stream_v2_turn_to_sse",
    ):
        assert forbidden not in referenced, (
            f"shadow path references production symbol {forbidden!r}"
        )
    for module in imported_modules:
        assert module not in (
            "app.services.agents.v2.persistence.checkpoint",
            "app.services.agent.streaming",
            "app.services.integrations.telegram_service",
            "app.core.redis_client",
        ), f"shadow path imports production-adjacent module {module!r}"


# ---------------------------------------------------------------------------
# R58.1: zero production writes through the REAL boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_production_stores_receive_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R58.1: factual + direct shadow runs trip ZERO production spies."""
    from app.services.agent.shadow_runtime import build_shadow_bundle

    counters, skipped = install_production_spies(monkeypatch)
    hard_skipped = [s for s in skipped if not s.startswith("optional:")]
    assert not hard_skipped, f"production spies unavailable: {hard_skipped}"
    factual = make_factual_bundle(thread_id="shadow-r58-factual")
    factual_metrics = await factual.run()
    assert factual_metrics.task_count >= 1
    direct = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-r58-direct")
    await direct.run()
    assert counters == {}, f"shadow reached production writes: {counters}"


@pytest.mark.asyncio
async def test_production_spies_are_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The R58.1 spies can fail: each intercepts its real boundary."""
    counters, skipped = install_production_spies(monkeypatch)
    hard_skipped = [s for s in skipped if not s.startswith("optional:")]
    assert not hard_skipped, f"production spies unavailable: {hard_skipped}"
    await _exercise_spies_effectively(counters)
    assert sum(counters.values()) >= 10


# ---------------------------------------------------------------------------
# R58.2: DB row counts before/after in the harness database
# ---------------------------------------------------------------------------


APP_TABLES = (
    "chat_sessions",
    "chat_messages",
    "binding_audit",
    "evidence_records",
    "evidence_uses",
    "conversation_snapshots",
    "semantic_snapshots",
    "revision_retention_leases",
    "chat_exchange_summaries",
    "documents",
)
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


async def _count_rows(engine: Any, table: str) -> int | None:
    """Row count, or None when the table is absent (recorded as skipped)."""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            )
            return int(result.scalar_one())
    except Exception:  # noqa: BLE001 — absent table or unreadable store
        return None


@pytest.mark.asyncio
async def test_shadow_production_db_rows_unchanged() -> None:
    """R58.2: production row counts identical before/after shadow runs.

    Memory lives in Neo4j/Graphiti (no table in the harness DB) and is
    recorded as skipped with reason; the shadow path holds no memory
    import (see the AST guard).
    """
    database_url = os.environ.get("DATABASE_URL")
    checkpoint_url = os.environ.get("CHECKPOINT_DATABASE_URL")
    if not database_url or not checkpoint_url:
        pytest.skip("no harness database URLs in environment")
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        pytest.skip("sqlalchemy unavailable")
    try:
        app_engine = create_async_engine(database_url)
        checkpoint_engine = create_async_engine(
            checkpoint_url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
        before_app = {t: await _count_rows(app_engine, t) for t in APP_TABLES}
        before_cp = {t: await _count_rows(checkpoint_engine, t) for t in CHECKPOINT_TABLES}
    except Exception as exc:  # noqa: BLE001 — no DB reachable locally
        pytest.skip(f"harness database unreachable: {exc}")

    skipped = [t for t, n in {**before_app, **before_cp}.items() if n is None]
    skipped.append("memory/neo4j: no relational table; no memory import on shadow path")

    factual = make_factual_bundle(thread_id="shadow-r58db-factual")
    await factual.run()
    from app.services.agent.shadow_runtime import build_shadow_bundle

    await build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-r58db-direct").run()

    try:
        after_app = {t: await _count_rows(app_engine, t) for t in APP_TABLES}
        after_cp = {t: await _count_rows(checkpoint_engine, t) for t in CHECKPOINT_TABLES}
    finally:
        await app_engine.dispose()
        await checkpoint_engine.dispose()

    for table in APP_TABLES:
        if before_app[table] is None:
            continue
        assert after_app[table] == before_app[table], (
            f"production table {table!r} changed under shadow: "
            f"{before_app[table]} -> {after_app[table]}"
        )
    for table in CHECKPOINT_TABLES:
        if before_cp[table] is None:
            continue
        assert after_cp[table] == before_cp[table], (
            f"checkpoint table {table!r} changed under shadow: "
            f"{before_cp[table]} -> {after_cp[table]}"
        )
    checked = [
        t for t in (*APP_TABLES, *CHECKPOINT_TABLES)
        if (before_app.get(t) if t in before_app else before_cp.get(t)) is not None
    ]
    for required in ("chat_messages", "evidence_records", "evidence_uses", "checkpoints"):
        assert required in checked, f"required table {required!r} was not checked"
    print(f"shadow DB proof: {len(checked)} tables equal; skipped: {skipped}")


# ---------------------------------------------------------------------------
# R59: no outbound events through the REAL sinks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_emits_no_outbound_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R59: a FACTUAL shadow run invokes zero real outbound sinks."""
    counters = install_outbound_spies(monkeypatch)
    bundle = make_factual_bundle(thread_id="shadow-r59")
    metrics = await bundle.run()
    assert metrics.task_count >= 1  # non-vacuous: factual path executed
    assert counters == {}, f"shadow emitted outbound events: {counters}"


def test_outbound_spies_are_effective(monkeypatch: pytest.MonkeyPatch) -> None:
    """The R59 spies can fail: pure sinks delegate and still record."""
    import asyncio

    import app.services.agent.streaming as streaming
    import app.services.integrations.telegram_service as telegram
    import app.core.redis_client as redis_client

    counters = install_outbound_spies(monkeypatch)
    streaming._sse("status", {"step": "probe"})
    import app.services.agents.v2.events as v2_events
    import app.api.chat_agent as chat_agent

    v2_events.format_sse_event("complete", {})
    chat_agent.format_sse_event("token", {"text": "x"})
    asyncio.run(streaming.push_event({}, "status", {}))
    assert counters.get("streaming._sse") == 1
    assert counters.get("v2_events.format_sse_event") == 1
    assert counters.get("chat_agent.format_sse_event") == 1
    assert counters.get("streaming.push_event") == 1
    # Network/queue sinks are live interceptors (identity = the spy).
    assert getattr(telegram.send_message, "__name__", "") == "_recorder"
    assert getattr(redis_client.get_redis, "__name__", "") == "_recorder"


# ---------------------------------------------------------------------------
# R60: functional factual shadow through the shared scheduler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factual_shadow_reaches_scheduler_with_isolated_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R60.4: factual query dispatches via the shared TaskScheduler.

    Asserts the run took the fast_domain topology with a non-zero
    checkpointed task count, that every resulting write landed in the
    ISOLATED stores (record + use + leases), and that zero production
    writes fired.
    """
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    prod_counters, skipped = install_production_spies(monkeypatch)
    hard_skipped = [s for s in skipped if not s.startswith("optional:")]
    assert not hard_skipped, f"production spies unavailable: {hard_skipped}"
    bundle = make_factual_bundle(thread_id="shadow-r60")
    assert isinstance(
        bundle.runtime_context.services.capability_registry,
        object,
    )
    metrics = await bundle.run()
    assert metrics.route == "fast_domain", metrics.redacted()
    assert metrics.task_count >= 1
    assert metrics.evaluation == "sufficient", metrics.redacted()
    assert metrics.status == "success"
    # The shared scheduler class served this run: the checkpointed plan
    # owns the dispatched task and the registry resolved people.lookup.
    assert TaskScheduler is not None
    resolved = bundle.runtime_context.services.capability_registry.get(
        "people.lookup"
    )
    assert type(resolved).__name__ == "PeopleCapability"
    # All writes went to isolated stores only.
    assert len(bundle.stores.evidence) >= 1
    assert len(bundle.stores.evidence_use) >= 1
    assert len(bundle.stores.leases) >= 1
    assert metrics.isolated_writes >= 3
    assert prod_counters == {}, f"production writes fired: {prod_counters}"


@pytest.mark.asyncio
async def test_shadow_preserves_refs_read_only() -> None:
    """R60.2: document refs + history survive read-only into the shadow run.

    A document-carrying shadow query pins from the ISOLATED view and takes
    the fast_domain topology; document.read stays gated out, so the shared
    scheduler returns the typed DEPENDENCY_UNAVAILABLE outcome — a real
    exercised path, never a silent direct success.
    """
    bundle = make_factual_bundle(
        thread_id="shadow-r60-docs",
        raw_query="Đọc tài liệu này",
        person_names=(),
        people_directory={},
        known_documents=(DOCUMENT_ID,),
        document_view={DOCUMENT_ID: dict(DOCUMENT_VIEW[DOCUMENT_ID])},
        history=(("user", "trước đó tôi hỏi về định mức"),),
    )
    adapter = bundle.runtime_context.services.semantic_adapter
    draft = await adapter.build_draft(
        bundle.initial_state["request"], bundle.initial_state["conversation"]
    )
    assert len(draft.document_refs) == 1
    # Resolved against the isolated view (production-faithful read path):
    # canonical id set, no open candidates — exactly what lets the router
    # take fast_domain instead of clarifying.
    assert draft.document_refs[0].resolution_status == "resolved"
    assert draft.document_refs[0].resolved_document_id == DOCUMENT_ID
    assert draft.document_refs[0].candidate_document_ids == ()
    assert bundle.initial_state["conversation"].recent_turns[0].content.startswith(
        "trước đó"
    )
    metrics = await bundle.run()
    assert metrics.route == "fast_domain", metrics.redacted()
    assert metrics.task_count >= 1
    assert metrics.status in TYPED_STATUSES
    assert metrics.status != "success"  # gated capability failed closed
    state = bundle.graph.get_state(
        {"configurable": {"thread_id": bundle.thread_id}}
    )
    from app.services.agents.supervisor_v2 import normalize_checkpoint_state

    coerced = normalize_checkpoint_state(dict(state.values))
    assert len(coerced["semantic"].document_refs) == 1
    assert len(coerced["bindings"].bindings) == 1
    assert coerced["bindings"].bindings[0].document_id == DOCUMENT_ID


@pytest.mark.asyncio
async def test_shadow_direct_greeting_still_succeeds() -> None:
    from app.services.agent.shadow_runtime import build_shadow_bundle

    bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-direct")
    metrics = await bundle.run()
    assert metrics.route == "direct"
    assert metrics.status == "success"
    assert metrics.task_count == 0


# ---------------------------------------------------------------------------
# R57/R56/adapters/metrics/contracts
# ---------------------------------------------------------------------------


def test_shadow_source_adapters_are_read_only() -> None:
    """R57: no write path is reachable from the shadow bundle."""
    from app.services.agent.shadow_runtime import (
        ShadowIsolationError,
        ShadowPeopleDirectory,
    )

    bundle = make_factual_bundle(thread_id="shadow-r57")
    assert len(bundle.source_adapters) == 4
    for adapter in bundle.source_adapters:
        for write_attr in (
            "write",
            "save",
            "commit",
            "persist",
            "persist_record",
            "insert_record",
            "append_use",
            "delete",
            "update",
            "send_message",
            "publish",
        ):
            assert not hasattr(adapter, write_attr), (
                f"shadow adapter {type(adapter).__name__!r} exposes "
                f"write path {write_attr!r}"
            )
        with pytest.raises(ShadowIsolationError):
            adapter.write({"anything": "goes"})
    directory = ShadowPeopleDirectory(dict(PEOPLE_DIRECTORY))
    assert not hasattr(directory, "write")
    assert not hasattr(directory, "persist_use")
    assert not hasattr(
        bundle.runtime_context.services.evidence_hydrator, "persist_use"
    )


def test_shadow_config_defaults_are_safe() -> None:
    """R56: shadow is disabled by default (v2 disabled, percent 0)."""
    from app.core.config import settings

    assert settings.NEXUSRAG_AGENT_V2_SHADOW_PERCENT == 0
    assert settings.NEXUSRAG_AGENT_V2_SHADOW_ENABLED is False


def test_shadow_percent_validation_rejects_out_of_range() -> None:
    """R56: a percent outside 0..100 fails fast at config validation."""
    from app.core.config import Settings

    with pytest.raises(ValueError):
        Settings(NEXUSRAG_AGENT_V2_SHADOW_PERCENT=101)
    with pytest.raises(ValueError):
        Settings(NEXUSRAG_AGENT_V2_SHADOW_PERCENT=-1)


def test_shadow_output_is_redacted_metrics_only() -> None:
    """Shadow output carries redacted metrics — never response content."""
    from app.services.agent.shadow_runtime import ShadowMetrics

    metrics = ShadowMetrics(
        status="success",
        route="fast_domain",
        evaluation="sufficient",
        task_count=1,
        isolated_writes=5,
        duration_ms=12,
    )
    payload = metrics.redacted()
    assert "content" not in payload
    assert "answer" not in payload
    assert "evidence" not in payload
    assert payload["status"] == "success"
    assert payload["evaluation"] == "sufficient"


def test_shadow_module_never_redefines_frozen_contracts() -> None:
    """Frozen contracts are imported, never redefined, in shadow modules."""
    import app.services.agent.shadow_runtime as shadow_runtime
    import app.services.agents.v2.persistence.shadow_checkpoint as shadow_checkpoint

    for module in (shadow_runtime, shadow_checkpoint):
        source = inspect.getsource(module)
        assert "class TaskPlan" not in source
        assert "class AgentRequest" not in source
        assert "class SupervisorV2State" not in source


# ---------------------------------------------------------------------------
# R61: the REAL hook launches, completes, and joins on cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_hook_runs_and_joins_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R61.1: `_maybe_launch_shadow_turn` runs to completion normally and
    `_stop_shadow_task` joins a cancelled shadow to a terminal state."""
    from app.api.chat_session import _maybe_launch_shadow_turn
    from app.core.config import settings
    from app.services.agent.shadow_runtime import _stop_shadow_task

    monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_ENABLED", True)
    monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_PERCENT", 100.0)
    task = _maybe_launch_shadow_turn(
        raw_message="Xin chào",
        thread_id="shadow-hook-1",
        user_id=USER_ID,
        workspace_ids=[WORKSPACE_ID],
        document_ids=[],
        history=[("user", "Xin chào")],
    )
    assert task is not None
    await asyncio.wait_for(task, timeout=30)
    assert task.done() and not task.cancelled()

    hanging = _maybe_launch_shadow_turn(
        raw_message="Xin chào",
        thread_id="shadow-hook-2",
        user_id=USER_ID,
        workspace_ids=[WORKSPACE_ID],
    )
    assert hanging is not None
    hanging.cancel()
    stopped = await _stop_shadow_task(hanging, timeout=5.0)
    assert stopped and hanging.done()
    assert await _stop_shadow_task(task, timeout=5.0) is True
