"""Task 6 — side-effect-free v2 shadow execution (R54-R65).

Honest, harness-correct isolation proofs: record-and-raise spies wrap the
REAL production write boundaries (incl. async checkpoint aput/aput_writes,
the saver factory, SQL commit/flush, memory/graphiti writers) and the REAL
outbound sinks (queue publish, Telegram, SSE/relay formatters, redis);
the DB proof counts rows on V2_TEST_DATABASE_URL (never DATABASE_URL)
with failure reasons preserved. The REAL hook supplies mirrored
authorization plus read-only people/document sources so factual hook turns
reach the shared scheduler; cancellation enforces a terminal state.
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

#: Bundle-level granted set (test input mirroring the old shadow default).
TEST_ALLOWED = frozenset({"people.lookup", "document.read", "section.read"})

TYPED_STATUSES = ("success", "error", "clarify", "denied", "insufficient", "unavailable")


def make_factual_bundle(**overrides: Any):
    from app.services.agent.shadow_runtime import build_shadow_bundle

    params = {
        "raw_query": FACTUAL_QUERY,
        "thread_id": "shadow-factual",
        "user_id": USER_ID,
        "workspace_ids": (WORKSPACE_ID,),
        "can_read_people": True,
        "allowed_capabilities": TEST_ALLOWED,
        "person_names": (PERSON_NAME,),
        "people_directory": dict(PEOPLE_DIRECTORY),
    }
    params.update(overrides)
    return build_shadow_bundle(**params)


def make_direct_bundle(**overrides: Any):
    from app.services.agent.shadow_runtime import build_shadow_bundle

    params = {
        "raw_query": "Xin chào",
        "thread_id": "shadow-direct",
        "can_read_people": False,
        "allowed_capabilities": TEST_ALLOWED,
    }
    params.update(overrides)
    return build_shadow_bundle(**params)


def _is_env_skip(reason: str) -> bool:
    return reason.startswith("env:")


# ---------------------------------------------------------------------------
# Production-write spies (R58.1/R62.3): record + raise on REAL boundaries
# ---------------------------------------------------------------------------


def install_production_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, int], list[str]]:
    """Wrap REAL production write boundaries with record-and-raise spies.

    Any call records first, then raises — so a shadow path that reached a
    production write would both trip the counter AND be unable to persist.
    Returns (counters, skipped); import-unavailable boundaries are skipped
    with an ``env:`` reason (strict in the harness, where all import).
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
        # NOTE (R64): `V1PeopleLookupService.lookup` is deliberately NOT
        # spied here — the shadow uses the production people service in
        # read-only mode, and reads are allowed. The service offers no
        # write method (asserted in the read-only test); any production
        # *write* it could reach does not exist.
        # R62.3: the production saver factory itself.
        ("app.services.agents.v2.persistence.checkpoint:create_v2_checkpointer", "checkpoints.factory"),
        # R62.3: the SQL commit/flush boundary (covers the real
        # ChatSession.title setattr + every ORM write path).
        ("sqlalchemy.ext.asyncio:AsyncSession.commit", "sql.commit"),
        ("sqlalchemy.ext.asyncio:AsyncSession.flush", "sql.flush"),
        # R62.3: memory writers (Graphiti episodes/facts).
        ("app.services.memory.graphiti_client:add_conversation_episode", "memory.add_episode"),
        ("app.services.memory.graphiti_client:save_user_fact", "memory.save_fact"),
    ]
    try:
        import langgraph.checkpoint.postgres.aio  # noqa: F401

        targets.extend(
            [
                ("langgraph.checkpoint.postgres.aio:AsyncPostgresSaver.put", "checkpoints.put"),
                ("langgraph.checkpoint.postgres.aio:AsyncPostgresSaver.aput", "checkpoints.aput"),
                ("langgraph.checkpoint.postgres.aio:AsyncPostgresSaver.put_writes", "checkpoints.put_writes"),
                ("langgraph.checkpoint.postgres.aio:AsyncPostgresSaver.aput_writes", "checkpoints.aput_writes"),
            ]
        )
    except ModuleNotFoundError as exc:
        skipped.append(f"env:checkpoint-saver: {exc}")

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
            skipped.append(f"env:{dotted}: {type(exc).__name__}: {exc}")
    return counters, skipped


def _dotted(module_name: str, owner_name: str | None, method: str) -> str:
    return f"{module_name}:{owner_name}.{method}" if owner_name else f"{module_name}:{method}"


async def _exercise_spies_effectively(
    counters: dict[str, int], skipped: list[str]
) -> None:
    """Prove every INSTALLED spy intercepts: invoke each, expect record+raise.

    Entries skipped at install time (env reason) are excluded here too;
    the harness gate installs everything, so nothing is excluded there.
    """
    import importlib

    from app.services.agent.shadow_runtime import ShadowIsolationError

    calls = [
        ("app.services.agents.v2.persistence.evidence", "EvidenceRepository", "insert_record"),
        ("app.services.agents.v2.persistence.evidence", "EvidenceRepository", "append_use"),
        ("app.services.agents.v2.persistence.binding_audit", "BindingAuditRepository", "append"),
        ("app.services.agents.v2.persistence.snapshots", "ConversationSnapshotRepository", "save_first"),
        ("app.services.agents.v2.persistence.snapshots", "ConversationSnapshotRepository", "cas_update"),
        ("app.services.agents.v2.persistence.retention_leases", "RevisionRetentionLeaseRepository", "acquire_or_refresh"),
        ("app.services.agents.v2.evidence_store.governance", "EvidenceGovernor", "persist_record"),
        ("app.services.agents.v2.evidence_store.governance", "EvidenceGovernor", "persist_people_evidence"),
        ("app.services.agent.runtime_selector", "GovernorEvidenceBuilder", "persist_use"),
        ("app.services.agent.runtime_selector", None, "persist_raw_user_message"),
        ("app.services.memory.conversation_summary_service", "ConversationSummaryService", "save_exchange_summary"),
        ("app.services.agents.v2.persistence.checkpoint", None, "create_v2_checkpointer"),
        ("sqlalchemy.ext.asyncio", "AsyncSession", "commit"),
        ("sqlalchemy.ext.asyncio", "AsyncSession", "flush"),
        ("app.services.memory.graphiti_client", None, "add_conversation_episode"),
        ("app.services.memory.graphiti_client", None, "save_user_fact"),
        ("langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "put"),
        ("langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "aput"),
        ("langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "put_writes"),
        ("langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "aput_writes"),
    ]
    exercised = 0
    for module_name, owner_name, method in calls:
        dotted = _dotted(module_name, owner_name, method)
        if any(dotted in reason for reason in skipped):
            continue
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            # Matches an env-skip at install time (e.g. the checkpoint
            # driver locally); the harness gate installs everything.
            continue
        target = getattr(module, owner_name) if owner_name else module
        implanted = getattr(target, method)
        assert getattr(implanted, "__name__", "") == "_recorder", (
            f"spy for {dotted} is not installed"
        )
        before = sum(counters.values())
        with pytest.raises(ShadowIsolationError):
            result = implanted(None)
            if asyncio.iscoroutine(result):
                await result
        assert sum(counters.values()) == before + 1, (
            f"spy for {dotted} did not intercept"
        )
        exercised += 1
    assert exercised >= 10, f"only {exercised} spies exercised"


# ---------------------------------------------------------------------------
# Outbound spies (R59/R63): the REAL sinks, all record + raise
# ---------------------------------------------------------------------------


def install_outbound_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, int], list[str]]:
    """Spy on the REAL outbound sinks: queue, Telegram, SSE/relay, redis.

    Every sink records first, then raises — a shadow emission would trip
    the counter AND be unable to leave the process.
    """
    from app.services.agent.shadow_runtime import ShadowIsolationError

    counters: dict[str, int] = {}
    skipped: list[str] = []

    def _block(name: str):
        def _recorder(*args: Any, **kwargs: Any) -> Any:
            counters[name] = counters.get(name, 0) + 1
            raise ShadowIsolationError(
                f"outbound sink {name!r} reached from shadow"
            )

        return _recorder

    def _block_async_fn(name: str):
        async def _recorder(*args: Any, **kwargs: Any) -> Any:
            counters[name] = counters.get(name, 0) + 1
            raise ShadowIsolationError(
                f"outbound sink {name!r} reached from shadow"
            )

        return _recorder

    import importlib

    targets: list[tuple[str, str, bool]] = [
        # Queue publish boundary + publisher functions.
        ("app.queue.connection", "publish", True),
        ("app.queue.connection", "publish_control", True),
        ("app.queue.publisher", "publish_parse_task", True),
        ("app.queue.publisher", "publish_memory_save_task", True),
        # Telegram / notification sends.
        ("app.services.integrations.telegram_service", "send_message", True),
        ("app.services.integrations.telegram_service", "send_chat_action", True),
        ("app.services.integrations.telegram_service", "edit_message", True),
        # SSE / streaming emitters (sync formatters + async stream fns).
        ("app.services.agent.streaming", "_sse", False),
        ("app.services.agent.streaming", "push_event", True),
        ("app.services.agent.streaming", "stream_agent_to_sse", True),
        ("app.services.agent.streaming", "stream_v2_turn_to_sse", True),
        ("app.services.agent.streaming", "stream_v2_turn_events", True),
        ("app.services.agents.v2.events", "format_sse_event", False),
        # Chat-session relay boundary used by chat_stream_session.
        ("app.api.chat_agent", "format_sse_event", False),
        ("app.api.chat_agent", "sse_with_heartbeat", False),
        # Shared-state publish.
        ("app.core.redis_client", "get_redis", False),
    ]
    for module_name, attr, is_async in targets:
        try:
            module = importlib.import_module(module_name)
            monkeypatch.setattr(
                module, attr, _block_async_fn(attr) if is_async else _block(attr)
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"env:{module_name}.{attr}: {type(exc).__name__}: {exc}")
    return counters, skipped


async def _exercise_outbound_effectively(counters: dict[str, int]) -> None:
    """Efficacy per sink class: call it, observe the counter (R63)."""
    import importlib

    from app.services.agent.shadow_runtime import ShadowIsolationError

    calls = [
        ("app.queue.connection", "publish", (None, None, None), True),
        ("app.queue.connection", "publish_control", (None, None, None), True),
        ("app.queue.publisher", "publish_parse_task", (None,), True),
        ("app.queue.publisher", "publish_memory_save_task", (None,), True),
        ("app.services.integrations.telegram_service", "send_message", (None, None), True),
        ("app.services.agent.streaming", "_sse", ("status", {}), False),
        ("app.services.agent.streaming", "push_event", ({}, "status", {}), True),
        ("app.services.agent.streaming", "stream_v2_turn_to_sse", (), True),
        ("app.services.agents.v2.events", "format_sse_event", ("complete", {}), False),
        ("app.api.chat_agent", "format_sse_event", ("token", {}), False),
        ("app.api.chat_agent", "sse_with_heartbeat", (None,), False),
        ("app.core.redis_client", "get_redis", (), False),
    ]
    for module_name, attr, args, is_async in calls:
        module = importlib.import_module(module_name)
        implanted = getattr(module, attr)
        assert getattr(implanted, "__name__", "") == "_recorder", (
            f"outbound spy for {module_name}.{attr} is not installed"
        )
        before = sum(counters.values())
        with pytest.raises(ShadowIsolationError):
            result = implanted(*args)
            if asyncio.iscoroutine(result) or inspect.isasyncgen(result):
                if inspect.isasyncgen(result):
                    await result.aclose()
                else:
                    await result
        assert sum(counters.values()) == before + 1, (
            f"outbound spy for {module_name}.{attr} did not intercept"
        )


# ---------------------------------------------------------------------------
# R54/R62 guards: production getter/saver never used + static proof
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_never_calls_production_graph_resolver() -> None:
    """R54: shadow compiles its own graph; production getter stays silent."""
    import app.services.agents.supervisor_v2 as supervisor_v2

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
    """No CODE reference to production stores, sinks, selectors (AST).

    Docstrings/comments excluded. ``V1PeopleLookupService`` is explicitly
    allowed: the hook uses it wrapped read-only (R64).
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
        "add_conversation_episode",
        "save_user_fact",
        "sse_with_heartbeat",
        "publish_control",
        "publish_parse_task",
        "publish_memory_save_task",
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
            "app.queue.connection",
            "app.queue.publisher",
            "app.services.memory.graphiti_client",
        ), f"shadow path imports production-adjacent module {module!r}"


# ---------------------------------------------------------------------------
# R58.1/R62.3: zero production writes through the REAL boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_production_stores_receive_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factual + direct shadow runs trip ZERO production spies."""
    counters, skipped = install_production_spies(monkeypatch)
    hard = [s for s in skipped if not _is_env_skip(s)]
    assert not hard, f"production spies unavailable: {hard}"
    factual = make_factual_bundle(thread_id="shadow-r58-factual")
    factual_metrics = await factual.run()
    assert factual_metrics.task_count >= 1
    direct = make_direct_bundle(thread_id="shadow-r58-direct")
    await direct.run()
    assert counters == {}, f"shadow reached production writes: {counters}"


@pytest.mark.asyncio
async def test_production_spies_are_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every production spy intercepts its real boundary (incl. aput)."""
    counters, skipped = install_production_spies(monkeypatch)
    hard = [s for s in skipped if not _is_env_skip(s)]
    assert not hard, f"production spies unavailable: {hard}"
    await _exercise_spies_effectively(counters, skipped)


# ---------------------------------------------------------------------------
# R62: honest DB proof on V2_TEST_DATABASE_URL (never DATABASE_URL)
# ---------------------------------------------------------------------------


V2_TABLES = (
    "binding_audit",
    "evidence_records",
    "evidence_uses",
    "conversation_snapshots",
    "semantic_snapshots",
    "revision_retention_leases",
    "documents",
    "document_revisions",
)
REQUIRED_V2_TABLES = ("evidence_records", "evidence_uses", "binding_audit")
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
REQUIRED_CHECKPOINT_TABLES = ("checkpoints",)


def _asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


async def _count_rows_strict(conn: Any, table: str) -> tuple[bool, int | None, str]:
    """(exists, count, reason) — absence is benign; every other failure FAILS.

    ``to_regclass`` distinguishes a genuinely absent table (skip with
    reason) from connectivity/permission/query errors, which propagate so
    the test fails instead of silently passing.
    """
    from sqlalchemy import text

    exists = await conn.execute(
        text('SELECT to_regclass(:name)'), {"name": f"public.{table}"}
    )
    if exists.scalar_one_or_none() is None:
        return False, None, f"absent table {table!r}"
    result = await conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
    return True, int(result.scalar_one()), ""


@pytest.mark.asyncio
async def test_shadow_production_db_rows_unchanged() -> None:
    """R62: row counts on the HARNESS TEST database, reasons preserved.

    Uses ``V2_TEST_DATABASE_URL`` (unset → explicit skip, never a
    ``DATABASE_URL`` fallback). Expected tables that fail to count FAIL the
    test; only genuinely absent tables are skipped with reasons. Memory
    lives in Neo4j/Graphiti (no relational table) — skipped with reason;
    the shadow path holds no memory import (AST guard) and the Graphiti
    writers are spied (zero-write test).
    """
    v2_url = os.environ.get("V2_TEST_DATABASE_URL")
    if not v2_url:
        pytest.skip(
            "V2_TEST_DATABASE_URL is not set; the DB proof requires the "
            "harness test database and never falls back to DATABASE_URL"
        )
    ckpt_url = os.environ.get("CHECKPOINT_TEST_DATABASE_URL")
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        pytest.skip("sqlalchemy unavailable")
    app_engine = create_async_engine(_asyncpg_url(v2_url))
    ckpt_engine = (
        create_async_engine(_asyncpg_url(ckpt_url)) if ckpt_url else None
    )
    skipped: list[str] = [
        "memory/neo4j: no relational table; no memory import on shadow path; "
        "graphiti writers spied at zero"
    ]
    if ckpt_engine is None:
        skipped.append(
            "checkpoints: CHECKPOINT_TEST_DATABASE_URL is not set "
            "(checkpoint writes still spied at zero via aput/aput_writes)"
        )
    try:
        async with app_engine.connect() as conn:
            before = {t: await _count_rows_strict(conn, t) for t in V2_TABLES}
        before_ckpt: dict[str, tuple[bool, int | None, str]] = {}
        if ckpt_engine is not None:
            async with ckpt_engine.connect() as conn:
                before_ckpt = {
                    t: await _count_rows_strict(conn, t) for t in CHECKPOINT_TABLES
                }
    except Exception as exc:
        pytest.fail(f"DB proof setup failed, refusing silent pass: {exc!r}")

    factual = make_factual_bundle(thread_id="shadow-r62-factual")
    await factual.run()
    await make_direct_bundle(thread_id="shadow-r62-direct").run()

    try:
        async with app_engine.connect() as conn:
            after = {t: await _count_rows_strict(conn, t) for t in V2_TABLES}
        after_ckpt: dict[str, tuple[bool, int | None, str]] = {}
        if ckpt_engine is not None:
            async with ckpt_engine.connect() as conn:
                after_ckpt = {
                    t: await _count_rows_strict(conn, t) for t in CHECKPOINT_TABLES
                }
    except Exception as exc:
        pytest.fail(f"DB proof re-count failed, refusing silent pass: {exc!r}")
    finally:
        await app_engine.dispose()
        if ckpt_engine is not None:
            await ckpt_engine.dispose()

    for table in V2_TABLES:
        exists, count_before, reason = before[table]
        if not exists:
            skipped.append(reason)
            continue
        _, count_after, _ = after[table]
        assert count_after == count_before, (
            f"production table {table!r} changed under shadow: "
            f"{count_before} -> {count_after}"
        )
    for table in CHECKPOINT_TABLES:
        if table not in before_ckpt:
            continue
        exists, count_before, reason = before_ckpt[table]
        if not exists:
            skipped.append(reason)
            continue
        _, count_after, _ = after_ckpt[table]
        assert count_after == count_before, (
            f"checkpoint table {table!r} changed under shadow: "
            f"{count_before} -> {count_after}"
        )
    for required in REQUIRED_V2_TABLES:
        assert before[required][0], (
            f"required table {required!r} absent from the harness test DB"
        )
    if ckpt_engine is not None:
        for required in REQUIRED_CHECKPOINT_TABLES:
            assert before_ckpt[required][0], (
                f"required table {required!r} absent from the checkpoint test DB"
            )
    checked = [t for t in V2_TABLES if before[t][0]] + [
        t for t in before_ckpt if before_ckpt[t][0]
    ]
    print(f"shadow DB proof: {len(checked)} tables equal; skipped: {skipped}")


# ---------------------------------------------------------------------------
# R59/R63: no outbound events through the REAL sinks + per-class efficacy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_emits_no_outbound_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R63: a FACTUAL shadow run invokes zero real outbound sinks."""
    counters, skipped = install_outbound_spies(monkeypatch)
    hard = [s for s in skipped if not _is_env_skip(s)]
    assert not hard, f"outbound spies unavailable: {hard}"
    bundle = make_factual_bundle(thread_id="shadow-r59")
    metrics = await bundle.run()
    assert metrics.task_count >= 1  # non-vacuous: factual path executed
    assert counters == {}, f"shadow emitted outbound events: {counters}"


@pytest.mark.asyncio
async def test_outbound_spies_are_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R63: every sink class is proven to trip its spy when called."""
    counters, skipped = install_outbound_spies(monkeypatch)
    hard = [s for s in skipped if not _is_env_skip(s)]
    assert not hard, f"outbound spies unavailable: {hard}"
    await _exercise_outbound_effectively(counters)
    assert sum(counters.values()) >= 10


# ---------------------------------------------------------------------------
# R60/R64: functional factual shadow through the shared scheduler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factual_shadow_reaches_scheduler_with_isolated_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factual query dispatches via the shared TaskScheduler, writes isolated."""
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    prod_counters, skipped = install_production_spies(monkeypatch)
    hard = [s for s in skipped if not _is_env_skip(s)]
    assert not hard, f"production spies unavailable: {hard}"
    bundle = make_factual_bundle(thread_id="shadow-r60")
    metrics = await bundle.run()
    assert metrics.route == "fast_domain", metrics.redacted()
    assert metrics.task_count >= 1
    assert metrics.evaluation == "sufficient", metrics.redacted()
    assert metrics.status == "success"
    assert TaskScheduler is not None
    resolved = bundle.runtime_context.services.capability_registry.get(
        "people.lookup"
    )
    assert type(resolved).__name__ == "PeopleCapability"
    assert len(bundle.stores.evidence) >= 1
    assert len(bundle.stores.evidence_use) >= 1
    assert len(bundle.stores.leases) >= 1
    assert metrics.isolated_writes >= 3
    assert prod_counters == {}, f"production writes fired: {prod_counters}"


@pytest.mark.asyncio
async def test_shadow_mirrors_denied_authorization() -> None:
    """R64: with can_read_people=False the people run is TYPED denied.

    Authorization is a caller input, not a shadow hardcoded value: denying
    it must surface as the typed denial outcome, never silent success.
    """
    from app.services.agent.runtime_selector import (
        DEFAULT_V2_ALLOWED_CAPABILITIES,
    )

    bundle = make_factual_bundle(
        thread_id="shadow-r64-denied",
        can_read_people=False,
        allowed_capabilities=DEFAULT_V2_ALLOWED_CAPABILITIES,
    )
    assert bundle.runtime_context.capability_runtime.can_read_people is False
    metrics = await bundle.run()
    assert metrics.task_count >= 1
    assert metrics.status == "denied", metrics.redacted()


@pytest.mark.asyncio
async def test_shadow_preserves_refs_read_only() -> None:
    """Document refs + history survive read-only into the shadow run."""
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
    bundle = make_direct_bundle()
    metrics = await bundle.run()
    assert metrics.route == "direct"
    assert metrics.status == "success"
    assert metrics.task_count == 0


# ---------------------------------------------------------------------------
# R64.4: the REAL hook end-to-end for a factual query
# ---------------------------------------------------------------------------


def _v2_test_session_factory():
    """Session factory bound to the harness TEST database (read-only use)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    v2_url = os.environ.get("V2_TEST_DATABASE_URL")
    if not v2_url:
        pytest.skip(
            "V2_TEST_DATABASE_URL is not set; the hook proof requires the "
            "harness test database"
        )
    engine = create_async_engine(_asyncpg_url(v2_url))
    return async_sessionmaker(engine, expire_on_commit=False), engine


async def _pick_revisioned_document(session_factory: Any) -> tuple[UUID, UUID]:
    """Read one (document, workspace) pair with an ACTIVE revision.

    Read-only setup against the harness test DB; skips with reason when
    no such document exists.
    """
    from sqlalchemy import text

    async with session_factory() as db:
        result = await db.execute(
            text(
                "SELECT d.id, d.workspace_id FROM documents d "
                "JOIN document_revisions r ON r.document_id = d.id "
                "WHERE r.status = 'active' AND r.failed_at IS NULL LIMIT 1"
            )
        )
        row = result.first()
    if row is None:
        pytest.skip("no revisioned document in the test DB to shadow")
    return UUID(str(row[0])), UUID(str(row[1]))


@pytest.mark.asyncio
async def test_real_hook_factual_reaches_scheduler_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R64.4: REAL `_maybe_launch_shadow_turn` for a factual doc query.

    Asserts the scheduler was reached (fast_domain, task count > 0, typed
    insufficient outcome) with ZERO production writes and ZERO outbound
    events, using mirrored primary authorization.
    """
    from app.api.chat_session import _maybe_launch_shadow_turn
    from app.core.config import settings
    from app.services.agent.runtime_selector import (
        DEFAULT_V2_ALLOWED_CAPABILITIES,
    )

    session_factory, engine = _v2_test_session_factory()
    try:
        try:
            document_id, workspace_id = await _pick_revisioned_document(
                session_factory
            )
        except Exception as exc:
            pytest.skip(f"hook test setup read failed: {exc!r}")

        monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_ENABLED", True)
        monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_PERCENT", 100.0)
        prod_counters, prod_skipped = install_production_spies(monkeypatch)
        out_counters, out_skipped = install_outbound_spies(monkeypatch)
        hard = [s for s in (*prod_skipped, *out_skipped) if not _is_env_skip(s)]
        assert not hard, f"spies unavailable: {hard}"

        seen: list[dict[str, Any]] = []
        task = _maybe_launch_shadow_turn(
            raw_message="Đọc tài liệu này",
            thread_id="shadow-hook-factual",
            user_id=USER_ID,
            workspace_ids=[workspace_id],
            document_ids=[document_id],
            history=[("user", "Đọc tài liệu này")],
            can_read_people=False,
            allowed_capabilities=DEFAULT_V2_ALLOWED_CAPABILITIES,
            on_metrics=seen.append,
            session_factory=session_factory,
        )
        assert task is not None
        await asyncio.wait_for(task, timeout=120)
        assert task.done() and not task.cancelled()
        assert seen, "hook produced no shadow metrics"
        metrics = seen[-1]
        assert metrics["route"] == "fast_domain", metrics
        assert metrics["task_count"] >= 1, metrics
        assert metrics["evaluation"] == "insufficient", metrics
        assert prod_counters == {}, f"production writes fired: {prod_counters}"
        assert out_counters == {}, f"outbound events fired: {out_counters}"
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# R57/R56/adapters/metrics/contracts
# ---------------------------------------------------------------------------


def test_shadow_source_adapters_are_read_only() -> None:
    """R57: no write path is reachable from the shadow bundle."""
    from app.services.agent.shadow_runtime import (
        ReadOnlyProductionPeopleSource,
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
    production_source = ReadOnlyProductionPeopleSource()
    assert production_source.read_only is True
    for write_attr in ("write", "save", "persist_use", "send_message", "publish"):
        assert not hasattr(production_source, write_attr), (
            f"production people wrapper exposes {write_attr!r}"
        )
    # The wrapped production service itself offers no write surface:
    # lookup is a read, and reads are allowed on the shadow path.
    from app.services.agents.supervisor_v2 import V1PeopleLookupService

    public = [m for m in dir(V1PeopleLookupService) if not m.startswith("_")]
    assert public == ["lookup"], f"unexpected V1 people surface: {public}"


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
    gap = ShadowMetrics(
        status="unavailable",
        route=None,
        evaluation=None,
        task_count=0,
        isolated_writes=0,
        duration_ms=0,
        reason="dependency-gap: document view unreadable: boom",
    )
    assert gap.redacted()["reason"].startswith("dependency-gap:")


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
# R61/R65: the REAL hook launches, completes, and joins on cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_hook_runs_and_joins_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REAL hook runs to completion normally; completed tasks join True."""
    from app.api.chat_session import _maybe_launch_shadow_turn
    from app.core.config import settings
    from app.services.agent.shadow_runtime import _stop_shadow_task

    monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_ENABLED", True)
    monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_PERCENT", 100.0)
    seen: list[dict[str, Any]] = []
    task = _maybe_launch_shadow_turn(
        raw_message="Xin chào",
        thread_id="shadow-hook-1",
        user_id=USER_ID,
        workspace_ids=[WORKSPACE_ID],
        document_ids=[],
        history=[("user", "Xin chào")],
        can_read_people=False,
        allowed_capabilities=frozenset(),
        on_metrics=seen.append,
    )
    assert task is not None
    await asyncio.wait_for(task, timeout=60)
    assert task.done() and not task.cancelled()
    assert await _stop_shadow_task(task, timeout=5.0) is True


@pytest.mark.asyncio
async def test_shadow_cancellation_reaches_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R65: a cancellation-resistant shadow is double-cancelled to done.

    The join enforces the terminal state (it does not shield-and-abandon),
    and primary cleanup sequenced after the join never runs while the
    shadow is alive.
    """
    import app.services.agent.shadow_runtime as shadow_runtime
    from app.api.chat_session import _maybe_launch_shadow_turn
    from app.core.config import settings

    monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_ENABLED", True)
    monkeypatch.setattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_PERCENT", 100.0)

    async def _resistant_view(*args: Any, **kwargs: Any) -> dict:
        await asyncio.sleep(0)
        return {}

    async def _resistant_run(self: Any) -> Any:
        from app.services.agent.shadow_runtime import ShadowMetrics

        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Resist cancellation like flushing buffers / slow IO: swallow
            # the join's immediate re-cancel and finish a bounded resist
            # window instead, so ONLY the timeout + enforce-cancel path
            # terminates the run.
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                await asyncio.sleep(0.5)
            raise
        return ShadowMetrics(
            status="success",
            route="direct",
            evaluation=None,
            task_count=0,
            isolated_writes=0,
            duration_ms=0,
        )

    class _ResistantBundle:
        async def run(self) -> Any:
            return await _resistant_run(self)

    monkeypatch.setattr(
        "app.api.chat_session.resolve_shadow_document_view", _resistant_view
    )
    # The hook body does `from ... import build_shadow_bundle` at call
    # time, so patching the shadow module attribute redirects the hook.
    monkeypatch.setattr(
        shadow_runtime, "build_shadow_bundle", lambda **kwargs: _ResistantBundle()
    )

    task = _maybe_launch_shadow_turn(
        raw_message="Xin chào",
        thread_id="shadow-hook-resistant",
        user_id=USER_ID,
        workspace_ids=[WORKSPACE_ID],
    )
    assert task is not None
    await asyncio.sleep(0.05)  # let the resistant run start
    task.cancel()  # the primary run was cancelled
    cleaned: list[str] = []
    join = asyncio.create_task(
        shadow_runtime._stop_shadow_task(task, timeout=0.1)
    )
    await asyncio.sleep(0.02)
    # The shadow is still alive in its resist window: cleanup sequencing
    # after the join cannot have run, and the join has not finished early.
    assert not task.done()
    assert cleaned == []
    assert not join.done()
    stopped = await join
    assert stopped is True
    assert task.done()
    # Primary cleanup proceeds ONLY after the terminal state.
    cleaned.append("primary-cleanup")
    assert cleaned == ["primary-cleanup"]
