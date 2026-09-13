

# ---------------------------------------------------------------------------
# Task 3 fix-2 — pre-handler recovery of abandoned running claims (I-R1)
# ---------------------------------------------------------------------------
# The atomic claim (fix-1) no-ops on ANY non-pending stage, including
# ``running``. Because the running mark commits BEFORE the work, a durable
# ``running`` row outlives its owner whenever the broker redelivers after a
# crash (``message.redelivered``) or whenever the queue retry note failed
# before the retry publish (the redelivery carries the bounded
# ``x-retry-count`` header). Without recovery the redelivery no-ops AND is
# acked — the stage strands in ``running`` forever.
#
# The tests below drive the REAL ``_consume_on_channel`` consume branch
# (fake channel/queue/message, real DB, real handlers) and pin:
# - redelivered-after-crash recovery for all four workers;
# - retry-header self-heal after a swallowed retry-note failure;
# - terminal (completed/failed) redelivery stays a no-op (acked);
# - plain duplicates without an ownership-loss signal stay a no-op and
#   never touch another stage;
# - recovery infrastructure failure raises (no handler run, no ack);
# - the running mark is durably committed before heavy work (M-R1,
#   second-session observation);
# - the real timeout/error branch call sites map the exact stage (M-R2);
# - exhausted-note atomicity rolls the stage edge back when the revision
#   terminalization fails (M-R4).

import asyncio as _asyncio
import json as _json

import pytest

from app.services.agents.v2.persistence import document_revisions as _revisions_mod
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
)
from tests.workers.test_revision_pipeline import (
    _caption_payload,
    _caption_worker,
    _embed_payload,
    _embed_worker,
    _FakeEmbedder,
    _FakeVectorStore,
    _handler_maker,
    _handler_mirrors,
    _handler_revision_status,
    _handler_setup_full,
    _handler_stage_rows,
    _kg_payload,
    _kg_worker,
    _parse_payload,
    _parse_worker,
    _patch_caption,
    _patch_embed,
    _patch_finalize_recorder,
    _patch_kg,
    _patch_parse,
    _RecordingCaptionStore,
    _RecordingKGService,
    _RecordingParseStore,
    _seed_caption_table,
    _seed_raw_chunks,
)


class _ConsumeStop(Exception):
    """Test-only sentinel: the fake queue has no more messages."""


class _FakeConsumeMessage:
    def __init__(self, *, body: bytes, headers: dict, redelivered: bool = False):
        self.body = body
        self.headers = headers
        self.redelivered = redelivered
        self.acked = False
        self.processed = False
        self.rejected_requeue = None
        self.reject_calls = 0
        self.exc_rejected = False

    def process(self, *, requeue=False, ignore_processed=True):
        msg = self

        class _Ctx:
            async def __aenter__(self):
                return msg

            async def __aexit__(self, exc_type, exc, tb):
                # Mirror aio-pika ProcessContext with
                # ignore_processed=True: clean exit without an explicit
                # ack/reject auto-acks; a raise rejects (no ack).
                if exc_type is None:
                    if not msg.processed:
                        await msg.ack()
                    return False
                if not msg.processed:
                    msg.exc_rejected = True
                return False

        return _Ctx()

    async def ack(self):
        self.acked = True
        self.processed = True

    async def reject(self, requeue=False):
        self.rejected_requeue = requeue
        self.reject_calls += 1
        self.processed = True


class _FakeConsumeExchange:
    def __init__(self):
        self.published = []

    async def publish(self, message, routing_key=None):
        self.published.append((routing_key, message))


class _FakeConsumeQueue:
    def __init__(self, messages):
        self._pending = list(messages)
        self.bound = []

    async def bind(self, exchange, routing_key=None):
        self.bound.append(routing_key)

    def iterator(self):
        queue = self

        class _Iter:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if queue._pending:
                    return queue._pending.pop(0)
                raise _ConsumeStop("test: no more messages")

        return _Iter()


class _FakeConsumeChannel:
    def __init__(self, queue):
        self._queue = queue
        self.is_closed = False
        self.default_exchange = _FakeConsumeExchange()

    async def set_qos(self, prefetch_count=1):
        return None

    async def declare_exchange(self, *args, **kwargs):
        return _FakeConsumeExchange()

    async def declare_queue(self, *args, **kwargs):
        return self._queue

    async def close(self):
        self.is_closed = True


class _DummyConsumeConn:
    async def channel(self):
        raise AssertionError("test: retry consumer must stay disabled")


async def _drive_consume(
    monkeypatch,
    maker,
    *,
    queue_name,
    exchange_name,
    routing_key,
    payloads,
    handler,
    headers_list=None,
    redelivered_list=None,
    handler_timeout=60,
):
    """Drive the real ``_consume_on_channel`` over fake broker objects."""
    import app.core.database as _dbmod
    from app.queue import connection as _conn_mod

    messages = []
    for i, payload in enumerate(payloads):
        headers = (
            headers_list[i] if headers_list is not None else {"x-retry-count": 0}
        )
        redelivered = (
            redelivered_list[i] if redelivered_list is not None else False
        )
        messages.append(
            _FakeConsumeMessage(
                body=_json.dumps(payload).encode(),
                headers=dict(headers),
                redelivered=redelivered,
            )
        )
    queue = _FakeConsumeQueue(messages)
    channel = _FakeConsumeChannel(queue)

    async def _instant_sleep(delay, *args, **kwargs):
        return None

    monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
    monkeypatch.setattr(_dbmod, "async_session_maker", maker)
    real_flag = _conn_mod._retry_consumer_started
    _conn_mod._retry_consumer_started = True
    try:
        await _conn_mod._consume_on_channel(
            _DummyConsumeConn(),
            channel,
            exchange_name,
            queue_name,
            routing_key,
            handler,
            1,
            handler_timeout,
        )
    finally:
        _conn_mod._retry_consumer_started = real_flag
    return messages, channel


async def _consume_preset_running(maker, rev_id, stage):
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        _, claimed = await repo.claim_stage_running(rev_id, stage)
        assert claimed is True
        await db.commit()


async def _consume_setup_stage(monkeypatch, maker, doc_id, ws, rev_id, stage):
    """Patch heavy work for one stage; return (handler, payload, work_probe)."""
    if stage == "parse":
        store = _RecordingParseStore()
        _patch_parse(monkeypatch, maker, store)
        return (
            _parse_worker.handle_parse,
            _parse_payload(doc_id, ws, rev_id),
            lambda: (len(store.downloads), len(store.uploads)),
            store,
        )
    if stage == "embed":
        await _seed_raw_chunks(maker, doc_id)
        embedder, store = _FakeEmbedder(), _FakeVectorStore()
        _patch_embed(monkeypatch, maker, embedder, store)
        _patch_finalize_recorder(monkeypatch, _embed_worker)
        return (
            _embed_worker.handle_embed,
            _embed_payload(doc_id, ws, rev_id),
            lambda: (len(embedder.calls), len(store.added)),
            None,
        )
    if stage == "caption":
        await _seed_caption_table(maker, doc_id, rev_id)
        store = _RecordingCaptionStore()
        _patch_caption(monkeypatch, maker, store)
        _patch_finalize_recorder(monkeypatch, _caption_worker)
        return (
            _caption_worker.handle_caption,
            _caption_payload(doc_id, ws, rev_id),
            lambda: (len(store.downloads), len(store.uploads)),
            store,
        )
    assert stage == "kg"
    service = _RecordingKGService()
    _patch_kg(monkeypatch, maker, service)
    _patch_finalize_recorder(monkeypatch, _kg_worker)
    return (
        _kg_worker.handle_kg,
        _kg_payload(doc_id, ws, rev_id),
        lambda: (len(service.ingests),),
        None,
    )


def _consume_route(stage, ws):
    if stage == "parse":
        return ("hrag.parse", "hrag.parse", "parse")
    if stage == "embed":
        return ("hrag.embed", "hrag.embed", "embed")
    if stage == "caption":
        return ("hrag.caption", "hrag.caption", "caption")
    return (f"hrag.kg.{ws}", "hrag.kg", str(ws))


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["parse", "embed", "caption", "kg"])
async def test_task3_consume_redelivered_recovers_running_for_all_workers(
    async_engine, document_factory, monkeypatch, stage
):
    """Broker redelivery after a crash re-runs the abandoned stage (I-R1)."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="A" * 64)
    handler, payload, work_probe, _ = await _consume_setup_stage(
        monkeypatch, maker, doc_id, ws, rev_id, stage
    )
    await _consume_preset_running(maker, rev_id, stage)
    queue_name, exchange_name, routing_key = _consume_route(stage, ws)

    messages, _channel = await _drive_consume(
        monkeypatch,
        maker,
        queue_name=queue_name,
        exchange_name=exchange_name,
        routing_key=routing_key,
        payloads=[payload],
        handler=handler,
        headers_list=[{"x-retry-count": 0}],
        redelivered_list=[True],
    )

    # The abandoned running claim was recovered (pending) then re-claimed:
    # real work ran and the stage completed with attempt 2.
    assert all(n > 0 for n in work_probe())
    rows = await _handler_stage_rows(maker, rev_id)
    assert rows[stage] == ("completed", 2)
    assert messages[0].acked is True
    assert messages[0].reject_calls == 0


@pytest.mark.asyncio
async def test_task3_consume_retry_header_self_heals_after_failed_note(
    async_engine, document_factory, monkeypatch
):
    """A swallowed retry-note failure self-heals on the retry delivery."""
    from app.queue import connection as _conn_mod

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="B" * 64)
    store = _RecordingParseStore(fail_download=True)
    _patch_parse(monkeypatch, maker, store)

    async def _failed_note(*args, **kwargs):
        return False

    monkeypatch.setattr(_conn_mod, "note_stage_retry_pending", _failed_note)
    handler = _parse_worker.handle_parse
    payload = _parse_payload(doc_id, ws, rev_id)

    # Delivery 1: handler raises; the retry note fails (DB blip, swallowed);
    # the retry is still published and the message acked. The row is left
    # running(1) — exactly the strand I-R1 describes.
    messages1, channel1 = await _drive_consume(
        monkeypatch,
        maker,
        queue_name="hrag.parse",
        exchange_name="hrag.parse",
        routing_key="parse",
        payloads=[payload],
        handler=handler,
        headers_list=[{"x-retry-count": 0}],
        redelivered_list=[False],
    )
    assert messages1[0].acked is True
    assert await _handler_stage_rows(maker, rev_id) == {
        "parse": ("running", 1),
        "embed": ("pending", 0),
        "caption": ("pending", 0),
        "kg": ("pending", 0),
    }
    assert len(channel1.default_exchange.published) == 1

    # Delivery 2: the retry header marks an ownership-loss signal even
    # though redelivered=False. Pre-handler recovery moves running→pending
    # (via the repository directly, not the failed note helper) and the
    # worker claim executes the stage.
    store.fail_download = False
    messages2, _channel2 = await _drive_consume(
        monkeypatch,
        maker,
        queue_name="hrag.parse",
        exchange_name="hrag.parse",
        routing_key="parse",
        payloads=[payload],
        handler=handler,
        headers_list=[{"x-retry-count": 1}],
        redelivered_list=[False],
    )
    assert messages2[0].acked is True
    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["parse"] == ("completed", 2)
    # Two download attempts total: delivery 1 failed AFTER starting the
    # download (minio boom), delivery 2 succeeded after the self-heal.
    assert len(store.downloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["completed", "failed"])
async def test_task3_consume_terminal_redelivery_stays_noop(
    async_engine, document_factory, monkeypatch, preset
):
    """Terminal stages stay a no-op (acked) even with a redelivery signal."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="C" * 64)
    handler, payload, work_probe, _ = await _consume_setup_stage(
        monkeypatch, maker, doc_id, ws, rev_id, "parse"
    )
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        await repo.claim_stage_running(rev_id, "parse")
        if preset == "completed":
            await repo.mark_stage_completed(rev_id, "parse")
        else:
            await repo.mark_stage_failed(rev_id, "parse", failure_class="E")
        await db.commit()
    before_rows = await _handler_stage_rows(maker, rev_id)
    before_mirrors = await _handler_mirrors(maker, doc_id)

    messages, _channel = await _drive_consume(
        monkeypatch,
        maker,
        queue_name="hrag.parse",
        exchange_name="hrag.parse",
        routing_key="parse",
        payloads=[payload],
        handler=handler,
        headers_list=[{"x-retry-count": 1}],
        redelivered_list=[True],
    )

    assert work_probe() == (0, 0)
    assert await _handler_stage_rows(maker, rev_id) == before_rows
    assert await _handler_mirrors(maker, doc_id) == before_mirrors
    assert messages[0].acked is True


@pytest.mark.asyncio
async def test_task3_consume_plain_duplicate_noops_without_touching_siblings(
    async_engine, document_factory, monkeypatch
):
    """No ownership-loss signal: duplicate no-ops, siblings untouched."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="D" * 64)
    handler, payload, work_probe, _ = await _consume_setup_stage(
        monkeypatch, maker, doc_id, ws, rev_id, "parse"
    )
    await _consume_preset_running(maker, rev_id, "parse")
    await _consume_preset_running(maker, rev_id, "embed")

    messages, _channel = await _drive_consume(
        monkeypatch,
        maker,
        queue_name="hrag.parse",
        exchange_name="hrag.parse",
        routing_key="parse",
        payloads=[payload],
        handler=handler,
        headers_list=[{"x-retry-count": 0}],
        redelivered_list=[False],
    )

    assert work_probe() == (0, 0)
    assert await _handler_stage_rows(maker, rev_id) == {
        "parse": ("running", 1),
        "embed": ("running", 1),
        "caption": ("pending", 0),
        "kg": ("pending", 0),
    }
    assert messages[0].acked is True


@pytest.mark.asyncio
async def test_task3_recover_abandoned_claim_raises_on_infra_failure(
    async_engine, document_factory, monkeypatch
):
    """Recovery storage failure raises (never a silent no-op)."""
    import app.core.database as _dbmod
    from app.queue.connection import (
        AbandonedStageRecoveryError,
        _recover_abandoned_claim,
    )

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    _ws, rev_id = await _handler_setup_full(maker, doc_id, sha="E" * 64)
    await _consume_preset_running(maker, rev_id, "parse")

    async def _db_down(self, *args, **kwargs):
        raise RuntimeError("db blip")

    monkeypatch.setattr(
        _revisions_mod.DocumentRevisionsRepository,
        "mark_stage_retry_pending",
        _db_down,
    )
    monkeypatch.setattr(_dbmod, "async_session_maker", maker)
    message = _FakeConsumeMessage(
        body=b"{}", headers={"x-retry-count": 0}, redelivered=True
    )

    with pytest.raises(AbandonedStageRecoveryError):
        await _recover_abandoned_claim(
            message=message,
            queue_name="hrag.parse",
            exchange_name="hrag.parse",
            revision_id=rev_id,
            retry_count=0,
        )

    assert await _handler_stage_rows(maker, rev_id) == {
        "parse": ("running", 1),
        "embed": ("pending", 0),
        "caption": ("pending", 0),
        "kg": ("pending", 0),
    }


@pytest.mark.asyncio
async def test_task3_consume_recovery_infra_failure_rejects_without_ack(
    async_engine, document_factory, monkeypatch
):
    """End to end: recovery failure rejects, never acks a no-op (I-R1)."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="E2" * 32)
    handler, payload, work_probe, _ = await _consume_setup_stage(
        monkeypatch, maker, doc_id, ws, rev_id, "parse"
    )
    await _consume_preset_running(maker, rev_id, "parse")

    async def _db_down(self, *args, **kwargs):
        raise RuntimeError("db blip")

    monkeypatch.setattr(
        _revisions_mod.DocumentRevisionsRepository,
        "mark_stage_retry_pending",
        _db_down,
    )

    # The raise propagates past the handler try/excepts to the broker
    # context (which rejects instead of acking) and is then absorbed by
    # the consumer restart loop — so the drive returns, but the message
    # was rejected, never acked, and the handler never ran.
    messages, _channel = await _drive_consume(
        monkeypatch,
        maker,
        queue_name="hrag.parse",
        exchange_name="hrag.parse",
        routing_key="parse",
        payloads=[payload],
        handler=handler,
        headers_list=[{"x-retry-count": 0}],
        redelivered_list=[True],
    )

    assert work_probe() == (0, 0)
    assert messages[0].acked is False
    assert messages[0].exc_rejected is True
    assert await _handler_stage_rows(maker, rev_id) == {
        "parse": ("running", 1),
        "embed": ("pending", 0),
        "caption": ("pending", 0),
        "kg": ("pending", 0),
    }


@pytest.mark.asyncio
async def test_task3_running_claim_durably_committed_before_work(
    async_engine, document_factory, monkeypatch
):
    """A second session observes running/1 while the handler works (M-R1)."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="F" * 64)
    store = _RecordingParseStore()
    _patch_parse(monkeypatch, maker, store)

    # Observe from a SECOND session at the first point after the claim —
    # wrapping mark_revision_building, which runs before the next commit.
    # (Spying the later MinIO download would be too late: a subsequent
    # commit persists the claim anyway.)
    observed = {}
    _original_building = _parse_worker.mark_revision_building

    async def _spying_building(db, revision_id):
        async with maker() as db2:
            rows = await DocumentRevisionsRepository(db2).get_stages(rev_id)
            for row in rows:
                if row.stage == "parse":
                    observed["parse"] = (row.state, row.attempt_count)
        return await _original_building(db, revision_id)

    monkeypatch.setattr(
        _parse_worker, "mark_revision_building", _spying_building
    )

    await _parse_worker.handle_parse(_parse_payload(doc_id, ws, rev_id))

    # Without the commit right after the claim, the second session would
    # still see pending/0 here (the claim would ride the later commit).
    assert observed["parse"] == ("running", 1)
    assert await _handler_stage_rows(maker, rev_id) == {
        "parse": ("completed", 1),
        "embed": ("pending", 0),
        "caption": ("pending", 0),
        "kg": ("pending", 0),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queue,exchange,stage",
    [
        ("hrag.parse", "hrag.parse", "parse"),
        ("hrag.embed", "hrag.embed", "embed"),
        ("hrag.caption", "hrag.caption", "caption"),
        ("hrag.kg.00000000-0000-0000-0000-000000000001", "hrag.kg", "kg"),
    ],
)
async def test_task3_consume_error_branch_notes_exact_stage(
    async_engine, document_factory, monkeypatch, queue, exchange, stage
):
    """The real error branch reports running→pending for its own stage."""

    async def _boom(payload):
        raise RuntimeError("handler boom")

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    _ws, rev_id = await _handler_setup_full(maker, doc_id, sha="G" * 64)
    for s in ("parse", "embed", "caption", "kg"):
        await _consume_preset_running(maker, rev_id, s)
    payload = {"document_id": str(doc_id), "revision_id": str(rev_id)}

    messages, channel = await _drive_consume(
        monkeypatch,
        maker,
        queue_name=queue,
        exchange_name=exchange,
        routing_key="parse" if stage != "kg" else "00000000-0000-0000-0000-000000000001",
        payloads=[payload],
        handler=_boom,
        headers_list=[{"x-retry-count": 0}],
        redelivered_list=[False],
    )

    states = await _handler_stage_rows(maker, rev_id)
    for s in ("parse", "embed", "caption", "kg"):
        if s == stage:
            assert states[s] == ("pending", 1), s
        else:
            assert states[s] == ("running", 1), s
    assert len(channel.default_exchange.published) == 1
    assert messages[0].acked is True
    assert await _handler_revision_status(maker, rev_id) == "draft"


@pytest.mark.asyncio
async def test_task3_consume_exhausted_branch_fails_stage_and_revision(
    async_engine, document_factory, monkeypatch
):
    """The real exhausted branch fails the exact stage + revision (M-R2)."""

    async def _boom(payload):
        raise RuntimeError("handler boom")

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    _ws, rev_id = await _handler_setup_full(maker, doc_id, sha="H" * 64)
    await _consume_preset_running(maker, rev_id, "embed")
    payload = {"document_id": str(doc_id), "revision_id": str(rev_id)}

    messages, channel = await _drive_consume(
        monkeypatch,
        maker,
        queue_name="hrag.embed",
        exchange_name="hrag.embed",
        routing_key="embed",
        payloads=[payload],
        handler=_boom,
        headers_list=[{"x-retry-count": 3}],
        redelivered_list=[False],
    )

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"][0] == "failed"
    assert rows["parse"] == ("pending", 0)
    assert await _handler_revision_status(maker, rev_id) == "failed"
    assert messages[0].acked is False
    assert messages[0].rejected_requeue is False
    assert len(channel.default_exchange.published) == 0


@pytest.mark.asyncio
async def test_task3_exhausted_note_rolls_back_stage_on_revision_failure(
    async_engine, document_factory, monkeypatch
):
    """Stage-failed rolls back with a failed revision terminalization."""

    async def _revision_down(self, *args, **kwargs):
        raise RuntimeError("revision store down")

    monkeypatch.setattr(
        _revisions_mod.DocumentRevisionsRepository, "mark_failed", _revision_down
    )
    import app.core.database as _dbmod
    from app.queue.connection import note_stage_exhausted

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    _ws, rev_id = await _handler_setup_full(maker, doc_id, sha="I" * 64)
    await _consume_preset_running(maker, rev_id, "embed")

    monkeypatch.setattr(_dbmod, "async_session_maker", maker)
    assert (
        await note_stage_exhausted(rev_id, "embed", failure_class="ValueError")
        is False
    )

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"] == ("running", 1)
    assert await _handler_revision_status(maker, rev_id) == "draft"
