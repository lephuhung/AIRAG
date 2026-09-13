"""Side-effect-free v2 shadow execution (Phase 3, Task 6).

A shadow v2 run replays a primary turn through the SAME supervisor
topology — ``create_supervisor_v2_graph(checkpointer=isolated_saver)`` —
while proving it can never write production state or emit outbound events.

Global constraints honored here:

- Shadow compiles its OWN graph with an isolated saver bundle
  (``v2.persistence.shadow_checkpoint.create_shadow_checkpointer``); it
  NEVER calls the production ``get_supervisor_v2_graph()`` and NEVER
  constructs the production saver (``v2.persistence.checkpoint`` is not
  imported on any path in this module).
- Read-only source adapters only: every shadow source is wrapped in
  :class:`ReadOnlySourceAdapter`, whose write surface raises
  :class:`ShadowIsolationError` (an ``AttributeError``, so ``hasattr``
  reports no write path at all).
- Cancellation follows the primary run (``asyncio`` cancellation
  propagates through :meth:`ShadowBundle.run`; the run holds no commit
  path, so a cancelled shadow persists nothing anywhere).
- Output is discarded except redacted metrics (:class:`ShadowMetrics` —
  status/route/counts/timing only, never response content).
- Exactly one ownership chain: the shadow graph is the same
  ``create_supervisor_v2_graph`` topology — no alternate graph/agent.
- Frozen contract types are imported, never redefined.
- Only the shared ``TaskScheduler`` may dispatch a capability; the shadow
  bundle wires no capability registry (``None``), so any factual route
  fails closed instead of dispatching. Shadow traffic is therefore
  direct-route-shaped by construction.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.services.agents.supervisor_v2 import create_supervisor_v2_graph
from app.services.agents.v2.capabilities import CapabilityRuntimeContext
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.semantic import SemanticContext, SemanticDraft
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.persistence.shadow_checkpoint import (
    ShadowCheckpointBundle,
    create_shadow_checkpointer,
    is_shadow_saver,
)

__all__ = [
    "ShadowIsolationError",
    "ShadowMetrics",
    "ReadOnlySourceAdapter",
    "IsolatedShadowStores",
    "ShadowBundle",
    "build_shadow_bundle",
    "should_run_shadow",
    "shadow_sampling_enabled",
    "emit_outbound_event",
]


logger = logging.getLogger(__name__)


class ShadowIsolationError(AttributeError):
    """A shadow run reached for a forbidden write or outbound surface.

    Subclasses ``AttributeError`` on purpose: write attributes raise this
    instead of existing, so ``hasattr(adapter, "write")`` is ``False`` —
    no write path is even reachable from the shadow bundle, let alone
    callable.
    """


def emit_outbound_event(*args: Any, **kwargs: Any) -> None:
    """The single outbound-event funnel (SSE/webhook/Telegram/chat).

    Production wiring routes every outbound emission through here. The
    shadow path NEVER calls it — any call raises, so a stray emission
    fails the run loudly instead of leaking an event. Tests spy on this
    funnel (plus the real SSE formatter) to prove R55 silence.
    """
    raise ShadowIsolationError(
        "shadow runs must never emit outbound events "
        f"(got args={args!r} kwargs={kwargs!r})"
    )


#: Attribute names that count as a write path on a source adapter. Any
#: access to one of these on a :class:`ReadOnlySourceAdapter` raises
#: :class:`ShadowIsolationError`.
_WRITE_SURFACE = frozenset(
    {
        "write",
        "save",
        "commit",
        "persist",
        "delete",
        "update",
        "append",
        "insert",
        "upsert",
        "remove",
        "put",
        "post",
        "send",
        "emit",
        "publish",
        "notify",
    }
)


class ReadOnlySourceAdapter:
    """Read-only wrapper around one shadow source.

    Only the wrapped ``read``/``lookup``/``fetch`` behavior is exposed;
    every write-surface attribute raises :class:`ShadowIsolationError`.
    The wrapper keeps no reference to any production session, connection,
    or store — it is constructed over the isolated shadow stores only.
    """

    def __init__(self, name: str, reader: Any = None) -> None:
        self._name = name
        self._reader = reader

    @property
    def adapter_name(self) -> str:
        return self._name

    @property
    def read_only(self) -> bool:
        return True

    async def read(self, *args: Any, **kwargs: Any) -> Any:
        """The only data path: delegate a read, or return no rows."""
        if self._reader is None:
            return None
        read = getattr(self._reader, "read", None)
        if callable(read):
            result = read(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return None

    async def lookup(self, *args: Any, **kwargs: Any) -> Any:
        """Read-only lookup alias (same delegation as :meth:`read`)."""
        return await self.read(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in _WRITE_SURFACE:
            raise ShadowIsolationError(
                f"shadow source adapter {self.__dict__.get('_name', '?')!r} "
                f"is read-only; write path {name!r} is not reachable"
            )
        raise AttributeError(
            f"shadow source adapter has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _WRITE_SURFACE:
            raise ShadowIsolationError(
                f"shadow source adapter is read-only; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)


@dataclass
class IsolatedShadowStores:
    """Per-run isolated stores: fresh dicts, never production handles.

    Keys mirror the production tables the isolation proof watches
    (checkpoint/evidence/use/audit/chat/memory/title) so the test can
    assert the production mappings are untouched while the shadow run
    writes only here.
    """

    checkpoint: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_use: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    chat: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    title: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowMetrics:
    """Redacted shadow-run metrics — the ONLY shadow output.

    Carries status/route/counts/timing. Response content, citations,
    evidence payloads, and user text are deliberately absent: there is no
    field that could carry them.
    """

    status: str
    route: str | None
    task_count: int
    duration_ms: int

    def redacted(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": self.route,
            "task_count": self.task_count,
            "duration_ms": self.duration_ms,
        }


class _ShadowSemanticAdapter:
    """Deterministic read-only semantic adapter over the raw query."""

    def __init__(self, query: str) -> None:
        self._query = query

    async def build_draft(
        self, request: RequestContext, conversation: ConversationContext
    ) -> SemanticDraft:
        return SemanticDraft(
            provisional_contextualized_query=self._query.strip().lower(),
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            preliminary_ambiguities=(),
        )


class _ShadowBindingResolver:
    """Read-only binding resolver: pins nothing, returns the empty set."""

    async def resolve(
        self, document_refs: Any, capability_runtime: CapabilityRuntimeContext
    ) -> DocumentBindingSet:
        return DocumentBindingSet(bindings=(), revision_requirement_refs=())


@dataclass
class ShadowBundle:
    """One shadow run: isolated saver + stores + read-only adapters + graph.

    ``production_rows`` is an optional READ handle to the production-row
    mapping under test; the bundle never writes through it (the isolation
    test asserts identity of the mapping afterwards).
    """

    raw_query: str
    thread_id: str
    checkpoint_bundle: ShadowCheckpointBundle
    stores: IsolatedShadowStores
    source_adapters: tuple[ReadOnlySourceAdapter, ...]
    graph: Any = field(repr=False)
    runtime_context: Any = field(repr=False)
    initial_state: SupervisorV2State = field(repr=False)
    production_rows: Any = field(default=None, repr=False)
    metrics: ShadowMetrics | None = field(default=None)

    async def run(self) -> ShadowMetrics:
        """Run one shadow turn; return redacted metrics, discard the rest.

        Cancellation (primary-run follow) propagates as
        ``asyncio.CancelledError``: the run holds no commit path, so a
        cancelled shadow persists nothing — not even to its isolated
        stores. No outbound emitter is invoked on any path.
        """
        started = time.monotonic()
        config = {"configurable": {"thread_id": self.thread_id}}
        result = await self.graph.ainvoke(
            dict(self.initial_state), config, context=self.runtime_context
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        # Output is discarded except redacted metrics: read the terminal
        # status/route/counts WITHOUT keeping content, citations, or uses.
        final = result.get("final_response")
        if isinstance(final, dict):
            status = final.get("status", "error")
        else:
            status = getattr(final, "status", "error")
        decision = result.get("route_decision")
        if isinstance(decision, dict):
            route = decision.get("route")
        else:
            route = getattr(decision, "route", None)
        execution = result.get("execution")
        if isinstance(execution, dict):
            plan = execution.get("plan")
        else:
            plan = getattr(execution, "plan", None)
        if plan is None:
            task_count = 0
        elif isinstance(plan, dict):
            task_count = len(plan.get("tasks", ()))
        else:
            task_count = len(getattr(plan, "tasks", ()))
        self.metrics = ShadowMetrics(
            status=str(status),
            route=route,
            task_count=task_count,
            duration_ms=duration_ms,
        )
        # Record only counts in the isolated stores (never content).
        self.stores.checkpoint[self.thread_id] = {
            "status": self.metrics.status,
            "task_count": task_count,
        }
        logger.info(
            "shadow v2 turn complete: status=%s route=%s tasks=%d duration_ms=%d",
            self.metrics.status,
            self.metrics.route,
            task_count,
            duration_ms,
        )
        return self.metrics

    async def run_forever(self) -> ShadowMetrics:
        """Park the shadow run until the primary run cancels it.

        Models "cancellation follows the primary run": the parked shadow
        holds no resources and persists nothing; primary cancellation
        arrives as ``asyncio.CancelledError``.
        """
        while True:
            await asyncio.sleep(3600)


def build_shadow_bundle(
    *,
    raw_query: str,
    thread_id: str,
    user_id: UUID | None = None,
    workspace_ids: tuple[UUID, ...] = (),
    production_rows: Any = None,
) -> ShadowBundle:
    """Assemble one isolated shadow bundle (R54 ownership chain).

    Fresh isolated saver + isolated conversation/evidence/use/audit
    stores + read-only source adapters
    -> ``create_supervisor_v2_graph(checkpointer=isolated_saver)``.

    The SAME ``create_supervisor_v2_graph`` topology as production — no
    alternate graph/agent. The production graph resolver, the production
    saver factory, and the production graph resolver in
    ``agent.runtime_selector`` are not referenced on this path.
    """
    saver = create_shadow_checkpointer()
    assert is_shadow_saver(saver)
    namespace = f"shadow-{uuid4().hex[:12]}"
    stores = IsolatedShadowStores()
    adapters = (
        ReadOnlySourceAdapter("conversation"),
        ReadOnlySourceAdapter("evidence"),
        ReadOnlySourceAdapter("documents"),
        ReadOnlySourceAdapter("memory"),
    )
    graph = create_supervisor_v2_graph(checkpointer=saver)
    capability_runtime = CapabilityRuntimeContext(
        request_id=f"shadow-req-{uuid4().hex[:12]}",
        run_id=f"shadow-run-{uuid4().hex[:12]}",
        user_id=user_id or UUID("00000000-0000-0000-0000-000000000000"),
        workspace_ids=tuple(workspace_ids),
        can_read_people=False,
        allowed_capabilities=frozenset(),
        deadline_at=datetime.now(UTC),
    )
    services = RuntimeServices(
        retention_leases=None,
        semantic_adapter=_ShadowSemanticAdapter(raw_query),
        binding_resolver=_ShadowBindingResolver(),
        capability_registry=None,
        chat_messages=None,
        authorization=None,
        evidence_hydrator=None,
        answer_draft_channel=None,
    )
    runtime_context = GraphRuntimeContext(
        capability_runtime=capability_runtime,
        services=services,
    )
    initial_state = SupervisorV2State(
        contract_version=CONTRACT_VERSION,
        request=RequestContext(
            contract_version=CONTRACT_VERSION,
            request_id=capability_runtime.request_id,
            thread_id=thread_id,
            original_query=raw_query,
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="", active_entities=(), last_focus=None, recent_turns=()
        ),
        semantic=SemanticContext(
            contextualized_query="",
            normalized_query="",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        ),
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=None,
        route_decision=None,
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )
    return ShadowBundle(
        raw_query=raw_query,
        thread_id=f"{namespace}:{thread_id}",
        checkpoint_bundle=ShadowCheckpointBundle(
            saver=saver, namespace=namespace, _production_rows_ref=production_rows
        ),
        stores=stores,
        source_adapters=adapters,
        graph=graph,
        runtime_context=runtime_context,
        initial_state=initial_state,
        production_rows=production_rows,
    )


def should_run_shadow(*, percent: float, sample: float | None = None) -> bool:
    """Sample one turn for shadowing: ``sample`` in [0, 100) vs ``percent``.

    ``percent=0`` (the safe default) never shadows; ``percent=100``
    always shadows. A caller-supplied ``sample`` keeps tests
    deterministic; live callers draw ``random.uniform(0, 100)``.
    """
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    value = random.uniform(0, 100) if sample is None else sample
    return value < percent


def shadow_sampling_enabled() -> bool:
    """True only when shadow wiring is enabled with a nonzero percent."""
    from app.core.config import settings

    return bool(
        getattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_ENABLED", False)
    ) and float(getattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_PERCENT", 0)) > 0
