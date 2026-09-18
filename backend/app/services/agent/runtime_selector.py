"""Lazy v1/v2 agent runtime selector + per-request v2 ingress (Phase 2, Task 7).

Selection contract (controller rulings):

- Default is v1; v1 behavior never changes through this module.
- ``NEXUSRAG_AGENT_GRAPH_VERSION`` selects ``v1`` | ``v2``; any other value
  fails fast (see ``normalize_agent_version`` + the ``Settings`` validator).
- Lazy factories ONLY: this module imports no graph at module import time.
  ``resolve_agent_graph`` imports ``get_supervisor_graph`` /
  ``get_supervisor_v2_graph`` inside the call, and the v2 arm runs
  ``require_v2_schema_ready`` (Phase-1 ``persistence.migrate.check_v2_schema``
  + ``persistence.checkpoint.check_v2_checkpointer``) BEFORE the v2 graph is
  built. A failed gate raises ``V2NotReadyError`` — fail closed, never a
  silent v1 fallback — and v2 stays unselectable until the lifespan has
  installed the compiled graph.
- Ordinary graph-version headers are ignored: version selection reads only
  the configured value plus the authenticated admin override. There is no
  header parameter anywhere in this module.
- The admin evaluation override is the ONLY per-request arm override and is
  authenticated-admin-only (``resolve_request_version``).

Ingress contract (T6 hand-off, T7-owned):

- Persist the raw user text BEFORE any normalization/semantic work
  (``persist_raw_user_message``; entrypoints call it before abbreviation
  expansion or preprocessing).
- Current runtime scope is authenticated scope ∩ requested scope, computed
  by ``resolve_runtime_scope`` — never widened by model/request input.
- Per-request construction lives in ``build_v2_ingress``: the real
  request-scoped ``RuntimeServices``/``CapabilityRegistry`` with concrete
  v1-backed capability adapters (T2-I4), the ``binding_resolver`` wrapping
  ``adapters/document.py::resolve_document_bindings`` into
  ``-> DocumentBindingSet`` (D5), deterministic/idempotent adapters (D3),
  a lease repository over a DEDICATED unit of work/session (M4a — never the
  shared app session), the runtime-only truncation channel (T6-I1), and the
  ``ClarificationUnsatisfiable`` runner contract.
- ``RuntimeServices`` has exactly ONE definition
  (``v2/contracts/state.py``); this module only constructs it via
  ``build_runtime_services``.
- Never run ``validate_supervisor_state`` on a non-success terminal whose
  semantic is still the ingress placeholder — see
  ``terminal_state_is_error``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

logger = logging.getLogger(__name__)

AgentGraphVersion = Literal["v1", "v2"]

VALID_AGENT_GRAPH_VERSIONS: tuple[str, ...] = ("v1", "v2")
DEFAULT_AGENT_GRAPH_VERSION: str = "v1"

#: The capability set a v2 ingress grants when the caller supplies none
#: (``build_v2_ingress(allowed_capabilities=None)``). Single owner of the
#: default so the primary ingress and the shadow hook (R64) derive the same
#: granted set instead of duplicating the literal.
DEFAULT_V2_ALLOWED_CAPABILITIES: frozenset[str] = frozenset(
    {
        "people.lookup",
        "document.search",
        # P0 Task 5: permitted by default; actual dispatch additionally
        # requires the ``v1-revision-retrieval`` service gate (manifest +
        # provider dependencies live) in the request-scoped registry.
        "document.retrieve",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
        "abbreviation.resolve",
    }
)


def normalize_agent_version(value: Any) -> str:
    """Validate a graph-version selection; fail fast on anything but v1|v2."""
    if not isinstance(value, str):
        raise ValueError(
            f"agent graph version must be one of {VALID_AGENT_GRAPH_VERSIONS}; "
            f"got {value!r}"
        )
    normalized = value.strip().lower()
    if normalized not in VALID_AGENT_GRAPH_VERSIONS:
        raise ValueError(
            f"agent graph version must be one of {VALID_AGENT_GRAPH_VERSIONS}; "
            f"got {value!r}"
        )
    return normalized


class V2NotReadyError(RuntimeError):
    """v2 was selected but is not ready — fail closed, never fall back to v1."""


def configured_agent_version() -> str:
    """Read the configured arm (``NEXUSRAG_AGENT_GRAPH_VERSION``), validated."""
    from app.core.config import settings

    return normalize_agent_version(settings.NEXUSRAG_AGENT_GRAPH_VERSION)


def resolve_request_version(
    *,
    user: Any = None,
    admin_override: str | None = None,
) -> str:
    """Resolve the arm for one request.

    The configured version wins. The ONLY per-request override is
    ``admin_override`` (the admin evaluation surface), which requires an
    authenticated superadmin principal; anything else — anonymous callers,
    non-admin users, invalid values — raises the matching HTTP error.
    Ordinary request headers are not consulted (no such parameter exists).
    """
    from app.core.exceptions import ForbiddenError, UnauthorizedError
    from fastapi import HTTPException, status

    base = configured_agent_version()
    if admin_override is None:
        return base
    if user is None:
        raise UnauthorizedError("Not authenticated")
    if not bool(getattr(user, "is_superadmin", False)):
        raise ForbiddenError("Superadmin access required")
    try:
        return normalize_agent_version(admin_override)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


async def resolve_serving_arm(
    *,
    db: Any,
    workspace_ids: Any,
    request_id: str,
    is_write_endpoint: bool = False,
) -> str:
    """Resolve the serving arm for chat ingress (Task 7B canary, R8).

    Thin entrypoint-facing wrapper over
    ``app.services.agent.rollout_control.resolve_serving_arm`` (the single
    owner of the canary rule). Server-owned inputs only: the authenticated
    workspace scope, the persisted request ID, and the ingress call site's
    deterministically-known Write flag. Ordinary headers are ignored (no
    such parameter exists); public traffic takes no per-request override.
    Import is function-local so this module stays graph-free at import time.
    """
    from app.services.agent.rollout_control import (
        resolve_serving_arm as _canary_serving_arm,
    )

    return await _canary_serving_arm(
        db=db,
        workspace_ids=workspace_ids,
        request_id=request_id,
        is_write_endpoint=is_write_endpoint,
    )


def resolve_runtime_scope(
    *,
    authenticated_ids: Any,
    requested_ids: Any | None = None,
) -> tuple[UUID, ...]:
    """Intersect the authenticated scope with the requested scope.

    The result is always a subset of ``authenticated_ids`` (order-preserving)
    — request/model input can narrow the runtime scope but never widen it.
    ``requested_ids=None`` means "no narrowing requested" and keeps the full
    authenticated scope.
    """
    authenticated = tuple(authenticated_ids or ())
    if requested_ids is None:
        return authenticated
    wanted = {item for item in (requested_ids or ())}
    return tuple(item for item in authenticated if item in wanted)


# ---------------------------------------------------------------------------
# Readiness gate + lazy resolver
# ---------------------------------------------------------------------------


async def require_v2_schema_ready(
    *,
    schema_check: Any | None = None,
    checkpoint_check: Any | None = None,
) -> None:
    """Gate v2 selection on Phase-1 schema + checkpointer readiness.

    Runs the real Phase-1 checks unless precomputed outcomes are supplied
    (the test seam). Anything but a fully clean schema and a fully present
    checkpointer raises ``V2NotReadyError`` — selection fails closed.
    """
    # Connection/IO failures of the live probes fail closed with the typed
    # gate error (never a raw driver traceback to the caller).
    try:
        if schema_check is None:
            schema_check = await asyncio.to_thread(_real_schema_check)
        if checkpoint_check is None:
            checkpoint_check = await _real_checkpoint_check()
    except V2NotReadyError:
        raise
    except Exception as exc:
        raise V2NotReadyError(
            f"supervisor v2 readiness probe failed: {exc}; keeping v1"
        ) from exc
    problems: list[str] = []
    # Canonical verdicts live on the Phase-1 result types
    # (SchemaCheck.is_clean, CheckpointCheck.is_ready); the field-level
    # detail below only shapes the error message and must not drift from them.
    schema_clean = getattr(schema_check, "is_clean", None)
    if schema_clean is None:
        schema_clean = bool(
            getattr(schema_check, "applied", False)
        ) and not (
            getattr(schema_check, "missing_tables", frozenset())
            or getattr(schema_check, "extra_tables", frozenset())
            or getattr(schema_check, "shape_errors", frozenset())
        )
    if not schema_clean:
        if not bool(getattr(schema_check, "applied", False)):
            problems.append("v2 schema is not applied")
        else:
            missing = (
                getattr(schema_check, "missing_tables", frozenset())
                or frozenset()
            )
            extra = (
                getattr(schema_check, "extra_tables", frozenset())
                or frozenset()
            )
            shape = (
                getattr(schema_check, "shape_errors", frozenset())
                or frozenset()
            )
            if missing:
                problems.append(f"v2 schema missing tables: {sorted(missing)}")
            if extra:
                problems.append(f"v2 schema extra tables: {sorted(extra)}")
            if shape:
                problems.append(f"v2 schema shape errors: {sorted(shape)}")
            if not (missing or extra or shape):
                problems.append("v2 schema check is not clean")
    checkpoint_ready = getattr(checkpoint_check, "is_ready", None)
    if checkpoint_ready is None:
        checkpoint_ready = not (
            getattr(checkpoint_check, "missing", frozenset()) or frozenset()
        )
    if not checkpoint_ready:
        missing_cp = getattr(checkpoint_check, "missing", frozenset()) or frozenset()
        if missing_cp:
            problems.append(
                f"v2 checkpointer missing tables: {sorted(missing_cp)}"
            )
        else:
            problems.append("v2 checkpointer is not ready")
    if problems:
        raise V2NotReadyError(
            "supervisor v2 is not ready (" + "; ".join(problems) + "); "
            "keeping the v1 default"
        )


def _real_schema_check() -> Any:
    """Run Phase-1 ``check_v2_schema`` against the application database."""
    from app.core.config import settings
    from app.services.agents.v2.persistence.migrate import (
        check_v2_schema,
        make_engine,
    )

    engine = make_engine(settings.DATABASE_URL)
    try:
        return check_v2_schema(engine)
    finally:
        engine.dispose()


async def _real_checkpoint_check() -> Any:
    """Run Phase-1 ``check_v2_checkpointer`` (read-only) against the store."""
    from app.core.config import settings
    from app.services.agents.v2.persistence.checkpoint import check_v2_checkpointer

    return await check_v2_checkpointer(settings.CHECKPOINT_DATABASE_URL)


async def resolve_agent_graph(version: AgentGraphVersion) -> Any:
    """Return the compiled supervisor graph for ``version`` (lazy factories).

    No graph is constructed at import time: the v1/v2 getters are imported
    inside this call. The v2 arm runs ``require_v2_schema_ready`` BEFORE the
    v2 graph is touched, and the lifespan-owned singleton getter fails fast
    when v2 was never installed — both fail closed to ``V2NotReadyError`` /
    ``SupervisorV2Error``, never to a silent v1 fallback.
    """
    normalized = normalize_agent_version(version)
    if normalized == "v1":
        from app.services.agents.supervisor import get_supervisor_graph

        return get_supervisor_graph()
    await require_v2_schema_ready()
    # Both names import up front (still function-local, still lazy): if the
    # import itself fails we raise the typed gate error instead of letting
    # an ImportError mask the original failure inside the handler below.
    try:
        from app.services.agents.supervisor_v2 import (
            SupervisorV2Error,
            get_supervisor_v2_graph,
        )
    except ImportError as exc:
        raise V2NotReadyError(
            f"supervisor v2 module is not importable: {exc}"
        ) from exc
    try:
        return get_supervisor_v2_graph()
    except SupervisorV2Error as exc:
        raise V2NotReadyError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Ingress primitive: raw text persists before normalization
# ---------------------------------------------------------------------------


async def persist_raw_user_message(
    db: Any,
    *,
    session_id: Any,
    user_id: Any,
    raw_text: str,
    message_id: str | None = None,
    document_ids: Any | None = None,
) -> Any:
    """Persist the RAW user text before any normalization/semantic work.

    Entrypoints MUST call this with the verbatim request text before
    abbreviation expansion, preprocessing, or graph invocation, so the chat
    history owns exactly what the user sent — never a rewritten form.
    """
    from app.models.chat_message import ChatMessage

    row = ChatMessage(
        session_id=session_id,
        message_id=message_id or f"msg_{uuid.uuid4().hex[:8]}",
        role="user",
        content=raw_text,
        user_id=user_id,
        document_ids=document_ids,
    )
    db.add(row)
    await db.commit()
    return row


# ---------------------------------------------------------------------------
# Per-request v2 services (T6 hand-off, T7-owned construction)
# ---------------------------------------------------------------------------


class ChatMessagesService:
    """Request-scoped ``chat_messages`` service over the ChatMessage table.

    Loads the raw clarification reply by ``ChatMessage.id``; the persisted
    row stays authoritative (the resolution parser keeps no user text).
    """

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def get_user_message(self, message_id: UUID) -> Any:
        from sqlalchemy import select

        from app.models.chat_message import ChatMessage

        async with self._session_factory() as db:
            result = await db.execute(
                select(ChatMessage).where(ChatMessage.id == message_id)
            )
            row = result.scalar_one_or_none()
        if row is None:
            raise LookupError(f"chat message {message_id!r} not found")
        return row


class V2AuthorizationService:
    """Request-scoped ``authorization`` service over the current runtime.

    ``require_document`` returns ``None`` on success and raises
    ``PermissionError`` when the document is outside the current authorized
    scope — the clarify resume path translates that into the typed
    ``ClarificationUnauthorized`` denial. Only the CURRENT runtime scope
    decides; checkpointed state is never consulted.
    """

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def require_document(
        self, document_id: UUID, capability_runtime: Any
    ) -> None:
        from sqlalchemy import select

        from app.models.document import Document

        if isinstance(document_id, str):
            document_id = UUID(document_id)
        scope = {item for item in (capability_runtime.workspace_ids or ())}
        async with self._session_factory() as db:
            result = await db.execute(
                select(Document.workspace_id).where(Document.id == document_id)
            )
            workspace_id = result.scalar_one_or_none()
        if workspace_id is None or workspace_id not in scope:
            raise PermissionError(
                f"document {document_id!r} is not in the current runtime scope"
            )


class GovernorEvidenceBuilder:
    """Production ``EvidenceBuilder`` over the governed Evidence Store.

    Non-people evidence goes through the public ``persist_record`` pipeline
    (minimize → classify → hash → encrypt → idempotent persist). People
    evidence arrives already minimized by the capability; it is re-persisted
    through the public ``persist_people_evidence`` entry point with the
    minimized field set as the required fields — re-minimization is
    deterministic (canonical JSON), so identical content converges to the
    same bytes and hash instead of bypassing the §15.3 proof. The minted use
    passes structural ``validate_evidence_use``; plan/target resolution is
    validated downstream at hydration (the builder has no plan by design).
    Flush-only: the caller commits (see ``V2Ingress.commit_evidence``).
    """

    def __init__(self, governor: Any, *, run_id: str) -> None:
        self._governor = governor
        self._run_id = run_id

    async def persist_use(
        self,
        *,
        source: Any,
        content: str,
        provenance: Any,
        task_id: str,
        purpose: Any,
        target_id: str | None,
    ) -> Any:
        from app.services.agents.v2.contracts.evidence import (
            DocumentSourceIdentity,
            EvidenceUse,
            EvidenceUseEnvelope,
            EvidenceUseRef,
            PeopleSourceIdentity,
        )
        from app.services.agents.v2.contracts.validation import validate_evidence_use

        if not isinstance(content, str) or not content.strip():
            raise ValueError("evidence content must be non-blank")
        if isinstance(source, PeopleSourceIdentity):
            try:
                raw = json.loads(content)
            except ValueError as exc:
                raise ValueError(
                    "people evidence content must be minimized canonical JSON"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    "people evidence content must be a minimized JSON object"
                )
            evidence_id = await self._governor.persist_people_evidence(
                record_id=source.record_id,
                raw_record=raw,
                required_fields=list(raw.keys()),
                provenance=provenance,
            )
        else:
            # Document evidence must reference the authoritative revision it
            # was read from: ``document_revision`` is the string form of the
            # revision UUID (see ``adapters/document.py``), parsed here and
            # passed as ``revision_id`` so workspace ACL resolves through the
            # immutable revision row. Anything else fails closed before
            # persistence — never a fallback to a current pointer/request.
            persist_kwargs: dict[str, Any] = {}
            if isinstance(source, DocumentSourceIdentity):
                try:
                    persist_kwargs["revision_id"] = UUID(
                        source.document_revision
                    )
                except (ValueError, AttributeError, TypeError) as exc:
                    from app.services.agents.v2.evidence_store.governance import (
                        EvidenceValidationError,
                    )

                    raise EvidenceValidationError(
                        "document evidence revision "
                        f"{source.document_revision!r} is not the authoritative "
                        "revision_id it was read from"
                    ) from exc
            evidence_id = await self._governor.persist_record(
                source=source,
                content=content,
                provenance=provenance,
                **persist_kwargs,
            )
        use = EvidenceUse(
            use_id=uuid.uuid4(),
            evidence_id=evidence_id,
            task_id=task_id,
            purpose=purpose,
            target_id=target_id,
        )
        validate_evidence_use(use)
        stored_use_id = await self._governor.repository.append_use(
            EvidenceUseEnvelope(
                contract_version="2.0",
                run_id=self._run_id,
                use=use,
            )
        )
        return EvidenceUseRef(use_id=stored_use_id)


class PlanBindingResolver:
    """The 15-line pure ``PinnedTargetResolver`` (T6 handoff, T7 ships it).

    Fed the authoritative checkpointed plan/bindings — ``feed`` is owned by
    the shared ``TaskScheduler`` at dispatch time (authoritative; installs
    the checkpointed plan + bindings immediately before every dispatch, so
    fresh non-resume turns resolve) with the outer runner (T8) resume
    pre-feed kept as an idempotent compatibility refresh. This is not
    scheduler input materialization: ``TaskSpec.input`` is never mutated,
    rewritten, or lazily built — the feed only installs request-scoped
    authority the scheduler was already handed. Until fed (e.g. a
    first-turn read before any plan checkpoint exists), ``resolve`` returns
    ``None`` and the read capabilities fail closed per their contract —
    never a guessed target. The resolver never reads supervisor/graph
    state and is never fed from ``AgentRequest`` or
    ``CapabilityRuntimeContext``.
    """

    def __init__(self) -> None:
        self._plan: Any | None = None
        self._documents: dict[str, Any] = {}

    def feed(self, plan: Any, bindings: Any) -> None:
        """Install the authoritative checkpointed plan + binding set.

        Atomic: the replacement mapping is built locally first, then
        ``_plan`` and ``_documents`` are assigned together. On any build
        failure no stale authority is kept (both slots are cleared) and
        the error propagates so the scheduler surfaces it instead of
        dispatching against a half-installed map.
        """
        from app.services.agents.v2.capabilities import ResolvedTarget

        try:
            documents: dict[str, Any] = {}
            for binding in bindings.bindings or ():
                unit = next(
                    (
                        item
                        for item in plan.target_units
                        if item.binding_id == binding.binding_id
                    ),
                    None,
                )
                if unit is None:
                    continue
                documents[unit.target_id] = ResolvedTarget(
                    target_unit=unit, document=binding
                )
        except Exception:
            self._plan = None
            self._documents = {}
            raise
        self._plan = plan
        self._documents = documents

    def resolve(self, target_id: str) -> Any | None:
        """Map a planned ``target_id`` to its pinned ``ResolvedTarget``."""
        return self._documents.get(target_id)


def clarification_reply_is_fresh_turn(error: BaseException) -> bool:
    """Runner contract for candidate-free clarification requests.

    A reply to a request that offered NO selectable candidate raises
    ``ClarificationUnsatisfiable`` — the runner MUST treat that reply as a
    fresh user turn instead of a bad reply (re-ask loops can never resolve
    it). Any other clarification error keeps the standard typed-denial path.
    """
    from app.services.agents.v2.nodes.clarification import ClarificationUnsatisfiable

    return isinstance(error, ClarificationUnsatisfiable)


def terminal_state_is_error(state: Any) -> bool:
    """True when the turn already ended non-success (typed error/denied).

    Early-conversion terminals keep the INGRESS placeholder semantic (blank
    ``contextualized_query``), which the frozen aggregate validator rejects
    by construction. A state for which this returns True is terminal-errored,
    not corrupt — the runner must surface it and must NEVER run
    ``validate_supervisor_state`` on it as a health check.
    """
    if isinstance(state, dict):
        final = state.get("final_response")
    else:
        final = getattr(state, "final_response", None)
    if final is None:
        return False
    status = final.get("status") if isinstance(final, dict) else getattr(
        final, "status", None
    )
    return status is not None and status != "success"


def undispatched_tasks(plan: Any, results: Any) -> tuple[str, ...]:
    """Task ids in the checkpointed plan with no result yet (T3-N2 recipe).

    Single source of truth is T6's ``supervisor_v2.undispatched_tasks``
    (imported function-locally to keep this module graph-free at import
    time); see its docstring for the deliberate-loss record. The outer
    runner calls this at the terminal boundary: a non-empty remainder with
    no active run means an incomplete dispatch — raise rollback/error from
    it, never present partial results as complete.
    """
    from app.services.agents.supervisor_v2 import (
        undispatched_tasks as _t6_undispatched_tasks,
    )

    return _t6_undispatched_tasks(plan, results)


def make_v1_preprocess_closure(
    *,
    user_id: UUID,
    workspace_ids: tuple[UUID, ...],
    document_ids: tuple[UUID, ...] = (),
    session_id: str | None = None,
    can_read_people: bool = False,
    session_factory: Callable[[], Any] | None = None,
) -> Callable[[str], Any]:
    """Build the D3 preprocess closure for ``DeterministicSemanticAdapter``.

    The SAME ``(request, conversation)`` MUST yield an equal draft on every
    call (the draft is rebuilt once per node that needs it), and building is
    read-only: the closure only runs the v1 preprocessing DAG over the raw
    query with the CURRENT runtime scope — it writes nothing and keeps no
    per-call state, so new ref ids per call cannot diverge the turn.
    """

    async def _preprocess(raw_query: str) -> Any:
        from app.services.agents.deep_research.contracts import (
            ConsumedBudget,
            ModelSnapshot,
            PreprocessorBudgetConfig,
            RuntimeContext,
            ToolBudget,
        )
        from app.services.agents.semantic_preprocessor import preprocess_query
        import time as time_module
        import uuid as uuid_module

        from app.core.config import settings

        deadline_offset = getattr(
            settings, "NEXUSRAG_PREPROCESSOR_DEADLINE_SEC", 25.0
        )
        config_revision = getattr(settings, "NEXUSRAG_CONFIG_REVISION", "unknown")
        ctx = RuntimeContext(
            principal_id=user_id,
            allowed_workspace_ids=list(workspace_ids),
            authorized_document_handles=set(document_ids),
            people_permission=bool(can_read_people),
            session_id=session_id,
            run_id=str(uuid_module.uuid4()),
            config_revision=config_revision,
            absolute_deadline_monotonic=time_module.monotonic() + deadline_offset,
            absolute_deadline_epoch=None,
            remaining_budget_sec=deadline_offset,
            model_snapshot=ModelSnapshot(
                provider="preprocessor",
                model="internal",
                config_revision=config_revision,
                langfuse_tags=[],
            ),
            tool_budget=ToolBudget(),
            consumed_budget=ConsumedBudget(),
            preprocessing=PreprocessorBudgetConfig(),
            tool_allowlist=set(),
        )
        if session_factory is None:
            from app.core.database import async_session_maker

            factory = async_session_maker
        else:
            factory = session_factory
        async with factory() as db:
            return await preprocess_query(raw_query, ctx, db)

    return _preprocess


def make_abbreviation_lookup(
    session_factory: Callable[[], Any],
) -> Callable[[str], str | None]:
    """Build the sync abbreviation lookup closure (T6 seam, T7 wires it).

    The v1 query shape mirrors ``tools.search_abbreviation``: an active
    ``short_form`` match. The closure runs its own short-lived session per
    call so abbreviation reads never join another unit of work.
    """
    import asyncio

    def _lookup(token: str) -> str | None:
        async def _run() -> str | None:
            from sqlalchemy import select

            from app.models.abbreviation import Abbreviation

            async with session_factory() as db:
                result = await db.execute(
                    select(Abbreviation).where(
                        Abbreviation.short_form == token,
                        Abbreviation.is_active.is_(True),
                    )
                )
                row = result.scalar_one_or_none()
                return row.full_form if row is not None else None

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        # Inside a running loop the sync protocol cannot block: fail closed
        # so the capability maps it to typed unavailable, never a guess.
        from app.services.agents.supervisor_v2 import V1ServiceUnavailable

        raise V1ServiceUnavailable(
            "abbreviation lookup needs a sync context; refusing to guess"
        )

    return _lookup


@dataclass
class V2Ingress:
    """One request's live v2 ingress: runtime context + owned sessions.

    ``lease_session`` backs ONLY the retention-lease repository (M4a);
    ``evidence_session`` backs the governor/hydrator/builder. Both close on
    ``aclose``. Nodes flush; ``commit_evidence`` commits the evidence unit
    of work AFTER the terminal checkpoint succeeds (terminal lease release
    stays with the T8 outer runner — never here, never in the finalizer).
    """

    runtime_context: Any
    initial_state: Any
    plan_resolver: PlanBindingResolver
    lease_session: Any
    evidence_session: Any
    _closed: bool = field(default=False, init=False)

    async def commit_evidence(self) -> None:
        await self.evidence_session.commit()

    async def rollback_evidence(self) -> None:
        try:
            await self.evidence_session.rollback()
        except Exception:  # noqa: BLE001 — close-path best effort
            logger.warning("v2 ingress evidence rollback failed", exc_info=True)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for session in (self.lease_session, self.evidence_session):
            try:
                close = getattr(session, "close", None)
                if callable(close):
                    result = close()
                    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                        await result
            except Exception:  # noqa: BLE001 — close-path best effort
                logger.warning("v2 ingress session close failed", exc_info=True)


#: Bound on the rolling-summary window loaded per ingress turn (Task 8 fix
#: round 1, Important-2): the summary is the concatenation of every
#: ExchangeSummary row, so an unbounded read is O(thread) DB + checkpoint
#: growth per turn. ``summary_version``/``built_through_message_id`` still
#: come from the highest ``exchange_index`` (DESC-first read).
MAX_HISTORY_SUMMARIES = 20


def _merge_known_documents(
    current: tuple[Any, ...],
    conversation_resources: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Merge current-turn known resources with conversation candidates (Task 1B).

    Current-turn entries (``attachment``/``ui_selection``/``api_explicit``)
    keep their order and semantics untouched; conversation survivors follow,
    renumbered deterministically (``conv-1`` …) after cross-dedupe by
    document UUID so a re-attached or already-known document never shifts
    conversation ordinals or creates a duplicate entry.
    """
    seen = {known.document_id for known in current}
    survivors: list[Any] = []
    for known in conversation_resources or ():
        if known.document_id in seen:
            continue
        seen.add(known.document_id)
        survivors.append(known)
    renumbered = tuple(
        known.model_copy(update={"resource_id": f"conv-{index + 1}"})
        for index, known in enumerate(survivors)
    )
    return tuple(current) + renumbered


async def load_conversation_for_thread(
    session_factory: Callable[[], Any],
    thread_id: str,
    max_recent_turns: int = 20,
) -> Any:
    """Load the persisted discourse window for one v2 ingress turn (Phase 4C).

    Reads the last ``max_recent_turns`` ``ChatMessage`` rows (DB-side
    DESC + LIMIT, role/content columns only) plus the newest
    ``MAX_HISTORY_SUMMARIES`` ``ExchangeSummary`` rows for the thread
    and projects them through
    ``adapters/conversation.py::context_from_legacy`` (typed entities +
    derived ``last_focus``). Returns an empty ``ConversationHistory`` when
    the thread is not a persisted session (``standalone-...``), when no
    history exists, or when any load fails (fail-open: a turn without
    history is always safer than a failed turn).

    Returns a ``ConversationHistory`` bundle: the existing label/text-only
    ``ConversationContext`` plus server-issued ``KnownDocumentResource``
    candidates (``source="conversation"``) projected from the
    ``document_ids``/``citations``/``sources`` columns of the same bounded
    window — never UUIDs parsed from message text. Malformed identity
    values are ignored.

    Confinement: only labels/text enter ``ConversationContext``; document
    UUIDs travel exclusively in the typed ``resources`` (candidates only),
    so history can never reauthorize an out-of-scope resource — resolution
    always re-checks the current runtime scope.
    """
    from app.services.agents.v2.adapters.conversation import (
        DEFAULT_RECENT_TURN_LIMIT,
        ConversationHistory,
        context_from_legacy,
        conversation_resources_from_legacy,
    )

    limit = max(0, int(max_recent_turns or DEFAULT_RECENT_TURN_LIMIT))
    empty = ConversationHistory.empty()
    try:
        thread_uuid = UUID(str(thread_id))
    except (ValueError, AttributeError, TypeError):
        return empty
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import load_only

        from app.models.chat_message import ChatMessage
        from app.models.exchange_summary import ExchangeSummary

        async with session_factory() as db:
            # Task 8 fix round 1 (Important-2): DB-side DESC + LIMIT,
            # then reverse to chronological. Only the contract-consumed
            # columns (role/content) are loaded — heavy JSON columns
            # (thinking/agent_steps/sources/...) never leave the row.
            msg_result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == thread_uuid)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit if limit else DEFAULT_RECENT_TURN_LIMIT)
                .options(
                    load_only(
                        ChatMessage.role,
                        ChatMessage.content,
                        ChatMessage.document_ids,
                        ChatMessage.citations,
                        ChatMessage.sources,
                    )
                )
            )
            messages = list(reversed(msg_result.scalars().all()))
            sum_result = await db.execute(
                select(ExchangeSummary)
                .where(ExchangeSummary.session_id == thread_uuid)
                .order_by(ExchangeSummary.exchange_index.desc())
                .limit(MAX_HISTORY_SUMMARIES)
            )
            summaries = list(sum_result.scalars().all())
        if not messages and not summaries:
            return empty
        # Labels/text only enter the discourse context; server-issued
        # document identities project separately onto typed candidates.
        recent = messages[-limit:] if limit else []
        return ConversationHistory(
            context=context_from_legacy(
                messages=recent,
                exchange_summaries=summaries,
                max_recent_turns=limit,
            ),
            resources=conversation_resources_from_legacy(recent),
        )
    except Exception:  # noqa: BLE001 — fail-open to an empty window
        logger.warning(
            "v2 ingress history load failed; continuing without history",
            exc_info=True,
        )
        return empty


@asynccontextmanager
async def build_v2_ingress(
    *,
    user_id: UUID,
    authenticated_workspace_ids: Any,
    requested_workspace_ids: Any | None = None,
    raw_query: str,
    thread_id: str,
    request_id: str | None = None,
    run_id: str | None = None,
    can_read_people: bool = False,
    allowed_capabilities: frozenset[str] | None = None,
    deadline_seconds: float = 120.0,
    known_documents: tuple[Any, ...] = (),
    document_ids: tuple[UUID, ...] = (),
    session_factory: Callable[[], Any] | None = None,
    lease_session_factory: Callable[[], Any] | None = None,
    preprocess: Callable[[str], Any] | None = None,
    abbreviation_lookup: Callable[[str], str | None] | None = None,
    available_services: frozenset[str] | None = None,
    load_history: bool = True,
    max_recent_turns: int = 20,
) -> AsyncIterator[V2Ingress]:
    """Build the real request-scoped v2 runtime for one ingress turn.

    Scope is ``authenticated ∩ requested`` (never widened). The lease
    repository owns a DEDICATED session (M4a): when no explicit
    ``lease_session_factory`` is given, a separate ``async_session_maker()``
    session is opened for leases and a different one for evidence — the
    shared app session is never injected into ``binding_node``'s commit
    path. Every turn gets a fresh ``AnswerDraftChannel`` (runtime-only
    truncation handoff) and a fresh ``PlanBindingResolver`` for T8 to feed.
    """
    from app.services.agents.supervisor_v2 import (
        DeterministicSemanticAdapter,
        V1BindingResolver,
        V1PeopleLookupService,
        V1ServiceBundle,
        build_graph_runtime_context,
        build_runtime_services,
        build_v2_capability_registry,
    )
    from app.services.agents.v2.capabilities import CapabilityRuntimeContext
    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.request import RequestContext
    from app.services.agents.v2.evidence_store.governance import EvidenceGovernor
    from app.services.agents.v2.evidence_store.hydration import (
        GovernorEvidenceHydrator,
    )
    from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel
    from app.services.agents.v2.synthesis.adapter import StructuredLLMDraftBuilder
    from app.services.agents.v2.synthesis.citations import StoreCitationResolver
    from app.services.llm import get_main_provider_for_synthesis
    from app.services.agents.v2.planning import AdaptivePlanner, AdaptiveReplanner
    from app.services.agents.v2.semantic.document_identity import (
        DocumentIdentityResolver,
    )
    from app.core.config import settings
    from app.services.agents.v2.semantic.intent import IntentClassifier
    from app.services.agents.v2.semantic.multi_intent import MultiIntentClassifier
    from app.services.agents.supervisor_v2 import build_initial_v2_state

    if session_factory is None:
        from app.core.database import async_session_maker

        session_factory = async_session_maker
    if lease_session_factory is None:
        from app.core.database import async_session_maker

        lease_session_factory = async_session_maker

    scope = resolve_runtime_scope(
        authenticated_ids=authenticated_workspace_ids,
        requested_ids=requested_workspace_ids,
    )
    request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    # M4a via T6's dedicated_retention_leases: the lease repository owns a
    # session used for nothing else for the whole request; the evidence
    # session below is a different session. The lease CM must span the
    # `yield`, so it is entered through an AsyncExitStack whose close runs
    # in the `finally` below — deterministic on every path (normal, raise,
    # cancellation), unlike a hand-rolled __aenter__/__aexit__ pair.
    from app.services.agents.supervisor_v2 import dedicated_retention_leases

    stack = AsyncExitStack()
    lease_session = None
    evidence_session = None
    ingress = None
    try:
        leases = await stack.enter_async_context(
            dedicated_retention_leases(lease_session_factory)
        )
        lease_session = leases.session
        evidence_session = session_factory()
        capability_runtime = CapabilityRuntimeContext(
            request_id=request_id,
            run_id=run_id,
            user_id=user_id,
            workspace_ids=scope,
            can_read_people=bool(can_read_people),
            allowed_capabilities=(
                frozenset(allowed_capabilities)
                if allowed_capabilities is not None
                else DEFAULT_V2_ALLOWED_CAPABILITIES
            ),
            deadline_at=datetime.now(timezone.utc)
            + timedelta(seconds=deadline_seconds),
        )

        governor = EvidenceGovernor(evidence_session)
        evidence_builder = GovernorEvidenceBuilder(governor, run_id=run_id)
        plan_resolver = PlanBindingResolver()
        hydrator = GovernorEvidenceHydrator(governor)
        # `leases` is T6's dedicated_retention_leases repository: its
        # session is used for nothing else for the whole request (M4a).

        semantic_adapter = DeterministicSemanticAdapter(
            preprocess=preprocess
            or make_v1_preprocess_closure(
                user_id=user_id,
                workspace_ids=scope,
                document_ids=tuple(document_ids or ()),
                session_id=thread_id,
                can_read_people=bool(can_read_people),
                session_factory=session_factory,
            ),
            # Final review I2: exactly one request-scoped v1 identity
            # wrapper per ingress turn; the request-scoped cache behind it
            # means repeated semantic draft builds resolve once. The
            # existing v2 binding resolver stays the only revision-pin
            # authority. Task 8 fix round 1: the live people permission
            # gates person-mention resolution in ``build_draft``
            # (default-deny).
            identity_resolver=DocumentIdentityResolver(),
            session_factory=session_factory,
            workspace_ids=scope,
            can_read_people=bool(can_read_people),
        )
        # F1 role policy: the production semantic adapter emits
        # ``requested_role=None`` for every user reference, so the wired
        # resolver supplies the ``"target"`` default — otherwise every
        # resolved reference fails closed before any DB access.
        binding_resolver = V1BindingResolver(
            session_factory=session_factory, default_role="target"
        )
        # One request-scoped people service instance shared by the
        # capability registry AND the runtime services bag: the capability
        # fills its raw-record/display snapshot during dispatch, and the
        # non-LLM people presentation plus the streaming adapter read the
        # same snapshot for the v1 people card — no second Mongo round trip.
        people_lookup = V1PeopleLookupService()
        bundle = V1ServiceBundle(
            session_factory=session_factory,
            evidence=evidence_builder,
            resolver=plan_resolver,
            user_id=user_id,
            workspace_id=scope[0] if scope else None,
            abbreviation_lookup=abbreviation_lookup
            if abbreviation_lookup is not None
            else make_abbreviation_lookup(session_factory),
            people_lookup=people_lookup,
        )
        registry = build_v2_capability_registry(
            capability_runtime,
            bundle=bundle,
            evidence=evidence_builder,
            resolver=plan_resolver,
            **(
                {"available_services": available_services}
                if available_services is not None
                else {}
            ),
        )
        services = build_runtime_services(
            retention_leases=leases,
            semantic_adapter=semantic_adapter,
            binding_resolver=binding_resolver,
            capability_registry=registry,
            chat_messages=ChatMessagesService(session_factory),
            authorization=V2AuthorizationService(session_factory),
            evidence_hydrator=hydrator,
            answer_draft_channel=AnswerDraftChannel(),
            # Exactly one privacy-safe synthesis builder per ingress turn
            # (spec §7.1): the effective ``main`` provider under
            # content-suppressed tracing — never the generic full-content
            # wrapper, since prompts carry hydrated evidence plaintext.
            answer_draft_builder=StructuredLLMDraftBuilder(
                get_main_provider_for_synthesis
            ),
            # Exactly one typed v1 intent adapter per ingress turn (Phase
            # 4A, Task 2): the request-scoped cache behind it means repeated
            # semantic draft builds classify once per turn.
            intent_classifier=IntentClassifier(),
            # Multi-intent routing (spec §33): exactly one
            # MultiIntentClassifier per ingress turn, constructed ONLY
            # when the feature flag is on. Flag-off leaves the slot
            # ``None`` and never constructs the service, so the
            # deterministic Phase-4A path stays byte-identical.
            multi_intent_classifier=(
                MultiIntentClassifier()
                if getattr(
                    settings, "V2_MULTI_INTENT_ROUTING_ENABLED", False
                )
                else None
            ),
            # Exactly one governed planner per ingress turn (Phase 5,
            # Task 10): proposal-only initial planning behind the existing
            # validate/lease/checkpoint/scheduler boundary. Deterministic
            # skills still win when they cover the work type (a covering
            # refusal is final); the model path runs only for uncovered
            # work types and never dispatches tools.
            adaptive_planner=AdaptivePlanner(),
            # Exactly one governed replanner per ingress turn (Phase 5,
            # Task 11): proposal-only bounded append-only replanning from
            # minimized evaluator gaps behind the existing
            # validate/lease/checkpoint/scheduler boundary. The
            # deterministic gap policy still wins when it can construct
            # (zero model calls); the model path serves only advisable
            # gaps the deterministic policy cannot build and never
            # dispatches tools.
            adaptive_replanner=AdaptiveReplanner(),
            # The exact same request-scoped resolver the document
            # capabilities resolve through: the shared TaskScheduler feeds
            # it from the checkpointed plan + bindings before dispatch.
            pinned_target_resolver=plan_resolver,
            # Exactly one citation resolver per ingress turn (spec §11.1):
            # the authoritative document/revision/evidence stores behind
            # the single CitationProjector owner.
            citation_resolver=StoreCitationResolver(session_factory),
            # The same request-scoped instance the capability registry
            # dispatches through: its raw-record/display snapshot feeds the
            # v1 people card without a second Mongo round trip.
            people_lookup=people_lookup,
        )
        runtime_context = build_graph_runtime_context(
            capability_runtime, services=services
        )
        # Phase 4C (Task 8): production ingress supplies the persisted
        # discourse window (labels/text only — never document identity,
        # so history cannot reauthorize out-of-scope resources).
        # Fail-open: history load failures yield an empty window.
        conversation = None
        conversation_resources: tuple[Any, ...] = ()
        if load_history:
            history = await load_conversation_for_thread(
                session_factory, thread_id, max_recent_turns=max_recent_turns
            )
            conversation = history.context
            conversation_resources = history.resources
        initial_state = build_initial_v2_state(
            request=RequestContext(
                contract_version=CONTRACT_VERSION,
                request_id=request_id,
                thread_id=thread_id,
                original_query=raw_query,
                known_documents=_merge_known_documents(
                    tuple(known_documents or ()), conversation_resources
                ),
            ),
            conversation=conversation,
        )
        ingress = V2Ingress(
            runtime_context=runtime_context,
            initial_state=initial_state,
            plan_resolver=plan_resolver,
            lease_session=lease_session,
            evidence_session=evidence_session,
        )
        yield ingress
    finally:
        if ingress is not None:
            await ingress.aclose()
        elif evidence_session is not None:
            # Pre-yield failure: the evidence session never reached the
            # ingress; close it here (the lease session closes via the
            # stack below).
            try:
                await evidence_session.close()
            except Exception:  # noqa: BLE001 — close-path best effort
                logger.warning(
                    "v2 ingress session close failed", exc_info=True
                )
        await stack.aclose()


# F8: the interim T7 ``run_v2_turn_sse`` runner was removed — it had no
# production caller (all entrypoints stream through the reviewed
# ``stream_v2_turn_events``/``stream_v2_turn_to_sse`` adapter) and shared
# the pre-F4 suspend-detection gap. No unreferenced helpers remain with
# it (its imports were function-local).
