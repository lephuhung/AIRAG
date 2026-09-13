"""Admin agent-evaluation surface (Phase 2, Task 7).

The ONLY per-request arm override: an authenticated superadmin may run one
evaluation turn against an explicitly chosen arm (``v1`` | ``v2``),
bypassing the configured default for that request alone. Ordinary chat
traffic never takes this path and ordinary request headers never select an
arm. Evaluation turns are not written to chat history.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.deps import get_db, require_superadmin
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/agent",
    tags=["agent_admin"],
    dependencies=[Depends(require_superadmin)],
)


class EvaluateRequest(BaseModel):
    """One admin evaluation turn against an explicitly chosen arm."""

    message: str = Field(..., min_length=1, max_length=5000)
    version: str = Field(
        ..., description="Arm to evaluate: 'v1' or 'v2' (admin override)."
    )
    workspace_ids: Optional[list[UUID]] = Field(
        default=None,
        description=(
            "Optional narrowing scope; intersected with the admin's "
            "accessible scope, never widened."
        ),
    )


class EvaluateResponse(BaseModel):
    version: str
    events: list[dict]


class RolloutControlResponse(BaseModel):
    """The DB rollout control row plus the environment ceilings."""

    enabled: bool
    shadow_percent: int
    canary_percent: int
    canary_workspaces: list[str]
    kill_switch: bool
    updated_by: Optional[str] = None
    version: int
    env_enabled: bool
    env_canary_percent: float
    env_canary_workspaces: list[str]
    effective_canary_percent: float


class RolloutControlUpdate(BaseModel):
    """Operator tuning for the control row (all fields optional)."""

    enabled: Optional[bool] = None
    canary_percent: Optional[int] = Field(default=None, ge=0, le=100)
    canary_workspaces: Optional[list[str]] = None
    kill_switch: Optional[bool] = None


@router.get("/status")
async def agent_status(
    user: User = Depends(require_superadmin),
) -> dict:
    """Read-only arm status: configured default + v2 readiness (no mutation)."""
    from app.services.agent.runtime_selector import (
        V2NotReadyError,
        configured_agent_version,
        require_v2_schema_ready,
    )

    configured = configured_agent_version()
    try:
        await require_v2_schema_ready()
        v2_ready, problems = True, []
    except V2NotReadyError as exc:
        v2_ready, problems = False, [str(exc)]
    except Exception as exc:  # noqa: BLE001 — readiness must never 500 status
        v2_ready, problems = False, [f"readiness probe failed: {exc}"]
    return {
        "configured_version": configured,
        "v2_ready": v2_ready,
        "problems": problems,
    }


@router.get("/rollout", response_model=RolloutControlResponse)
async def get_rollout_control(
    user: User = Depends(require_superadmin),
    db: Any = Depends(get_db),
) -> RolloutControlResponse:
    """Read the DB rollout control row + environment ceilings (no mutation)."""
    from app.services.agent.rollout_control import (
        effective_canary_percent,
        parse_canary_workspaces,
        read_canary_env,
        read_control_row,
    )

    env = read_canary_env()
    row = await read_control_row(db)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="rollout control row (id=1) is absent; run the v2 schema migration",
        )
    workspaces = list(parse_canary_workspaces(row.canary_workspaces))
    return RolloutControlResponse(
        enabled=bool(row.enabled),
        shadow_percent=int(row.shadow_percent or 0),
        canary_percent=int(row.canary_percent or 0),
        canary_workspaces=workspaces,
        kill_switch=bool(row.kill_switch),
        updated_by=row.updated_by,
        version=int(row.version or 1),
        env_enabled=env.enabled,
        env_canary_percent=env.canary_percent,
        env_canary_workspaces=list(env.canary_workspaces),
        effective_canary_percent=effective_canary_percent(
            env.canary_percent, int(row.canary_percent or 0)
        ),
    )


@router.put("/rollout", response_model=RolloutControlResponse)
async def update_rollout_control(
    body: RolloutControlUpdate,
    user: User = Depends(require_superadmin),
    db: Any = Depends(get_db),
) -> RolloutControlResponse:
    """Retune the control row (superadmin only; bumps ``version``)."""
    from sqlalchemy import select

    from app.models.agent_rollout_control import AgentRolloutControl

    result = await db.execute(
        select(AgentRolloutControl).where(AgentRolloutControl.id == 1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="rollout control row (id=1) is absent; run the v2 schema migration",
        )
    if body.enabled is not None:
        row.enabled = bool(body.enabled)
    if body.canary_percent is not None:
        row.canary_percent = int(body.canary_percent)
    if body.canary_workspaces is not None:
        row.canary_workspaces = [str(item) for item in body.canary_workspaces]
    if body.kill_switch is not None:
        row.kill_switch = bool(body.kill_switch)
    row.updated_by = str(user.email or user.id)
    row.version = int(row.version or 1) + 1
    await db.commit()
    return await get_rollout_control(user=user, db=db)


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_arm(
    body: EvaluateRequest,
    user: User = Depends(require_superadmin),
    db: Any = Depends(get_db),
) -> EvaluateResponse:
    """Run one evaluation turn on the chosen arm (superadmin only)."""
    events = await run_admin_evaluation(
        message=body.message,
        version=body.version,
        user=user,
        workspace_ids=body.workspace_ids,
        db=db,
    )
    return EvaluateResponse(version=body.version.strip().lower(), events=events)


async def run_admin_evaluation(
    *,
    message: str,
    version: str,
    user: Any,
    workspace_ids: Optional[list[UUID]] = None,
    db: Any = None,
) -> list[dict]:
    """Execute one evaluation turn on the admin-chosen arm.

    Guards (in order): authenticated-admin-only override resolution (401 /
    403 / 400 on bad value), lazy graph resolution through the selector
    (v2 readiness gated; unready maps to 503, never a silent v1 fallback),
    then a single-turn run whose terminal event is collected. Evaluation
    never writes chat history.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.services.agent.runtime_selector import (
        V2NotReadyError,
        resolve_agent_graph,
        resolve_request_version,
    )

    selected = resolve_request_version(user=user, admin_override=version)
    if selected == "v2":
        # The kill switch is authoritative even over an explicit admin
        # evaluation turn: an engaged brake refuses v2 instead of serving it.
        from app.services.agent.rollout_control import is_v2_killed

        if await is_v2_killed(db):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="v2 kill switch is engaged; refusing explicit v2 turn",
            )
    try:
        graph = await resolve_agent_graph(selected)
    except V2NotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    if db is None:
        from app.core.database import async_session_maker

        async with async_session_maker() as session:
            return await _run_selected_arm(
                selected, graph, session, user, message, workspace_ids
            )
    if isinstance(db, AsyncSession):
        return await _run_selected_arm(
            selected, graph, db, user, message, workspace_ids
        )
    async with db() as session:
        return await _run_selected_arm(
            selected, graph, session, user, message, workspace_ids
        )


async def _run_selected_arm(
    selected: str,
    graph: Any,
    db: Any,
    user: Any,
    message: str,
    workspace_ids: Optional[list[UUID]],
) -> list[dict]:
    if selected == "v2":
        return await _run_v2_eval(graph, db, user, message, workspace_ids)
    return await _run_v1_eval(graph, db, user, message, workspace_ids)


async def _accessible_scope(db: Any, user: Any) -> list[UUID]:
    from app.api.chat_agent import _get_accessible_workspaces

    return await _get_accessible_workspaces(db, user)


async def _run_v1_eval(
    graph: Any,
    db: Any,
    user: Any,
    message: str,
    workspace_ids: Optional[list[UUID]],
) -> list[dict]:
    from app.prompts.chat import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
    from app.services.agent.runtime_selector import resolve_runtime_scope
    from app.services.agent.streaming import build_initial_state, stream_agent_events

    authenticated = await _accessible_scope(db, user)
    scope = resolve_runtime_scope(
        authenticated_ids=authenticated, requested_ids=workspace_ids
    )
    if not scope:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No accessible workspaces in the requested scope.",
        )
    initial_state = build_initial_state(
        workspace_ids=list(scope),
        message=message,
        history=[],
        system_prompt=DEFAULT_SYSTEM_PROMPT + HARD_SYSTEM_PROMPT,
        enable_thinking=False,
        db=db,
        user_id=user.id,
        session_id=None,
        document_ids=None,
        user_can_use_people=bool(user.is_superadmin),
    )
    events: list[dict] = []
    async for event in stream_agent_events(graph, initial_state, channel="admin-eval"):
        events.append(event)
    return events


async def _run_v2_eval(
    graph: Any,
    db: Any,
    user: Any,
    message: str,
    workspace_ids: Optional[list[UUID]],
) -> list[dict]:
    """Run one admin evaluation turn through the reviewed v2 adapter.

    F4: this surface used to ``ainvoke`` the graph directly, catch only
    ``GraphInterrupt``, and 502 on a clarify route (a top-level suspend is
    RETURNED via ``__interrupt__``, not raised) while never releasing
    leases after a terminal. It now streams the turn through
    ``stream_v2_turn_events`` — the same suspension detection, one-terminal
    rule, and terminal lease release the user entrypoints use — and
    commits the evidence unit of work after the terminal event, rolling
    back when the stream itself raises (mirroring ``chat_agent_lg``).
    """
    from app.services.agent.runtime_selector import (
        build_v2_ingress,
        resolve_runtime_scope,
    )
    from app.services.agent.streaming import stream_v2_turn_events

    authenticated = await _accessible_scope(db, user)
    if not resolve_runtime_scope(
        authenticated_ids=authenticated, requested_ids=workspace_ids
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No accessible workspaces in the requested scope.",
        )
    thread_id = f"admin-eval-{uuid.uuid4().hex[:12]}"
    async with build_v2_ingress(
        user_id=user.id,
        authenticated_workspace_ids=authenticated,
        requested_workspace_ids=workspace_ids,
        raw_query=message,
        thread_id=thread_id,
        can_read_people=bool(user.is_superadmin),
    ) as ingress:
        try:
            events = [
                event
                async for event in stream_v2_turn_events(
                    graph=graph,
                    runtime_context=ingress.runtime_context,
                    thread_id=thread_id,
                    initial_state=ingress.initial_state,
                    plan_resolver=ingress.plan_resolver,
                )
            ]
        except Exception:
            await ingress.rollback_evidence()
            raise
        await ingress.commit_evidence()
        return events
