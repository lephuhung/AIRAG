"""
AuditMiddleware — automatically records management mutations to the audit log.

A small rule table maps (method, path) → (action, resource_type). Any matching
2xx mutation is logged with the acting user, the resource id parsed from the
path, and a human label pulled from the JSON response body when available.

Domains covered automatically: workspaces, document-types, tenants, admin/users.
The /abbreviations domain is instrumented EXPLICITLY inside its endpoints (so it
is deliberately NOT listed here — avoiding double entries).
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

MUTATION_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Ordered: most specific patterns first. id_group selects which named group is
# treated as the primary resource id.
RULES: list[dict] = [
    # ── Workspaces ────────────────────────────────────────────────────────
    {"m": "POST", "re": r"^/api/v1/workspaces/?$", "action": "create", "type": "workspace"},
    {"m": "POST", "re": r"^/api/v1/workspaces/(?P<id>[^/]+)/set-default$", "action": "set_default", "type": "workspace"},
    {"m": "PUT", "re": r"^/api/v1/workspaces/(?P<id>[^/]+)$", "action": "update", "type": "workspace"},
    {"m": "DELETE", "re": r"^/api/v1/workspaces/(?P<id>[^/]+)$", "action": "delete", "type": "workspace"},
    # ── Document types ────────────────────────────────────────────────────
    {"m": "POST", "re": r"^/api/v1/document-types/?$", "action": "create", "type": "document_type"},
    {"m": "PUT", "re": r"^/api/v1/document-types/(?P<id>[^/]+)/prompt/(?P<ws>[^/]+)$", "action": "update_prompt", "type": "document_type"},
    {"m": "DELETE", "re": r"^/api/v1/document-types/(?P<id>[^/]+)/prompt/(?P<ws>[^/]+)$", "action": "delete_prompt", "type": "document_type"},
    {"m": "PUT", "re": r"^/api/v1/document-types/(?P<id>[^/]+)/prompt$", "action": "update_prompt", "type": "document_type"},
    {"m": "PUT", "re": r"^/api/v1/document-types/(?P<id>[^/]+)$", "action": "update", "type": "document_type"},
    {"m": "DELETE", "re": r"^/api/v1/document-types/(?P<id>[^/]+)$", "action": "delete", "type": "document_type"},
    # ── Tenants ───────────────────────────────────────────────────────────
    {"m": "POST", "re": r"^/api/v1/tenants/?$", "action": "create", "type": "tenant"},
    {"m": "PUT", "re": r"^/api/v1/tenants/(?P<id>[^/]+)$", "action": "update", "type": "tenant"},
    {"m": "DELETE", "re": r"^/api/v1/tenants/(?P<id>[^/]+)$", "action": "delete", "type": "tenant"},
    {"m": "POST", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/set-admin/(?P<uid>[^/]+)$", "action": "set_admin", "type": "tenant_member", "id_group": "uid"},
    {"m": "POST", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/users/(?P<uid>[^/]+)/approve$", "action": "approve", "type": "tenant_member", "id_group": "uid"},
    {"m": "POST", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/users/(?P<uid>[^/]+)/reject$", "action": "reject", "type": "tenant_member", "id_group": "uid"},
    {"m": "PUT", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/users/(?P<uid>[^/]+)/role$", "action": "update_role", "type": "tenant_member", "id_group": "uid"},
    {"m": "DELETE", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/users/(?P<uid>[^/]+)$", "action": "delete", "type": "tenant_member", "id_group": "uid"},
    {"m": "POST", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/invites$", "action": "create", "type": "tenant_invite"},
    {"m": "DELETE", "re": r"^/api/v1/tenants/(?P<id>[^/]+)/invites/(?P<iid>[^/]+)$", "action": "delete", "type": "tenant_invite", "id_group": "iid"},
    # ── Users (admin) ─────────────────────────────────────────────────────
    {"m": "POST", "re": r"^/api/v1/admin/users/(?P<id>[^/]+)/reset-password$", "action": "reset_password", "type": "user"},
    {"m": "PUT", "re": r"^/api/v1/admin/users/(?P<id>[^/]+)$", "action": "update", "type": "user"},
    {"m": "DELETE", "re": r"^/api/v1/admin/users/(?P<id>[^/]+)$", "action": "delete", "type": "user"},
]

for _r in RULES:
    _r["rx"] = re.compile(_r["re"])

# Vietnamese labels used to build a readable one-line summary.
_RES_VI = {
    "workspace": "không gian (workspace)",
    "document_type": "loại tài liệu",
    "tenant": "tổ chức (tenant)",
    "tenant_member": "thành viên tổ chức",
    "tenant_invite": "lời mời tổ chức",
    "user": "người dùng",
    "abbreviation": "từ viết tắt",
}
_ACT_VI = {
    "create": "Tạo",
    "update": "Cập nhật",
    "delete": "Xóa",
    "set_default": "Đặt mặc định",
    "set_admin": "Phân quyền quản trị",
    "approve": "Phê duyệt",
    "reject": "Từ chối",
    "update_role": "Đổi vai trò",
    "update_prompt": "Cập nhật system prompt",
    "delete_prompt": "Xóa system prompt",
    "reset_password": "Đặt lại mật khẩu",
}

_LABEL_KEYS = ("name", "full_name", "title", "text", "short_form", "slug", "email")


def _match(method: str, path: str):
    for rule in RULES:
        if rule["m"] == method:
            m = rule["rx"].match(path)
            if m:
                return rule, m
    return None, None


def _extract_label(body: bytes) -> str | None:
    if not body:
        return None
    try:
        data = json.loads(body)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for k in _LABEL_KEYS:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:255]
    return None


def build_summary(action: str, resource_type: str, label: str | None, resource_id: str | None) -> str:
    verb = _ACT_VI.get(action, action)
    res = _RES_VI.get(resource_type, resource_type)
    name = f' "{label}"' if label else (f" {resource_id}" if resource_id else "")
    return f"{verb} {res}{name}".strip()


async def _resolve_actor(request) -> tuple[uuid.UUID | None, str | None, str | None]:
    """Best-effort actor identity from the bearer token."""
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None, None, None
    token = auth.split(" ", 1)[1].strip()
    try:
        from app.core.security import decode_token

        payload = decode_token(token)
        sub = payload.get("sub")
        actor_id = uuid.UUID(sub) if sub else None
    except Exception:
        return None, None, None
    if actor_id is None:
        return None, None, None
    try:
        from sqlalchemy import select
        from app.core.database import async_session_maker
        from app.models.user import User

        async with async_session_maker() as db:
            user = (
                await db.execute(select(User).where(User.id == actor_id))
            ).scalar_one_or_none()
            if user is not None:
                return user.id, user.email, user.full_name
    except Exception:
        pass
    return actor_id, None, None


def _client_ip(request) -> str | None:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        method = request.method
        path = request.url.path

        if method not in MUTATION_METHODS:
            return await call_next(request)

        rule, match = _match(method, path)
        if rule is None:
            return await call_next(request)

        # Buffer the response so we can inspect status + body, then re-emit it.
        response = await call_next(request)
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        headers = dict(response.headers)
        headers.pop("content-length", None)
        new_response = Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )

        if 200 <= response.status_code < 300:
            try:
                await self._record(request, rule, match, body, response.status_code)
            except Exception as e:  # pragma: no cover - best effort
                logger.warning(f"[audit] middleware record failed for {path}: {e}")

        return new_response

    async def _record(self, request, rule, match, body: bytes, status_code: int) -> None:
        from app.services.audit_service import AuditService

        id_group = rule.get("id_group", "id")
        groups = match.groupdict()
        resource_id = groups.get(id_group) or groups.get("id")
        label = _extract_label(body)
        action = rule["action"]
        resource_type = rule["type"]

        actor_id, actor_email, actor_name = await _resolve_actor(request)
        summary = build_summary(action, resource_type, label, resource_id)

        await AuditService.record(
            action=action,
            resource_type=resource_type,
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            resource_id=resource_id,
            resource_label=label,
            summary=summary,
            extra={k: v for k, v in groups.items() if v} or None,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            ip_address=_client_ip(request),
            source="auto",
        )
