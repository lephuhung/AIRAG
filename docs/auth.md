# Auth — how to authenticate the harness

Two credential types resolve to the same principal `User`; a request runs with
exactly that user's workspace/tenant permissions.

| Type | Header | Accepted by | Notes |
|------|--------|-------------|-------|
| **JWT bearer** | `Authorization: Bearer <access_token>` | every protected route | 30-min expiry (`JWT_ACCESS_TOKEN_EXPIRE_MINUTES`) |
| **API key** | `X-API-Key: nrk_...` | only routes depending on `get_principal` | shown once at creation; sha256-hashed at rest |

**Which routes take which** (`app/core/deps.py`):
- `get_current_active_user` (JWT only) — the vast majority: `/rag/*`,
  `/rag/chat/sessions/*`, `/workspaces`, `/documents/*`, `/config/*`, …
  **`/rag/debug-chat/{ws}` is here → the A/B + RAG harness needs a JWT.**
- `get_principal` (JWT **or** `X-API-Key`) — the external agent entrypoint
  `/rag/chat/agent-lg/stream` and `/rag/chat/agent-lg/{ws}/stream`.
- `require_superadmin` — `/audit-logs`, some `/workers/*`, telegram bot config.

## Get a JWT — login

```bash
API=http://localhost:8080/api/v1
TOKEN=$(curl -s -X POST $API/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@hrag.local","password":"'"$AB_PASSWORD"'"}' \
  | jq -r .access_token)
curl -s $API/auth/me -H "Authorization: Bearer $TOKEN" | jq .email
```

Login body: `{email, password, totp_code?}`. If the account has 2FA on, a missing
code returns `401 TWO_FACTOR_REQUIRED` (frontend sentinel) — pass `totp_code`.
Response: `{access_token, refresh_token, user}`. Refresh via
`POST /auth/refresh {refresh_token}`.

## Mint a JWT without a password (test-only, inside the container)

When you have DB access but not a plaintext password (e.g. verifying the harness):

```bash
docker exec hrag-backend python -c "
import asyncio; from sqlalchemy import select
from app.core.database import async_session_maker
from app.models.user import User
from app.core.security import create_access_token
async def m():
    async with async_session_maker() as db:
        u=(await db.execute(select(User).where(User.is_active==True).limit(1))).scalar_one()
        print(create_access_token(u.id))
asyncio.run(m())"
```

This is exactly how the A/B harness was verified. Prefer real login for anything
user-facing; the minted token is a shortcut for local test drives only.

## API keys (service-to-service)

Create (needs a JWT): `POST /integrations/api-keys` → returns the plaintext
`nrk_...` **once**. List: `GET /integrations/api-keys`. Revoke:
`DELETE /integrations/api-keys/{id}`. Send as `X-API-Key`. Only the agent-lg
stream endpoints accept it — `debug-chat` does **not**.

## How the harness consumes auth

`backend/scripts/ab_eval.py` resolves, in order: `--token` → `AB_TOKEN` →
auto-login with `AB_USER` + `AB_PASSWORD` (+ `AB_TOTP`) against
`{API_ROOT}/auth/login`. The `Makefile` `ab` target forwards
`AB_TOKEN/AB_USER/AB_PASSWORD/AB_TOTP` into the container. Set them once:

```bash
export AB_USER=admin@hrag.local AB_PASSWORD=...   # or export AB_TOKEN=<jwt>
```
