# Backend Route Permissions

Document version: 2026-04-01
Branch: `KG-building`

---

## Authentication & Authorization Model

### Dependencies (`app/core/deps.py`)

| Dependency | Checks | Used for |
|---|---|---|
| `get_current_user` | Valid JWT token | All authenticated endpoints |
| `get_current_active_user` | JWT + `user.is_active=True` | Most endpoints (approved accounts) |
| `require_superadmin` | JWT + `user.is_superadmin=True` | Admin-only endpoints |
| `verify_workspace_access(ws_id, user, db)` | Workspace visibility + membership | Workspace-scoped endpoints |

### Workspace Access Logic (`verify_workspace_access`)

```
Superadmin          → access all workspaces
Legacy (owner=None) → all authenticated users
public visibility   → all authenticated users
tenant visibility   → approved tenant members only
personal visibility → owner only
```

### User Roles

| Role | Flag | Access Scope |
|---|---|---|
| SuperAdmin | `user.is_superadmin=True` | Full system + all workspaces |
| Tenant Admin | `TenantUser.role="admin"` | Own tenant management |
| Tenant Member | `TenantUser.role="member"` | Tenant workspaces (visibility-based) |
| Pending User | `user.is_active=False` | Auth only (`/auth/register` via invite bypasses) |
| Unauthenticated | No token | Public endpoints only |

---

## Route Permissions Matrix

### `/auth` — Authentication
| Route | Method | Auth Required | Notes |
|---|---|---|---|
| `/auth/register` | POST | **None** | Public. Creates inactive account unless invited |
| `/auth/login` | POST | **None** | Public |
| `/auth/refresh` | POST | **None** | Public (refresh token) |
| `/auth/me` | GET | Active user | Returns own profile |
| `/auth/me` | PUT | Active user | Update own name/password |
| `/auth/me/avatar` | POST | Active user | Upload own avatar |

---

### `/workspaces` — Knowledge Base Management
| Route | Method | Auth | Extra Permissions |
|---|---|---|---|
| `GET /workspaces` | GET | Active user | Lists workspaces user can see (filtered by visibility) |
| `POST /workspaces` | POST | Active user | Creates own personal workspace |
| `GET /workspaces/summary` | GET | Active user | Compact list for dropdowns |
| `GET /workspaces/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `PUT /workspaces/{workspace_id}` | PUT | Active user | Owner, tenant admin, or superadmin can edit name/description |
| `PUT /workspaces/{workspace_id}` (tenant/visibility) | PUT | **Superadmin only** | Only superadmin can change tenant_id or visibility |
| `DELETE /workspaces/{workspace_id}` | DELETE | Active user | Owner, tenant admin, or superadmin |

**Workspace edit/delete rules:**
- Superadmin: can edit/delete any workspace
- Non-superadmin with `owner_id`: only the owner
- Tenant workspace: tenant admin also qualifies

---

### `/documents` — Document Management
| Route | Method | Auth | Extra Permissions |
|---|---|---|---|
| `GET /documents/workspace/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `POST /documents/upload/{workspace_id}` | POST | Active user | `verify_workspace_access` |
| `POST /documents/upload/{workspace_id}/presign` | POST | Active user | `verify_workspace_access` |
| `POST /documents/upload/{workspace_id}/confirm` | POST | Active user | `verify_workspace_access` |
| `GET /documents/{document_id}` | GET | Active user | — |
| `GET /documents/{document_id}/markdown` | GET | Active user | — |
| `GET /documents/{document_id}/images` | GET | Active user | — |
| `GET /documents/{document_id}/download` | GET | Active user | — |
| `PATCH /documents/{document_id}` | PATCH | Active user | `verify_workspace_access` |
| `DELETE /documents/{document_id}` | DELETE | Active user | — |

---

### `/rag` — RAG Query & Processing
| Route | Method | Auth | Extra Permissions |
|---|---|---|---|
| `POST /rag/query/{workspace_id}` | POST | Active user | `verify_workspace_access` |
| `POST /rag/process/{document_id}` | POST | Active user | `verify_workspace_access` |
| `POST /rag/process-batch` | POST | Active user | — |
| `POST /rag/reindex/{document_id}` | POST | Active user | `verify_workspace_access` |
| `POST /rag/reindex-workspace/{workspace_id}` | POST | Active user | `verify_workspace_access` |
| `GET /rag/stats/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `GET /rag/chunks/{document_id}` | GET | Active user | — |
| `GET /rag/entities/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `GET /rag/relationships/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `GET /rag/graph/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `GET /rag/analytics/{workspace_id}` | GET | Active user | `verify_workspace_access` |
| `GET /rag/capabilities` | GET | Active user | — |
| `POST /rag/debug-chat/{workspace_id}` | POST | Active user | `verify_workspace_access` |

---

### `/rag/chat` — Chat Endpoints

#### Session-based (`/rag/chat/sessions`)
| Route | Method | Auth | Notes |
|---|---|---|---|
| `GET /rag/chat/sessions` | GET | Active user | Own sessions only (`user_id` filter) |
| `POST /rag/chat/sessions` | POST | Active user | Creates new session |
| `DELETE /rag/chat/sessions/{session_id}` | DELETE | Active user | Own session only |
| `GET /rag/chat/sessions/{session_id}/history` | GET | Active user | Own session only |
| `DELETE /rag/chat/sessions/{session_id}/history` | DELETE | Active user | Own session only |
| `POST /rag/chat/sessions/{session_id}/stream` | POST | Active user | Own session only (routes to LangGraph or legacy) |
| `POST /rag/chat/sessions/{session_id}/rate` | POST | Active user | Own session + message |

#### LangGraph Agent (`/rag/chat`)
| Route | Method | Auth | Notes |
|---|---|---|---|
| `POST /rag/chat/agent-lg/stream` | POST | Active user | Multi-workspace chat (all accessible workspaces) |
| `POST /rag/chat/agent-lg/{workspace_id}/stream` | POST | Active user | Single workspace + `verify_workspace_access` |

**Chat access note:** Chat does NOT use `verify_workspace_access` per workspace. Instead, `_get_accessible_workspaces()` returns all workspaces the user can access (public + own personal + tenant). The LLM searches across ALL accessible workspaces.

---

### `/tenants` — Tenant Management
| Route | Method | Auth | Notes |
|---|---|---|---|
| `POST /tenants` | POST | **Superadmin only** | Create tenant |
| `GET /tenants` | GET | **Superadmin only** | List all tenants with counts |
| `GET /tenants/invite/{token}` | GET | **None** | Public (validate invite link) |
| `GET /tenants/my` | GET | Active user | List user's own tenants |
| `GET /tenants/{tenant_id}` | GET | Tenant admin or Superadmin | — |
| `PUT /tenants/{tenant_id}` | PUT | **Superadmin only** | Update tenant |
| `DELETE /tenants/{tenant_id}` | DELETE | **Superadmin only** | Deactivate tenant |
| `POST /tenants/{tenant_id}/set-admin/{user_id}` | POST | **Superadmin only** | Assign tenant admin |
| `GET /tenants/{tenant_id}/users` | GET | Tenant admin or Superadmin | — |
| `POST /tenants/{tenant_id}/users/{user_id}/approve` | POST | Tenant admin or Superadmin | — |
| `POST /tenants/{tenant_id}/users/{user_id}/reject` | DELETE | Tenant admin or Superadmin | — |
| `DELETE /tenants/{tenant_id}/users/{user_id}` | DELETE | Tenant admin or Superadmin | — |
| `PUT /tenants/{tenant_id}/users/{user_id}/role` | PUT | Tenant admin or Superadmin | — |
| `POST /tenants/{tenant_id}/invites` | POST | Tenant admin or Superadmin | Create invite link |
| `GET /tenants/{tenant_id}/invites` | GET | Tenant admin or Superadmin | List invite links |
| `DELETE /tenants/{tenant_id}/invites/{invite_id}` | DELETE | Tenant admin or Superadmin | Revoke invite |

**Tenant admin check:** Superadmin bypasses the tenant admin check (implicit admin of all tenants).

---

### `/admin` — SuperAdmin Panel
**All routes require `require_superadmin`**

| Route | Method | Notes |
|---|---|---|
| `GET /admin/stats` | GET | System-wide statistics |
| `GET /admin/users` | GET | List all users (paginated, filterable) |
| `GET /admin/users/{user_id}` | GET | User detail + tenant memberships |
| `PUT /admin/users/{user_id}` | PUT | Update `is_active`, `is_superadmin`, `full_name` |
| `DELETE /admin/users/{user_id}` | DELETE | Delete user (cannot delete self) |
| `POST /admin/users/{user_id}/reset-password` | POST | Force reset password |

---

### `/workers` — Worker & Queue Management
**All routes require `require_superadmin`**

| Route | Method | Notes |
|---|---|---|
| `GET /workers/health` | GET | RabbitMQ + workers + pipeline health |
| `GET /workers/managed` | GET | List spawned subprocess workers |
| `POST /workers/start` | POST | Spawn worker subprocesses |
| `POST /workers/stop/{worker_type}` | POST | Stop worker subprocesses |
| `POST /workers/restart/{worker_type}` | POST | Restart all workers of type |
| `DELETE /workers/managed/{worker_type}` | DELETE | Remove dead worker entries |
| `GET /workers/dead-letter` | GET | Peek DLQ messages |
| `POST /workers/dead-letter/purge` | POST | Clear DLQ |
| `POST /workers/dead-letter/retry` | POST | Requeue DLQ messages |
| `DELETE /workers/queues/{queue_name}` | DELETE | Delete a queue |
| `GET /workers/overview` | GET | RabbitMQ queues + pipeline summary |
| `GET /workers/queues` | GET | List all hrag.* queues |
| `POST /workers/queues/{queue_name}/purge` | POST | Purge specific queue |
| `POST /workers/retry-failed` | POST | Retry all failed documents |
| `POST /workers/retry-failed/{document_id}` | POST | Retry single failed document |
| `GET /workers/pipeline` | GET | In-progress + failed documents |

---

### `/document-types` — Document Type Configuration
| Route | Method | Auth | Notes |
|---|---|---|---|
| `GET /document-types` | GET | Active user | List all (optionally include inactive) |
| `POST /document-types` | POST | **Superadmin only** | Create document type |
| `GET /document-types/{slug}` | GET | Active user | — |
| `PUT /document-types/{slug}` | PUT | **Superadmin only** | Update name/description/is_active |
| `DELETE /document-types/{slug}` | DELETE | **Superadmin only** | Soft-delete (is_active=False) |
| `GET /document-types/{slug}/prompt` | GET | Active user | Get global system prompt |
| `PUT /document-types/{slug}/prompt` | PUT | **Superadmin only** | Set global system prompt |
| `GET /document-types/{slug}/prompt/{workspace_id}` | GET | Active user | Get workspace-specific prompt |
| `PUT /document-types/{slug}/prompt/{workspace_id}` | PUT | **Superadmin only** | Set workspace-specific prompt |
| `DELETE /document-types/{slug}/prompt/{workspace_id}` | DELETE | **Superadmin only** | Remove workspace override |

---

### `/abbreviations` — Abbreviation Dictionary
| Route | Method | Auth | Notes |
|---|---|---|---|
| `POST /abbreviations` | POST | Active user | Create. Superadmin auto-activates |
| `GET /abbreviations` | GET | Active user | List with search/pagination |
| `GET /abbreviations/{id}` | GET | Active user | — |
| `PATCH /abbreviations/{id}` | PATCH | Owner or Superadmin | Owner can update text; only superadmin can toggle `is_active` |
| `DELETE /abbreviations/{id}` | DELETE | Owner or Superadmin | — |

---

### `/config` — Configuration Status
| Route | Method | Auth | Notes |
|---|---|---|---|
| `GET /config/status` | GET | Active user | Returns active LLM/embedding provider info |

---

### `/logs` — System Log Streaming
**All routes require `is_superadmin=True`**

| Route | Method | Notes |
|---|---|---|
| `GET /logs/stream` | GET | SSE stream of log file updates |
| `GET /logs/list` | GET | List available log files |
| `GET /logs/{filename}` | GET | Get last N lines of a log file |

---

### `/minio` — MinIO Webhook
| Route | Method | Auth | Notes |
|---|---|---|---|
| `POST /minio/events` | POST | **None** | S3 event webhook (internal network/MinIO) |

---

## Summary: Permission Levels

| Permission Level | Endpoints |
|---|---|
| **Public** (no auth) | `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `GET /tenants/invite/{token}`, `POST /minio/events` |
| **Authenticated user** (`is_active=True`) | Most endpoints — RAG, documents, workspaces, chat, abbreviations, config, document-types (read) |
| **Workspace access required** | Document CRUD, RAG queries, workspace-specific chat — uses `verify_workspace_access` |
| **Tenant admin or Superadmin** | Tenant user management, invite links |
| **Superadmin only** | Admin panel, workers, logs, document-type CRUD, tenant CRUD, abbreviation `is_active` toggle |
