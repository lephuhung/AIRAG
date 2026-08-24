/**
 * Admin LLM runtime-config types — V2 two-level architecture.
 * Mirrors the REST contract of /api/v1/admin/llm-config (§12 of
 * docs/plan-llm-runtime-config.md):
 *   - Connections hold endpoint + key, declared once.
 *   - Role assignments reference a connection + pick a model from it.
 */

export const ENV_CONN_ID = "@env";

export type LlmRole =
  | "main"
  | "vision"
  | "thinking"
  | "memory_agent"
  | "kg_extract"
  | "graphiti"
  | "stt"
  | "tts"
  | "embedding"
  | "rerank";

export const LLM_ROLES: LlmRole[] = [
  "main",
  "vision",
  "thinking",
  "memory_agent",
  "kg_extract",
  "graphiti",
  "stt",
  "tts",
  "embedding",
  "rerank",
];

/** A stored endpoint connection (key `llm_conn.<conn_id>` server-side). */
export interface ConnectionInfo {
  name: string;
  provider: string;
  base_url: string;
  has_api_key: boolean;
  /** Masked key from the server, e.g. "sk-••••a1b2" — never the real key. */
  masked_api_key?: string;
  extra?: Record<string, unknown>;
  updated_at?: string | null;
}

/** Assignment status for a single role as returned by GET /admin/llm-config. */
export interface RoleAssignmentStatus {
  /** Connection id or "@env" (.env defaults). */
  conn_id: string;
  /** Effective model currently in use (resolved). */
  model: string;
  /** "db" when an override exists; "env" = running .env defaults. */
  source: "db" | "env";
  resolved: {
    provider: string;
    base_url: string;
    model: string;
    masked_api_key?: string;
  };
  updated_at?: string | null;
  updated_by?: string | null;
}

/** GET /api/v1/admin/llm-config */
export interface LlmConfigState {
  roles: Record<LlmRole, RoleAssignmentStatus>;
  connections: Record<string, ConnectionInfo>;
  version: number;
}

/** PUT /api/v1/admin/llm-config/connections/{conn_id} */
export interface ConnectionUpsertPayload {
  name?: string;
  provider: string;
  base_url?: string;
  /** Omit/blank → keep the stored key; non-empty replaces it. Never returned. */
  api_key?: string;
  extra?: Record<string, unknown>;
}

/** PUT response for a connection save. */
export interface ConnectionSaveResult {
  ok: boolean;
  conn_id: string;
  version: number;
}

/** DELETE /api/v1/admin/llm-config/connections/{conn_id}?force= */
export interface ConnectionDeleteResult {
  deleted: boolean;
  referencing_roles: string[];
}

/** PUT /api/v1/admin/llm-config/{role} — V2 assignment shape. */
export interface RoleAssignPayload {
  /** Existing connection id or "@env". */
  conn_id: string;
  model?: string;
  extra?: Record<string, unknown>;
}

/** PUT /{role} response */
export interface RoleAssignResult {
  ok: boolean;
  role: string;
  conn_id: string;
  resolved: {
    provider: string;
    base_url: string;
    model: string;
    masked_api_key?: string;
  };
  version: number;
}

/** POST /api/v1/admin/llm-config/test — full trial config, nothing persisted. */
export interface LlmTestRequest {
  provider: string;
  base_url?: string;
  model?: string;
  api_key?: string;
  is_vllm?: boolean;
}

export interface LlmTestResult {
  ok: boolean;
  latency_ms?: number;
  models: string[];
  models_list_available: boolean;
  is_vllm_hint?: boolean;
  error?: string;
}

/** POST /api/v1/admin/llm-config/models — light listing only, no save.
 *  Accepts EITHER {conn_id} OR a raw {provider, base_url, api_key} triple. */
export interface LlmModelsRequest {
  conn_id?: string;
  provider?: string;
  base_url?: string;
  api_key?: string;
}

export interface LlmModelsResponse {
  ok: boolean;
  models: string[];
  /** "endpoint" when listed, "none" when blocked/unavailable. */
  source: string;
}
