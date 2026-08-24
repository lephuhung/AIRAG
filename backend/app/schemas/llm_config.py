"""Pydantic schemas for the admin LLM runtime-config API (app/api/llm_config.py).

V2 two-level architecture (docs/plan-llm-runtime-config.md §12):
connections hold endpoint + key; role assignments reference a connection.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ConnectionUpsert(BaseModel):
    """Body for PUT /admin/llm-config/connections/{conn_id}.

    api_key left None/blank → keep the previously stored key (URL/name-only
    update); a non-empty value replaces it. Never returned to the client.
    """

    name: str = ""
    provider: str = Field(..., description="gemini | ollama | openai_compatible")
    base_url: str = ""
    api_key: str | None = None
    extra: dict | None = Field(default=None, description="is_vllm, max_concurrency, ...")


class RoleAssignment(BaseModel):
    """Body for PUT /admin/llm-config/{role} — V2 assignment shape.

    conn_id is an existing connection id or "@env" (follow .env defaults).
    model is the model served by that connection for this role.
    """

    conn_id: str = Field(..., description="connection id or '@env'")
    model: str = ""
    extra: dict | None = None


class LLMConfigTestRequest(BaseModel):
    """Body for POST /admin/llm-config/test — full trial config, nothing persisted."""

    provider: str
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    is_vllm: bool = False
    thinking_level: str = ""
    # Service kind — non-LLM services (embed_rerank/stt) are probed via /health
    # instead of the OpenAI /models + chat-completions fallback.
    kind: str = ""


class ModelsListRequest(BaseModel):
    """Body for POST /admin/llm-config/models.

    Either ``conn_id`` (resolve provider/base_url/api_key from the stored
    connection) or a raw {provider, base_url, api_key} triple.
    """

    conn_id: str = ""
    provider: str = ""
    base_url: str = ""
    api_key: str = ""


class ConnectionDeleteResponse(BaseModel):
    """DELETE /admin/llm-config/connections/{conn_id} response."""

    deleted: bool
    referencing_roles: list[str] = []
