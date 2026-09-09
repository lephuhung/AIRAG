"""
Deep Research contracts (Section A.3 — A.6).

Per spec Section A: TaskSpec, TaskResult, Coverage, Evidence, Provenance,
RuntimeContext, ModelSnapshot, ToolBudget, ConsumedBudget, PreprocessorBudgetConfig.
All contracts use ConfigDict(extra="forbid", frozen=True).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# A.3 Contract: TaskSpec
# =============================================================================

class TaskSpec(BaseModel):
    """Deep Agent task specification from planner output."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    target_ref: str | None = None  # ref_id from semantic_context
    target_handle: uuid.UUID | None = None  # Document.id
    work_type: Literal[
        "retrieve_section", "compare_pair", "cross_agent_lookup",
        "summarize_chunk", "defer_resolution",
    ]
    depends_on: list[str] = []
    allowed_tools: list[str] = []
    scope_constraints: dict = {}
    completion_criteria: dict = {}
    budget_hint: dict | None = None

    @model_validator(mode="after")
    def _check_target_required(self):
        if self.work_type != "defer_resolution":
            if self.target_ref is None and self.target_handle is None:
                raise ValueError(f"task {self.task_id} needs target_ref or target_handle")
        return self


# =============================================================================
# A.4 Contract: TaskResult + Coverage
# =============================================================================

class Coverage(BaseModel):
    """Coverage measurement for a task.

    Per A.4: NOT inferred from bool(sources). Each dimension has explicit
    denominator: requested = task plan targets; resolved = doc_handle resolved;
    read = structurally full range covered; truncated = cutoff by budget.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested: int = 0
    resolved: int = 0
    read: int = 0
    truncated: int = 0


class TaskResult(BaseModel):
    """Result of executing one TaskSpec."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    status: Literal["ok", "partial", "missing", "ambiguous", "error"]
    evidence_ids: list[str] = []
    coverage: Coverage = Coverage()
    missing_requirements: list[str] = []
    artifact_refs: list[str] = []
    error_detail: str | None = None


# =============================================================================
# A.5 Contract: Evidence + Provenance
# =============================================================================

class Provenance(BaseModel):
    """Immutable record of HOW evidence was obtained.

    Per A.5: `acl_checked_at` and `acl_version` fields.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    fetcher: Literal["search_document_section", "search_documents_number",
                     "kg_query", "people_search", "attachment_read", "deep_worker"]
    fetched_at: float  # epoch seconds
    fetched_by: uuid.UUID  # principal_id
    workspace_scope: list[uuid.UUID]
    acl_checked: bool = True
    acl_checked_at: float | None = None  # epoch seconds; recorded when ACL verified
    acl_version: str | None = None  # policy version (e.g. "v1")
    tool_call_id: str | None = None
    run_id: str


class Evidence(BaseModel):
    """Provenance-anchored evidence with byte-safe truncation.

    Per A.5: raw_content may be truncated to MAX_RAW_CONTENT_BYTES.
    raw_content_bytes tracks original uncut size.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str  # "{task_id}:c{N}" or central UUID
    task_id: str
    source_id: str  # server-generated UUID; unique per fetch
    raw_content: str  # possibly truncated
    content_hash: str  # sha256 of raw_content
    content_size_bytes: int  # bytes of raw_content (== len(raw_content.encode('utf-8')))
    raw_content_bytes: int  # bytes of ORIGINAL uncut content (>= content_size_bytes)
    redacted: bool = False
    document_id: uuid.UUID | None = None
    document_version: str | None = None
    workspace_id: uuid.UUID | None = None
    section_path: str | None = None
    page_or_chunk: str | None = None
    chunk_offsets: tuple[int, int] | None = None  # byte offsets in raw_content
    citation_number: str | None = None  # only when matches verified metadata
    citation_article: str | None = None
    provenance: Provenance

    MAX_RAW_CONTENT_BYTES: int = 50_000  # retention cap

    @model_validator(mode="after")
    def _check_citation_anchored(self):
        has_cit = self.citation_number or self.citation_article
        if has_cit:
            if self.document_id is None:
                raise ValueError("citation requires verified document_id")
            if self.chunk_offsets is None:
                raise ValueError("citation requires chunk_offsets")
        return self

    @model_validator(mode="after")
    def _check_size_retention(self):
        if self.content_size_bytes > self.MAX_RAW_CONTENT_BYTES:
            raise ValueError(f"raw_content exceeds retention cap {self.MAX_RAW_CONTENT_BYTES}")
        if self.raw_content_bytes < self.content_size_bytes:
            raise ValueError(
                f"raw_content_bytes ({self.raw_content_bytes}) < content_size_bytes ({self.content_size_bytes})"
            )
        return self


# =============================================================================
# A.6 Contract: RuntimeContext + budget types
# =============================================================================

class ModelSnapshot(BaseModel):
    """Frozen snapshot of model config at request ingress."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    base_url: str | None = None
    config_revision: str
    langfuse_tags: list[str] = []


class ToolBudget(BaseModel):
    """Budget limits for a deep agent run."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_parallel_branches: int = 2
    max_domain_tool_calls: int = 6
    max_coordinator_rounds: int = 4
    max_worker_llm_rounds: int = 2
    synthesis_output_cap_tokens: int = 800
    worker_output_cap_tokens: int = 400


class ConsumedBudget(BaseModel):
    """Atomic counters; mutated ONLY via BudgetGuard (which holds the lock)."""
    model_config = ConfigDict(extra="forbid", frozen=False)  # mutable for atomic increments

    coordinator_rounds: int = 0
    domain_tool_calls: int = 0
    worker_llm_rounds_per_task: dict[str, int] = {}
    tokens_emitted: int = 0
    evidence_emitted: int = 0


class PreprocessorBudgetConfig(BaseModel):
    """Per-stage budget gates for preprocessor."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    absolute_cutoff_offset_sec: float = 25.0  # preprocessor ends at deadline - 25s
    per_call_timeout_sec: float = 2.0
    disambig_reserve_sec: float = 3.0  # remaining > 3s required to START disambig


class RuntimeContext(BaseModel):
    """Request-scoped context for deep agents.

    Per A.6: ONE consistent clock (time.monotonic()) used throughout.
    consumed_budget mutated ONLY via BudgetGuard.try_consume_* methods.
    cancellation_event is coordinator-level shared; branches hold REFERENCE.
    """
    model_config = ConfigDict(extra="forbid", frozen=False)  # mutable for budget counters

    principal_id: uuid.UUID
    allowed_workspace_ids: list[uuid.UUID]
    authorized_document_handles: set[uuid.UUID]
    people_permission: bool
    session_id: str | None
    run_id: str
    config_revision: str
    absolute_deadline_monotonic: float  # time.monotonic() value
    absolute_deadline_epoch: float | None = None  # wall-clock for log correlation ONLY
    remaining_budget_sec: float  # = absolute_deadline_monotonic - time.monotonic()
    model_snapshot: ModelSnapshot
    tool_budget: ToolBudget
    consumed_budget: ConsumedBudget
    preprocessing: PreprocessorBudgetConfig
    tool_allowlist: set[str] = set()
    # NOTE: cancellation_event is set externally; do not store it here
    # as RuntimeContext is never serialized to checkpoint.
