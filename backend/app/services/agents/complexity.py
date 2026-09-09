"""
Complexity Router contracts (Section A.2).

Per spec Section A.2: RoutingDecision with validators.
All contracts use ConfigDict(extra="forbid", frozen=True).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from app.services.agents.semantic_preprocessor import PreprocessingResult


# =============================================================================
# Contract: RoutingDecision (A.2)
# =============================================================================

class RoutingDecision(BaseModel):
    """Decision from the complexity classifier.

    Per A.2: execution_mode ∈ {supervisor, deepagent, clarify}.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_mode: Literal["supervisor", "deepagent", "clarify"]
    work_type: Literal["lookup", "compare", "summarize", "multi_goal", "cross_agent", "other"]
    needs_document_probe: bool = False
    reason_code: Literal[
        "single_workflow", "inline_content", "multi_target_compare",
        "multi_goal", "cross_agent_dependency", "dependent_research",
        "long_document", "summary_size_unknown", "missing_reference",
    ]
    clarification_question: str | None = None

    @model_validator(mode="after")
    def _clarify_invariants(self):
        if self.execution_mode == "clarify":
            if self.reason_code != "missing_reference":
                raise ValueError("clarify mode requires reason_code=missing_reference")
            if not self.clarification_question:
                raise ValueError("clarify mode requires non-empty clarification_question")
            if self.needs_document_probe:
                raise ValueError("clarify mode cannot have needs_document_probe=true")
        else:
            if self.clarification_question is not None:
                raise ValueError(f"{self.execution_mode} mode requires null clarification_question")
        return self

    @model_validator(mode="after")
    def _probe_invariants(self):
        if self.needs_document_probe:
            if self.execution_mode != "supervisor":
                raise ValueError("needs_document_probe only valid when execution_mode=supervisor")
            if self.work_type != "summarize":
                raise ValueError("needs_document_probe only valid for summarize")
            if self.reason_code != "summary_size_unknown":
                raise ValueError("needs_document_probe requires reason_code=summary_size_unknown")
        return self


# =============================================================================
# FallbackReason enum
# =============================================================================

class FallbackReason:
    """Why the routing decision fell back to legacy fields."""
    NONE = "none"
    LLM_FAILED = "llm_failed"
    LLM_INVALID_JSON = "llm_invalid_json"
    TRUSTED_FACT = "trusted_fact"  # rule-based fallback
    SAFETY_OVERRIDE = "safety_override"


# =============================================================================
# build_routing_decision stub (C.4 full impl deferred to Phase 1B)
# =============================================================================

def build_routing_decision(
    llm_output: dict | None,
    semantic_context: "PreprocessingResult",
    runtime_hints: "RuntimeHints",
) -> tuple[RoutingDecision, FallbackReason]:
    """Centralized parse + validate + trusted-fact fallback.

    Per C.4: 10-rule fallback table with cross_domain threshold = >=2 refs.
    Implemented in Phase 1B; Phase 1A returns safe default.
    """
    # Phase 1A: no-op stub — supervisor_node uses legacy routing
    raise NotImplementedError("build_routing_decision implemented in Phase 1B")


# Stub types to avoid circular import at module level
class RuntimeHints(BaseModel):
    """Runtime hints derived from semantic_context for routing."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary_execution: Literal["single_pass", "needs_map_reduce", "unknown", "not_applicable"] = "not_applicable"
    inline_content_sufficient: bool | None = None
    cross_domain: bool = False
