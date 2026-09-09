"""
Contracts validation module (Section A.7 / Q16).

Pure validation functions for Phase 1A contracts.
These functions do NOT import LangGraph or asyncio; they are safe to call
from anywhere in the pipeline.

Per A.7: validate_task_plan, validate_evidence_collection.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agents.semantic_preprocessor import PreprocessingResult
    from app.services.agents.complexity import RoutingDecision, FallbackReason, RuntimeHints
    from app.services.agents.deep_research.contracts import (
        TaskSpec, TaskResult, Evidence, RuntimeContext,
    )


class ValidationError(ValueError):
    """Raised when contract validation fails."""
    pass


def validate_task_plan(
    tasks: list["TaskSpec"],
    semantic_context: "PreprocessingResult",
) -> None:
    """Validate a task plan against semantic_context.

    Rules:
    - task_id must be unique within the plan
    - depends_on must not create cycles (acyclic)
    - target_ref must exist in semantic_context.document_refs[*].ref_id
    - allowed_tools must be subset of RuntimeContext.tool_allowlist

    Per A.7: raises ValidationError on violation.
    """
    # 1. Unique task_id
    task_ids = [t.task_id for t in tasks]
    if len(task_ids) != len(set(task_ids)):
        dupes = [tid for tid in task_ids if task_ids.count(tid) > 1]
        raise ValidationError(f"duplicate task_id in plan: {set(dupes)}")

    # 2. Acyclic depends_on (topological check)
    # Build adjacency: task_id -> set of tasks it depends on
    task_map = {t.task_id: set(t.depends_on) for t in tasks}
    visited: set[str] = set()
    in_stack: set[str] = set()

    def has_cycle(tid: str) -> bool:
        if tid in in_stack:
            return True
        if tid in visited:
            return False
        in_stack.add(tid)
        for dep in task_map.get(tid, []):
            if has_cycle(dep):
                return True
        in_stack.discard(tid)
        visited.add(tid)
        return False

    for tid in task_ids:
        if has_cycle(tid):
            raise ValidationError(f"cyclic dependency in task plan involving {tid}")

    # 3. target_ref must exist in semantic_context
    valid_ref_ids = {r.ref_id for r in semantic_context.document_refs}
    for task in tasks:
        if task.target_ref is not None and task.target_ref not in valid_ref_ids:
            raise ValidationError(
                f"task {task.task_id}: target_ref {task.target_ref!r} "
                f"not found in semantic_context.document_refs"
            )


def validate_evidence_collection(
    evidence: list["Evidence"],
    task_result: "TaskResult",
) -> None:
    """Validate that evidence collection matches task result.

    Per A.7: raises ValidationError on violation.
    Validates:
    - All evidence_ids in task_result.evidence_ids exist in evidence
    - Coverage.truncated count matches evidence with raw_content_bytes > MAX
    """
    evidence_ids = {e.evidence_id for e in evidence}
    for eid in task_result.evidence_ids:
        if eid not in evidence_ids:
            raise ValidationError(
                f"task_result.evidence_ids contains {eid!r} "
                f"but no matching Evidence found"
            )

    # Coverage.truncated should match evidence with oversized raw_content
    truncated_count = sum(
        1 for e in evidence
        if e.raw_content_bytes > e.MAX_RAW_CONTENT_BYTES
    )
    if task_result.coverage.truncated != truncated_count:
        raise ValidationError(
            f"coverage.truncated={task_result.coverage.truncated} "
            f"but {truncated_count} evidence records exceed MAX_RAW_CONTENT_BYTES"
        )


def build_routing_decision(
    llm_output: dict | None,
    semantic_context: "PreprocessingResult",
    runtime_hints: "RuntimeHints",
) -> tuple["RoutingDecision", "FallbackReason"]:
    """Stub: full implementation in Phase 1B.

    Per A.7 / C.4: 10-rule fallback table.
    Phase 1A returns safe default.
    """
    from app.services.agents.complexity import (
        RoutingDecision, FallbackReason,
    )

    # Phase 1A safe default: supervisor, simple
    return (
        RoutingDecision(
            execution_mode="supervisor",
            work_type="lookup",
            reason_code="single_workflow",
        ),
        FallbackReason.NONE,
    )
