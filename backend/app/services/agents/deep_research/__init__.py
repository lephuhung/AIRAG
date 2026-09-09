"""
Deep Research module (Phase 2 pilot: compare_sections only).

Per spec Section D.2:
- graph.py: create_deep_research_graph()
- contracts.py: re-export from Section A
- tools.py: RetrieveSectionTool
- budget.py: deadline + budget + cancellation
- evidence.py: EvidenceRegistry + citation
"""

from app.services.agents.deep_research.contracts import (
    TaskSpec,
    TaskResult,
    Evidence,
    Coverage,
    Provenance,
    RuntimeContext,
    ModelSnapshot,
    ToolBudget,
    ConsumedBudget,
    PreprocessorBudgetConfig,
)

__all__ = [
    "TaskSpec",
    "TaskResult",
    "Evidence",
    "Coverage",
    "Provenance",
    "RuntimeContext",
    "ModelSnapshot",
    "ToolBudget",
    "ConsumedBudget",
    "PreprocessorBudgetConfig",
]
