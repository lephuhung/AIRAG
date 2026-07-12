"""
Supervisor Agent Prompts — backward-compat shim
================================================

The supervisor system prompt is now modular in
:mod:`app.prompts.agents.supervisor_scope`. This module keeps
``_SUPERVISOR_PROMPT`` exported so existing callers (notably
``tests/prompts/test_supervisor_routing.py`` and any code that still imports
the constant) keep working unchanged.

``_SUPERVISOR_PROMPT`` is the unrendered TEMPLATE for the "full" scope — it
still has ``{max_iterations}`` / ``{analyzer_context}`` placeholders so legacy
``prompt.format(max_iterations=N, analyzer_context="...")`` calls work.

New code SHOULD import from :mod:`app.prompts.agents.supervisor_scope`:

    from app.prompts.agents.supervisor_scope import (
        build_supervisor_system_prompt,
        classify_supervisor_scope,
    )

and call::

    scope = classify_supervisor_scope(query_for_classifier, has_doc_ids=bool(state.get("document_ids")))
    system = build_supervisor_system_prompt(scope, max_iterations=max_iter, analyzer_context=_analyzer_context)
"""

from __future__ import annotations

from app.prompts.agents.supervisor_scope import (
    _SCOPE_SECTIONS,
    build_supervisor_system_prompt,
    classify_supervisor_scope,
)

_SUPERVISOR_PROMPT = "\n\n".join(_SCOPE_SECTIONS["full"])

__all__ = [
    "_SUPERVISOR_PROMPT",
    "build_supervisor_system_prompt",
    "classify_supervisor_scope",
    "_SCOPE_SECTIONS",
]
