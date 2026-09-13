"""Task-strategy skills for v2 (amendment §4: framework-neutral policy).

A skill owns no authorization, persistence, TaskPlan, EvidenceUse, or
FinalResponse boundary: it encodes task know-how (which bounded reads a
comparison needs) that the one adaptive complex planner turns into a typed
``TaskPlan`` proposal. Native framework skill files, if Phase 0 selects them,
mirror ``skills/<name>/policy.py``; the Python policy stays the source of
truth. ``legal_analysis`` and ``compliance`` are NOT created here: they belong
to a future approved plan only.
"""
