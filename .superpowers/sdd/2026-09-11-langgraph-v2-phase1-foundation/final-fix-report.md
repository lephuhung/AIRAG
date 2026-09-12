# Phase 1 final whole-branch fix wave (Tasks 4–11)

Base: `c24f056` on `feat/langgraph-v2`. All findings below were fixed with TDD
(RED then GREEN). The commit is a NEW commit (no amend).

## I1 — lightrag KG lags the revision interface

**Changed**
- `backend/app/services/kg/knowledge_graph_service.py:55` — added
  `_warn_revision_kwargs_ignored(operation)` + module-level
  `_revision_kwargs_warned` (one-time per operation kind).
- `:294` `ingest(markdown_content, document_id=None, revision_id=None)` accepts
  and ignores `revision_id` (warns once).
- `:792` `get_relevant_context(..., revision_ids=None)` accepts and ignores
  `revision_ids` (warns once). `deep_retriever._kg_query` no longer raises
  `TypeError`.
- `:499` `delete_revision_artifacts(self, document_id, revision_id) -> int` —
  best-effort no-op returning `0` with a logged warning, so revision GC does
  not raise. Class/module docstrings document lightrag as **unsupported** for v2
  revision-scoped KG isolation (use `HRAG_KG_MODE=legal`).

**Covering test**: `backend/tests/workers/test_lightrag_kg_revision_interface.py`
(skips in the bench venv — numpy; runs in the full-dependency container).
- `test_lightrag_ingest_accepts_and_ignores_revision_id`
- `test_lightrag_query_accepts_and_ignores_revision_ids`
- `test_lightrag_delete_revision_artifacts_is_a_zero_noop`

**Command**: `docker run ... airag-backend ... pytest tests/workers/test_lightrag_kg_revision_interface.py -q`
→ RED: 3 errors (`_revision_kwargs_warned` / kwargs / method missing); GREEN:
3 passed.

## I2 — caption worker wrote the enriched markdown to the legacy key

**Changed**: `backend/app/workers/caption_worker.py:191-204`. Downloads and
re-uploads `document_views.revision_markdown_key(workspace_id, document_id,
revision_id)` (deterministic, matches what `parse_worker` uploaded); passes
`key=revision_md_key` to `upload_markdown`; the wrong "key unchanged" comment is
corrected. The legacy `kb_{ws}/doc_{doc}.md` object is never touched.

**Covering test**: `backend/tests/workers/test_caption_revision_markdown.py::
test_caption_worker_writes_back_to_revision_markdown_key` (full-dependency
container; aioboto3). Asserts download + upload keys are exactly the revision
key and the legacy key is absent.

**Command**: container run above → RED: FAILED (uploaded `None`/legacy key);
GREEN: 1 passed.

## I3 — finalize mirror lacked a current-revision/generation guard

**Changed**
- `backend/app/workers/utils.py:448` `apply_finalize_outcome(..., *,
  revision_id=None)`; `:476` skips the FAILED/INDEXED mirror when
  `documents.current_revision_id` names a different revision.
- `backend/app/workers/utils.py:621` and `backend/app/workers/parse_worker.py:400`
  pass `revision_id`.

**Covering test**: `backend/tests/workers/test_revision_pipeline.py::
test_stale_revision_failure_does_not_fail_a_live_document` — a newer revision
publishes (document INDEXED on it); a stale revision's late verify failure
leaves the document INDEXED, never FAILED.

**Command**: `PYTHONPATH=. .venv-v2-benchmark/bin/python -m pytest
"tests/workers/test_revision_pipeline.py::test_stale_revision_failure_does_not_fail_a_live_document" -q`
→ RED: FAILED (`apply_finalize_outcome() got an unexpected keyword argument
'revision_id'`); GREEN: passed.

## I4 — KG `revision_facts` null-description

**Changed**: `backend/app/services/kg/legal_kg_service.py:483` and `:504-505` —
`_revision_facts_init` and `_revision_facts_append` wrap the description with
`coalesce($description, '')` / `coalesce($desc, '')` (condition + array element),
so a `None` description can never yield a null property-array element. Docstrings
updated.

**Covering tests** (full-dependency container):
- `tests/agents/v2/persistence/test_revision_live_callers.py::
  test_revision_kg_fact_cypher_coalesces_null_description` (new).
- Updated `test_revision_kg_writes_store_fact_text_per_revision` and
  `test_revision_kg_write_cypher_uses_only_primitive_property_values` to assert
  the coalesced form; existing live Neo4j round-trip still passes.

**Command**: container run above → RED: 3 FAILED; GREEN: passed (incl. live
Neo4j, 0 skipped — confirms real execution).

## M1 — retention lease sweep had no production caller

**Changed**: `backend/app/workers/evidence_gc_worker.py:63` added
`sweep_expired_leases(session)`; `run_both_batches` runs it in its **own**
transaction after Predicate A/B (`:127`), reports `GcRunSummary.leases_released`,
and isolates its failures (`leases: ...`).

**Covering tests**: `tests/workers/test_evidence_gc_worker.py::
test_worker_sweeps_expired_leases_in_a_separate_transaction` and
`::test_lease_sweep_failure_does_not_abort_the_artifact_batches`.

**Command**: bench venv → RED: 2 FAILED (`unexpected keyword 'lease_sweeper'`);
GREEN: passed.

## M2 — agent tool omitted workspace_id in target resolution

**Changed**: `backend/app/services/agent/tools.py:1432-1444`
(`search_document_section`) resolves `resolve_document_targets` once per
requested workspace with `workspace_id=ws_uuid`, keyed by workspace, so a
document outside the workspace can never bind a vector store/revision (same
tenancy check as `api/rag.py`).

**Covering test**:
`backend/tests/services/test_agent_tools_workspace_scope.py::
test_search_document_section_resolves_targets_per_workspace` (full-dependency
container; chromadb).

**Command**: container run → RED: FAILED (`workspace_id` never passed); GREEN:
1 passed.

## M3 — reindex/process endpoints allocated drafts on tombstoned documents

**Changed**: `backend/app/api/rag.py:69` `_reject_tombstoned_document` helper
(typed `409 DOCUMENT_TOMBSTONED`); called in `process_document` (`:308`) and
`reindex_document` (`:502`). `reindex_workspace` filters
`Document.source_deleted_at.is_(None)` (`:587`) and the background `_reindex_all`
skips a document tombstoned after the endpoint's query.

**Covering tests** (`tests/api/test_revision_reindex_delete.py`):
- `test_process_and_reindex_reject_tombstoned_document`
- `test_reindex_workspace_skips_tombstoned_documents`

**Command**: R15 container `pytest tests/api/test_revision_reindex_delete.py
tests/api/test_query_revision_selection.py -q` → RED: 2 FAILED; GREEN: 13 passed.

## Parked (not fixed, per controller ruling)

C1, M4, M5, M6, M7, M8, and Task-10 checkpoint provisioning (ops).

## Full-suite evidence

- Bench venv: `cd backend && PYTHONPATH=. .venv-v2-benchmark/bin/python -m pytest
  tests/migrations/v2 tests/agents/v2 tests/workers -q` → **582 passed, 4
  skipped** (baseline 579 passed, 2 skipped; +3 passed, +2 skipped are the new
  numpy/aioboto3-gated modules).
- R15 container: `pytest tests/api/test_revision_reindex_delete.py
  tests/api/test_query_revision_selection.py -q` → **13 passed**.
- R15 container (new + KG live):
  `pytest tests/workers/test_lightrag_kg_revision_interface.py
  tests/workers/test_caption_revision_markdown.py
  tests/services/test_agent_tools_workspace_scope.py <two M3 tests>
  tests/agents/v2/persistence/test_revision_live_callers.py -q` →
  **29 passed** (0 skipped; live Neo4j executed).

`backend/requirements-v2-benchmark.txt` intentionally left uncommitted.
