# LangGraph v2 Defect Repair — Design

**Date:** 2026-09-13

**Status:** Approved design (pending spec review)

**Source report:** [`docs/reports/2026-09-13-langgraph-v2-live-test-report.md`](../../reports/2026-09-13-langgraph-v2-live-test-report.md)

**Branch:** `feat/langgraph-v2`

**Scope:** repair the confirmed defects from the authenticated live test, plus the
A/B harness defect and a controlled corpus reindex. No agent-architecture
redesign; all changes stay inside the existing v2 composition.

## 1. Objective

Close the four confirmed defects and one harness defect found by the live test so
that:

1. valid Vietnamese document-number queries no longer fail `PreprocessingResult`
   validation before routing;
2. multi-word Vietnamese law titles can bind to indexed documents;
3. a complex/factual turn with no valid verdict terminates as a typed
   `clarify`/`insufficient` response, never a generic `error` and never a
   fabricated `success`;
4. the admin A/B (`v1`) evaluation arm is usable again;
5. the legacy corpus in the target workspaces is reindexed into the v2 revision
   model so v2 factual routes can read it.

## 2. Verified root causes

All causes below were reproduced against the worktree code during design; see the
source report for the live evidence.

### 2.1 Overlapping document-reference spans (defect #1)

`extract_document_references()` (`semantic_preprocessor.py`) runs six independent
regexes and appends every match with no dedup/precedence. For
`53/2022/NĐ-CP …`:

```text
regex_doc_num        (0, 13) '53/2022/NĐ-CP'
regex_short_official (3, 13) '2022/NĐ-CP'      <- nested/overlapping
regex_abbr_then_doc  (8, 10) 'NĐ', (11,13) 'CP'
```

`PreprocessingResult._check_ref_spans_non_overlapping` then raises
`overlapping ref spans: (0, 13) and (3, 13)`, so `preprocess_query()` raises
before routing.

### 2.2 Incomplete multi-word title grammar (defect #2)

`_RE_NAMED_DOC` captures only `\S+` after the document-type keyword, so
`Luật An ninh mạng` yields `Luật An ` and `Luật Bảo vệ dữ liệu cá nhân` yields
`Luật Bảo `.

### 2.3 Generic error on a missing verdict (defect #3)

In the complex subgraph, `decide_node` emits a typed `ComplexResearchUnavailable`
marker when `plan is None`. `merge_complex_result_into_supervisor` only carries
`plan`/`task_results`/`evaluation`, so the marker is dropped. `_complex_branch`
then sends the turn to `finalizer`; `_finalize_factual` sees
`evidence_evaluation is None` and raises `FinalizerError`, which the node wrapper
converts to a generic `error` response.

### 2.4 `ChatSourceChunk` contract mismatch in the v1 harness (harness defect)

`streaming.py` passes a `ChatSourceChunk` object into
`SourcesSnapshotAccumulator.add()`, which reads `.chunk`/`.content_hash`/
`.source_id`/`.doc`. `ChatSourceChunk` has none of those fields → `AttributeError`
on the `v1` admin-evaluation arm.

### 2.5 Legacy corpus has no v2 revision identity (defect #4)

`document_views.load_current_revision_identity()` returns `None` when
`Document.current_revision_id IS NULL`. The v2 adapter correctly fails closed, so
`document.search` yields no candidate for an otherwise searchable document. This
is a data/operating condition, not a logic bug: it is resolved by the existing
revision-aware reindex, not by weakening the pinning guard.

## 3. Non-goals / out of scope

- No change to the v2 pinning contract. A document without a published revision
  must keep terminating as typed unavailable/insufficient.
- No v1 fallback for factual traffic (deployment is v2-only).
- No expansion of `_RE_SECTION`. Adapter `section_refs`/`person_refs` remain
  empty; People/Section reachability stays deferred.
- No unrelated refactor of the six-pattern extraction design beyond adding the
  arbitration stage.
- No `ResponseStatus` contract change: `clarify` and `insufficient` already
  exist.

## 4. Design

### 4.1 Extraction arbitration (defects #1 + #2) — Approach 3

The six regexes remain *candidate generators*. A new pure function
`_arbitrate_ref_candidates(candidates) -> list[RefExtraction]` runs at the end of
`extract_document_references()` and guarantees pairwise non-overlapping spans by
construction.

**Specificity rank (high → low):**

```text
regex_doc_num > regex_short_official > regex_section
              > regex_named_doc > regex_bare_number > regex_abbr_then_doc
```

**Selection rule:** scan candidates left→right; keep a candidate unless it
overlaps an already-kept candidate of equal or higher rank. Tie-break: longer
span, then higher rank, then earlier start. A candidate fully contained in a
higher-ranked one is dropped. Abbreviations are not part of `document_refs`
nesting (the "nested spans" allowance applies to `abbreviations`, not
`document_refs`).

**Greedy multi-word title** for `_RE_NAMED_DOC`: after the document-type keyword,
capture up to `N` tokens (default 8), stopping at a stop-token
(`và, với, của, cho, về, trong, tại, theo, gồm, hay, hoặc, mà, để, khi, nếu`) or
punctuation (`, . ; : ? ! " ( ) [ ]`). The optional trailing document-number group
is preserved.

**Title resolution (DB longest-match):** new resolver `_lookup_by_title()` called
from `_stage5_metadata_lookup` when `parse_basis == "regex_named_doc"`. It filters
by `ctx.allowed_workspace_ids` and searches `DocumentAlias` + `Document.document_title`,
choosing the longest matching title. If nothing matches, it falls back to the
arbitrated candidate (status `not_found`/`deferred`) rather than dropping the ref.

**Data flow:** raw → NFC → normalized(lower) → candidates → arbitration → title
resolution → `DocumentRefEntry[]` → `PreprocessingResult` (valid) → adapter →
`SemanticContext.document_refs`.

### 4.2 Typed terminal outcome (defect #3)

**Contract addition.** Add one nullable slot to the checkpointed execution
aggregate so the subgraph's typed reason survives the merge:

```text
Field: research_unavailable
Authoritative owner: complex-research subgraph (decide_node)
Produced by: decide_node when plan is None
Consumed by: finalizer (_finalize_factual) via the merged ExecutionState
Persisted? yes (inside ExecutionState checkpoint)
Derivable? no — the policy decision is made inside the subgraph and is not a
           function of the parent's checkpointed fields
Reason it must exist: without it the typed unavailable reason is lost at the
           merge boundary and the terminal outcome degrades to a generic error
```

`execution_update()` and `reset_execution()` handle the slot with the existing
None-means-keep / clear semantics; `merge_complex_result_into_supervisor()`
passes `child.get("unavailable")`.

**Finalizer rule.** In `_finalize_factual`, when `evaluation is None`, do not
raise. Emit a typed non-success response using the semantic signal (chosen
criterion A):

- `semantic.document_refs` contains any `resolution_status ∈ {unresolved, ambiguous}`
  **or** `semantic.blocking_ambiguities` is non-empty → `FinalResponse(status="clarify")`
  with `_NEEDS_INPUT_CONTENT` plus the unresolved/ambiguous spans (user text, safe).
- otherwise → `FinalResponse(status="insufficient")` with `_INSUFFICIENT_CONTENT`.

`research_unavailable`, when present, is a secondary signal only: it still maps to
`insufficient`; its `reason` (work type / policy) is never surfaced.

**Invariants preserved:** never emit `success` without a verdict; `_require_route`
and the persisted-clarification requirement for the `clarify` route remain
raises. Only the `evaluation is None` raise inside `_finalize_factual` is
replaced.

### 4.3 Harness fix (harness defect)

Add one owner for the conversion: classmethod
`Source.from_chat_source_chunk(chunk)` on `SourcesSnapshotAccumulator`'s `Source`:

```text
doc          <- chunk.source_file or chunk.document_number or str(chunk.document_id)
chunk        <- chunk.content
content_hash <- chunk.chunk_id, else a stable hash of chunk.content
source_id    <- chunk.index
```

`streaming.py` uses it for the object branch (replacing the bare
`sources_acc.add([s])`); the dict branch and objects already of type `Source` are
unchanged.

### 4.4 Corpus reindex (defect #4)

Use the existing revision-aware reindex; do not build a new path.

**Pre-check (read-only):** confirm `v2_schema_version` applied and workers/queue
healthy; list documents in the target workspaces with
`current_revision_id IS NULL AND source_deleted_at IS NULL`.

**Execute:** `POST /reindex-workspace/{workspace_id}` with a superadmin JWT.
`allocate_reindex_revision` allocates a new generation (copy-on-write); workers
build artifacts; `verify_draft` + `publish` succeed before
`Document.current_revision_id` is set. Legacy collections/vectors are never
deleted (guarded by existing tests).

**Verify:**

- DB: `current_revision_id IS NOT NULL`, `DocumentRevision.status='published'`,
  `DocumentRevisionBuild` carries the required artifacts + embedding manifest.
- Boundary trace: `load_current_revision_identity()` returns an identity;
  `resolve_retrieval_revisions()` no longer raises `RevisionNotReady`.
- Authenticated SSE: a factual query in `Luật` reaches answer (or typed
  `insufficient`), never a generic error.

**Abort:** on verification failure, `abandon_revision` for
draft/building/verified revisions so the previous pointer stays. Never set
`current_revision_id` by hand; GC owns reclamation.

**Deliverable:** `docs/runbooks/langgraph-v2-corpus-reindex.md`.

**Environment constraint:** execute only against the `hrag-*` test stack; never
restart the vLLM engines.

## 5. Error handling

- Arbitration must be total: any candidate set (including all-overlapping)
  resolves to a valid, non-overlapping list; no exception escapes extraction.
- Title resolution failures are non-fatal and degrade to `not_found`/`deferred`.
- Finalizer converts a missing verdict to a typed response; it never raises for
  that reason and never emits `success`.
- Reindex failures leave the document on its previous revision (fail-closed);
  the error is logged and surfaced, not partially published.

## 6. Testing

TDD: failing tests first, then implementation.

1. **Arbitration unit** (`tests/agents/test_semantic_preprocessor_extraction.py`):
   `53/2022/NĐ-CP …` → exactly one ref, span `(0,13)`, basis `doc_num`,
   `PreprocessingResult` constructs; `NĐ là gì?` → one ref `(0,2)`; multi-word
   compare → two full-title refs, non-overlapping; property test: arbitration
   output is pairwise non-overlapping over a sample query corpus.
2. **Title resolution:** fake session returning aliases/titles — longest-match,
   ACL filter, fallback when no DB match.
3. **Finalizer/#3:** `evaluation is None` + (a) unresolved ref or blocking
   ambiguity → `clarify`; (b) no refs → `insufficient`; (c) `research_unavailable`
   → `insufficient`; never `success`. Merge test: subgraph `unavailable` survives
   into `ExecutionState`; checkpoint round-trip with and without the new slot.
4. **Harness:** a `ChatSourceChunk` through the sources path raises no
   `AttributeError`; dedup identity is correct.
5. **Regression:** existing `test_semantic_preprocessor_db_resolution.py` and the
   `tests/agents` suite stay green.

## 7. Acceptance criteria

- No valid Vietnamese document-number query raises `PreprocessingResult`
  validation.
- Multi-word law titles bind when the data is indexed.
- Complex/factual without a verdict → typed `clarify`/`insufficient`; never a
  generic error, never `success`.
- A/B admin arm `v1` returns HTTP 200.
- After reindex, `document.search` (v2) admits the `Luật` documents and an
  authenticated factual SSE returns an answer or typed `insufficient`.
- New + focused tests pass; `detect_changes` (or CLI equivalent) reports only the
  expected symbols and flows.

## 8. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| New `ExecutionState` slot breaks checkpoint serde / old checkpoints | Nullable default `None`; checkpoint round-trip tests with and without the slot; `normalize_checkpoint_state` treats a missing slot as `None`. |
| Arbitration changes existing resolved-ref behavior | Broad unit tests with real Vietnamese queries; assert current `test_semantic_preprocessor_db_resolution.py` behavior is preserved where intended. |
| Multi-word stop-token set over/under-captures | DB longest-match + candidate fallback; bounded token window. |
| Reindex mutates the test corpus | Small workspace (`Luật`) first, verify, then `Nghị định`; abort path; never set the pointer manually. |
| Removing the finalizer raise weakens fail-closed behavior | Only the `evaluation is None` raise is replaced; all other raises and the never-`success`-without-verdict invariant remain; covered by tests. |

## 9. Rollout / rollback

- All code changes are on `feat/langgraph-v2`; each defect is an independently
  revertible change.
- Reindex is data-only and reversible by revision state (previous pointer is
  untouched until a new revision publishes; `abandon_revision` on failure).
- No vLLM restart; backend/worker code changes require only the normal
  backend/worker image rebuild for the test stack.

## 10. Deferred (recorded, not in this scope)

- `person_refs`/`section_refs` extraction and People/Section capability
  reachability.
- Default `available_services` exposing only `document.search` + `people.lookup`
  (deployment gating of KG/memory/read/abbreviation).
- `people.lookup` 5s-schema timeouts / indexing risk noted in the report.
