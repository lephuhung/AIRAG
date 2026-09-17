# Complex Research Discovery Bootstrap Design

**Date:** 2026-09-17  
**Status:** Approved design, revision 2 after code review
**Scope:** LangGraph v2 complex-research path (`summarize` and `compare` first)

## 1. Problem

The v2 complex-research planner currently requires document bindings before it can produce useful plans:

- `summarize` requires exactly one bound `target` document;
- `compare` requires exactly two bound documents, one `target` and one `reference`;
- `document.search` runs only after a plan exists;
- discovered documents can currently become only `supporting` or `discovered` bindings, not intent-selected research targets.

Consequently, a complex request without usable `document_ids` or resolved document references fails before the system can search for the documents needed to plan the request. Explicitly attached or quoted documents also act too much like hard target scope: a user can quote document A while asking about document B, but the graph cannot reliably assign A as context and discover B as the actual target.

## 2. Goals

1. Search the authenticated workspace before final research planning for supported complex requests.
2. Derive document roles from query meaning rather than treating every explicit document as a target.
3. Auto-select documents only when confidence is sufficient; otherwise suspend and ask the user to choose.
4. Preserve the execution invariant:

   ```text
   validated + checkpointed TaskPlan → shared TaskScheduler → capability
   ```

5. Preserve ACL, exact-revision pinning, retention leases, restart safety, cancellation, and model-facing privacy boundaries.
6. Support up to five search probes across at most two discovery rounds, with a 15-second discovery deadline.
7. Keep v1 and unrelated v2 fast paths unchanged. When the feature flag is on, bounded `summarize` intentionally leaves the fast path and enters complex discovery; this routing change is required for intent-derived roles.

## 3. Non-goals

- The planner will not call a database, vector index, capability, or legacy agent directly.
- The feature will not replace v1 or globally enable v2.
- Discovery will not run without hard budgets.
- The model will not mint document IDs, revisions, binding IDs, candidate IDs, roles, or authorization.
- A low-confidence candidate will not be silently promoted.
- The feature will not expand authenticated workspace scope.
- Write requests and unsupported routes remain on their existing fallback paths.

## 4. Chosen Architecture

Add a governed discovery-bootstrap phase inside the existing complex-research boundary. The router receives the server-owned feature snapshot as a pure-policy input. With the flag on, every `summarize` route enters `complex_research` before the current `bound_count == 1` fast-summary branch; `compare` already always enters `complex_research`. With the flag off, current routing is byte-compatible.

```text
route(flag ON + summarize/compare → complex)
  → target_slots
  → discovery_propose
  → discovery_validate_checkpoint
  → execute bootstrap search plan through shared TaskScheduler
  → rank_select
      ├─ missing coverage and budget remains → discovery_replan
      ├─ ambiguous → interrupt(document choices) → resume
      └─ confident → selection_settle (ACL recheck + exact pin)
  → research_expand
  → validate_plan_expansion / lease / checkpoint
  → execute research tasks
  → existing discovery_settle for non-bootstrap search tasks
  → evaluate
  → reduce
  → decide
  → synthesize or finalize
```

Discovery and final research planning remain in one complex-research subgraph and one append-only plan lineage. The first checkpointed plan contains only governed `document.search` bootstrap tasks. After selection, a new governed expansion operation appends target units and factual read tasks while retaining the search tasks and their result lineage. The graph never overwrites or silently replaces the accepted bootstrap plan.

This is preferred over letting the model call search directly, which would combine advisory planning with execution and weaken checkpoint/ACL ownership.

### 4.1 Composition with targetless retrieval fallback

The existing search-first fallback and bootstrap have distinct ownership:

- **Flag off:** the current `retrieve-unscoped` fallback remains unchanged for a covering skill refusal, including `summarize`, `compare`, `evaluate`, `multi_goal`, and `cross_domain`.
- **Flag on:** bootstrap replaces that fallback only for `summarize` and `compare`. A bootstrap timeout, ambiguity, or validation failure follows bootstrap clarification/typed-unavailable behavior; it does not fall through to targetless retrieval and bypass target selection.
- **Other work types:** `evaluate`, `multi_goal`, and `cross_domain` continue to use the existing targetless retrieval fallback when their owning skill refuses.
- **Retrieve:** ordinary scoped and unscoped `retrieve` policy remains unchanged.

Targetless retrieval remains valid for grounded workspace evidence. It is rejected only as a substitute for document-role selection in flag-enabled `summarize` and `compare`.

## 5. Ownership Rules

### 5.1 Discovery proposer

The proposer may suggest bounded search probes and objectives. It cannot execute them or choose raw document identity.

### 5.2 Validator and checkpoint owner

A single governed node validates search probes, builds the bootstrap `TaskPlan`, acquires any required leases, and returns the plan into checkpointed state before dispatch.

### 5.3 Shared scheduler

All `document.search` and subsequent factual tasks execute through the existing shared `TaskScheduler`. No discovery-specific scheduler is introduced.

### 5.4 Candidate selector

A deterministic server policy ranks candidates against target slots and applies confidence, margin, duplicate, and role rules. The model does not have selection authority.

### 5.5 Binding resolver

The binding resolver revalidates current ACL and workspace scope, pins the exact selected revision, writes binding provenance, and acquires a retention lease before the new binding becomes checkpoint-visible.

### 5.6 User

The user has final authority when automatic selection is ambiguous. Resume input can select only an opaque choice recorded in the suspended checkpoint.

## 6. Intent-derived Document Roles

Explicit or quoted documents are inputs to role resolution, not unconditional hard targets.

Examples:

| Request | Resulting roles |
|---|---|
| Quote A + “tóm tắt tài liệu này” | A is the primary `target`; confidently selected related documents that will be read are `reference` bindings. |
| Quote A + “tóm tắt Nghị định 13” | The confidently resolved Nghị định 13 is `target`; A stays bound but is not placed in a `TargetUnit` unless the query also asks to use A as context. |
| Quote A + “so sánh tài liệu này với Nghị định 13” | A fills one comparison side; Nghị định 13 is discovered for the other side. |
| “Tổng hợp các quy định về bảo vệ dữ liệu” | Multiple confident documents become summary targets. |

Under the feature flag, supported `summarize` and `compare` routes always perform discovery, even when explicit documents appear sufficient. The router therefore sends `summarize` to `complex_research` before applying the existing one-bound-document fast branch; `compare` needs no routing override because it is already complex. This permits the graph to detect query-named documents different from the quoted context. A discovery dependency failure is a typed unavailable result; it does not silently revert to the pre-feature interpretation.

## 7. Discovery Contracts

Create a focused `agents/v2/discovery/` package.

### 7.1 `TargetSlot`

A checkpointed requirement for a document role:

- `slot_id`
- `intended_role`: `target` or `reference`
- `subject_hint`
- `required`
- `explicit_binding_ids`
- `source`: semantic reference, explicit resource, or inferred research need

The server constructs slots from finalized semantics and route analysis. The model cannot add a role outside these slots.

### 7.2 `SearchProbe`

A checkpointable proposal containing:

- `probe_id`
- `slot_id`
- normalized query
- discovery round
- origin: deterministic semantic decomposition or governed model fallback

Validation requires a non-blank query, an existing slot, uniqueness within the run, the `document.search` capability in the request-scoped catalog, and remaining budget.

### 7.3 `DiscoveryCheckpoint`

Add one nullable checkpoint slot containing:

- target slots;
- accepted probes;
- candidate matches in persisted result order;
- selected candidate per slot;
- rounds and probes consumed;
- current discovery status;
- clarification choice manifest when suspended.

Old checkpoints missing this key normalize explicitly to `None`; current-version payloads missing it fail required-key validation.

### 7.4 Ranked candidates and calibration

Reuse `DocumentDiscoveryCandidate` and `DiscoveryCandidateRegistry`; extend the server-owned candidate shape compatibly so old checkpoint payloads remain loadable. A ranked candidate carries:

- opaque candidate identity;
- authoritative document identity and pinned revision;
- rank;
- raw vector distance and raw reranker score as server-only diagnostics;
- calibrated confidence in `[0, 1]`, or `None` when calibration is unavailable;
- match kind, such as exact document number, exact normalized title, or semantic rank;
- ACL-safe display title and document number;
- calibration artifact version/hash.

The current lightweight probe already obtains best vector distances but discards them and performs no rerank. The revised adapter preserves the best chunk per document ephemerally, performs one bounded batch rerank per probe, and never checkpoints or logs the chunk text used by the reranker. Reranker scores are not treated as probabilities.

A model-versioned `DiscoveryConfidenceCalibrator` maps raw reranker score plus exact-match features to confidence. The initial implementation uses an offline-fitted monotonic isotonic mapping over labeled `(query, candidate, relevant)` examples, stratified by query shape and workspace in evaluation but shared globally per retrieval/reranker model version; per-workspace calibrators are excluded because sparse workspace labels would be unstable. The calibration artifact records retrieval model hash, reranker model hash, dataset hash, fit metrics, and mapping version. A missing or hash-mismatched artifact disables semantic auto-selection: exact validated number/title matches may still auto-select, while semantic candidates go to clarification. Canary enablement is blocked until the artifact passes the labeled-data gate, preventing the common semantic case from degrading silently to permanent clarification.

Raw content is not part of a checkpointed candidate. A candidate with unavailable confidence can appear in clarification choices but cannot be auto-selected unless it has a validated exact document-number or exact normalized-title match.

The model-facing projection includes only probe outcome, rank bands, and gap reason. It excludes raw document IDs, revisions, ACL facts, runtime secrets, raw scores, calibration internals, and candidate text.

## 8. Probe Generation

Probe generation is hybrid:

1. Deterministic probes come from resolved document references, normalized titles/numbers, comparison sides, and thematic summary subjects in `SemanticContext`.
2. If required slots remain without usable probes, the governed planner model may propose additional query text for those existing slots.
3. The validator reconstructs accepted probes server-side.

Limits:

- maximum five probes total;
- maximum two rounds;
- top five candidates per probe;
- probes within one round may execute in parallel when the capability descriptor permits it;
- 15-second discovery deadline, also bounded by the request’s outer deadline;
- round two is permitted only for a recorded gap from round one;
- an identical or normalized-equivalent probe cannot execute twice.

## 9. Candidate Selection Policy

Selection order:

1. Validated exact document-number match.
2. Validated exact normalized-title match.
3. Current ACL/workspace visibility and pinnable revision.
4. Calibrated confidence threshold.
5. Top-one versus top-two confidence margin.
6. Slot-specific duplicate and role constraints.

Initial configuration defaults are:

- confidence threshold: `0.82`;
- confidence margin: `0.12`.

These thresholds apply to calibrated output, never raw similarity or reranker scores. The feature flag remains disabled until the versioned calibration artifact and golden dataset confirm or adjust the values. Changing the calibrator, retrieval/reranker model hash, threshold, or margin requires the rollout gate to be rerun.

A document cannot fill both sides of a comparison unless finalized semantics explicitly identify two sections of the same document. Candidate deduplication uses authoritative document identity while preserving every probe-to-candidate lineage edge.

### 9.1 Summary selection

For a summary naming a specific document:

- exactly one primary target is selected;
- up to four additional confident documents that will actually be read are bound as `reference`, not `supporting`;
- each selected reference receives its own `TargetUnit`, partial-read coverage criterion, and factual read/retrieve task;
- reference evidence may add context but cannot be rendered as content belonging to the primary target.

A `supporting` binding without a target unit remains discovery metadata only and is never claimed as answer context. This preserves the existing `validate_task_plan` rule that only `target` and `reference` bindings are plannable.

For a thematic summary:

- three target documents are selected by default;
- the policy may expand to at most five targets when every added target passes the confidence gate;
- map/reduce evaluation tracks coverage per target, so a missing target cannot be hidden by evidence from another document.

### 9.2 Comparison selection

Each comparison side has one primary slot and receives a separate probe set. Exactly one primary document is selected per side. Additional results are candidate metadata unless the policy explicitly selects them as readable `reference` context; they never create extra comparison sides.

## 10. Existing Discovery Composition, Binding, and Plan Expansion

Bootstrap reuses the existing `document.search` capability, `DocumentDiscoveryCandidate`, persisted `AgentResult`, `DiscoveryCandidateRegistry`, binding resolver, ACL checks, and exact-revision pinning. It does not introduce a second candidate namespace.

The two settle paths have disjoint ownership:

- existing `discovery_settle_node` continues to process non-bootstrap search task IDs, including R37/recovery discovery, and may add only `discovered` or `supporting` bindings under the existing policy;
- new `selection_settle_node` processes only bootstrap task IDs recorded in `DiscoveryCheckpoint` and may create `target` or `reference` bindings only for checkpointed target slots selected by the confidence policy or user;
- generic settle explicitly excludes bootstrap task IDs, while selection settle rejects every candidate not owned by them;
- both paths call shared registry/resolver helpers and deduplicate the same `(document_id, revision)` identity. An existing weaker-role binding is promoted only through explicit selection provenance; no duplicate candidate or pin is minted;
- existing `discovery_deferred` and `V2_MAX_DISCOVERED_DOCUMENTS` remain owned by generic settle. Bootstrap stores unresolved/unused candidates in `DiscoveryCheckpoint` and is governed by its separate probe/slot/summary-target budgets.

Add a distinct `BindingSelectionRequest` for checkpoint-selected target/reference roles. Existing `BindingAdditionRequest` and `BindingPromotionRequest` contracts remain unchanged and cannot create a target. Persist selection provenance containing selected binding ID, source discovery task ID, source candidate ID, target slot ID, and selection authority (`confidence_policy` or `user_choice`). This is server-policy selection, not planner promotion.

### 10.1 Expansion contract

Add a separate frozen validator:

```text
validate_plan_expansion(
    current,
    proposed,
    discovery_checkpoint,
    selected_bindings,
    discovery_outcomes,
    research_budget,
) -> TaskPlan
```

It is not `validate_replan`; the existing `validate_replan` rule that forbids target-unit changes remains unchanged. `validate_plan_expansion` enforces:

1. `plan_id`, goal, contract version, and every existing task are unchanged;
2. existing tasks are an exact prefix and existing target units are an exact prefix;
3. at least one target unit and one factual task are appended;
4. every new target unit points to exactly one checkpoint-selected slot/binding pair and that binding has role `target` or `reference`;
5. new target IDs and task IDs are unique and cannot shadow bootstrap IDs;
6. new tasks reference only appended target units, use the current capability catalog, and carry `DiscoveryExpansionTaskOrigin` linking the slot and bootstrap search task IDs;
7. discovery outcomes cover every attempted bootstrap task and selected candidates are owned by those outcomes;
8. discovery-task budget and research-task budget are counted separately; bootstrap probes do not consume factual research task capacity;
9. the complete proposed plan passes ordinary `validate_task_plan` against the expanded binding set.

`expand_discovery_plan` is the single authoritative constructor for this transition. It appends both target units and tasks, calls `validate_plan_expansion`, and returns nothing checkpointable before validation and lease commit. No other node may add target units or intent-selected bindings.

The AST ownership guard in `test_subagent_cannot_append_authoritative_tasks` must be updated deliberately: `expand_discovery_plan` becomes the second allowed append expression surface beside `append_replan_tasks`; its only caller/plan-return owner is `research_expand_node`. The guard must scan target-unit appends as well as task appends. All existing constructor and caller assertions remain explicit; a broad directory or wildcard allowlist is forbidden.

Evidence evaluation derives completion only from the expanded target units and their factual read/map tasks; bootstrap `document.search` tasks remain lineage-bearing discovery work and cannot satisfy, weaken, or create research coverage requirements.

The summarize skill is extended to support thematic multi-target map/reduce. Named-document summary remains one primary target plus explicitly readable reference target units. The compare skill retains exactly two primary sides.

## 11. Clarification and Resume

When selection is ambiguous, the graph interrupts rather than finalizing or guessing. The public payload contains at most three choices per unresolved slot:

```json
{
  "type": "document_selection",
  "question": "Chọn văn bản phù hợp",
  "slots": [
    {
      "slot_label": "Vế so sánh thứ hai",
      "choices": [
        {
          "choice_token": "opaque-token",
          "title": "Nghị định 13/2023/NĐ-CP",
          "document_number": "13/2023/NĐ-CP"
        }
      ]
    }
  ]
}
```

The payload never exposes document UUIDs, revision IDs, workspace/ACL details, raw scores, evidence IDs, or use IDs.

On resume, the graph validates:

1. the token belongs to the suspended checkpoint and target slot;
2. the candidate still belongs to the authenticated workspace scope;
3. the candidate revision remains pinnable and retained;
4. the intended role remains compatible with finalized semantics;
5. the lease commits before the selected binding is checkpointed.

A forged, stale, or unauthorized choice is rejected. The graph may return refreshed choices when safe; otherwise it produces a typed denial/unavailable result without revealing hidden metadata. Users may choose “none of these,” which terminates the suspended flow with a request to provide a name, number, or reformulated question.

## 12. Failure Behavior

| Condition | Required behavior |
|---|---|
| No candidate after two rounds | Interrupt requesting a document name/number; never pick arbitrarily. |
| Search timeout or dependency failure | Typed unavailable; no stale-data or arbitrary-document fallback. |
| Candidate appears in multiple probes | Deduplicate identity while preserving probe lineage. |
| Same document fills two comparison sides | Reject unless same-document section comparison is explicit. |
| ACL changes before resume | Revalidate, hide sensitive reason, refresh choices or return typed denial. |
| Selected revision is no longer pinnable | Reject selection; never silently move to current revision. |
| Discovery budget exhausted | Clarify or typed unavailable; no additional search. |
| Cancellation or kill switch | Terminal cancellation; planner and synthesis do not continue. |
| Missing or model-hash-mismatched calibration artifact | Exact matches may auto-select; semantic candidates clarify; canary enablement remains blocked. |
| Invalid expanded plan | Fail closed before research dispatch. |

## 13. Configuration

Declare and validate all fields in `Settings` and document them in `.env.example`:

```text
V2_DISCOVERY_BOOTSTRAP_ENABLED=false
V2_DISCOVERY_MAX_PROBES=5
V2_DISCOVERY_MAX_ROUNDS=2
V2_DISCOVERY_TOP_K=5
V2_DISCOVERY_DEADLINE_SECONDS=15
V2_DISCOVERY_SUMMARY_TARGETS=3
V2_DISCOVERY_MAX_SUMMARY_TARGETS=5
V2_DISCOVERY_CONFIDENCE_THRESHOLD=0.82
V2_DISCOVERY_MARGIN_THRESHOLD=0.12
V2_DISCOVERY_CALIBRATION_ARTIFACT=/app/config/discovery-calibration.json
```

Validation enforces positive integer limits, `summary_targets <= max_summary_targets`, thresholds within `[0, 1]`, and a discovery deadline no greater than the outer request deadline at runtime. Enabling bootstrap requires a readable calibration artifact whose recorded retrieval/reranker hashes match the effective providers; mismatch fails the semantic auto-selection readiness gate.

The implementation must also correct the existing configuration gap for `V2_MAX_TASKS`, `V2_MAX_PARALLEL_BRANCHES`, `V2_MAX_REPLANS`, and discovery-policy settings: any setting consumed by `V2ResearchLimits.from_settings()` or `build_discovery_policy()` must be a declared, validated Settings field. Environment values that Pydantic ignores are not considered configuration support.

## 14. Module Boundaries

Create:

```text
backend/app/services/agents/v2/discovery/
  __init__.py
  contracts.py
  policy.py
  calibration.py
  projection.py
  nodes.py
  plan_expansion.py
```

Integrate with:

- `v2/nodes/routing.py` for the flag-controlled summarize-to-complex decision;
- `v2/complex_research_graph.py` for topology, exclusion of bootstrap task IDs from generic settle, and explicit parent/child state mapping;
- `v2/capabilities/document.py` and its v1-backed adapter for ranked candidate output, bounded rerank, and calibration inputs;
- `v2/contracts/binding.py` for selection provenance;
- `v2/contracts/state.py` for the nullable discovery checkpoint;
- `supervisor_v2.py` and `runtime_selector.py` for service wiring;
- `agent/streaming.py` for discovery progress and clarification projection;
- `core/config.py` and `.env.example` for validated controls.

Keep scoring, role assignment, projection, and plan expansion outside the already large `complex_research_graph.py` except for graph composition and boundary adapters.

## 15. Observability and Privacy

Record metadata-only metrics:

- discovery rounds and probes used;
- candidates per probe;
- match-kind counts;
- automatic-selection, clarification, and “none” rates;
- discovery duration and timeout count;
- selected-target correctness in offline labeled evaluation;
- selection source: exact match, confidence gate, or user choice.

Do not log query text, candidate titles, document numbers, IDs, revisions, prompts, evidence, or user selections in timing/rollout metadata. Content-suppressed tracing applies to the discovery planner model. Clarification content follows the existing user-visible event path but is not copied into metadata-only timing spans.

## 16. Testing

### 16.1 Unit tests

- target-slot role derivation for explicit, quoted, named, and thematic requests;
- flag-off routing compatibility and flag-on bounded-summary routing to complex; compare remains complex;
- deterministic probe generation and model fallback projection;
- probe uniqueness and all budget validators;
- raw-score-to-confidence calibration, artifact/model-hash mismatch, exact-match bypass, margin, duplicate, and same-document policies;
- named and thematic summary target/reference selection;
- comparison side selection;
- public clarification redaction;
- `validate_plan_expansion` acceptance and every mutation/rebinding rejection;
- AST ownership guard coverage for both task and target-unit appends;
- separation and deduplication of bootstrap selection settle versus existing generic settle.

### 16.2 Graph and durability tests

- summary without `document_ids`;
- comparison without `document_ids`;
- quoted A while asking for B;
- quoted A compared with discovered B;
- thematic multi-document summary;
- interrupt and resume with a valid choice;
- forged choice token;
- ACL revocation before resume;
- crash after search, selection, lease commit, and plan expansion;
- resume performs zero duplicate capability dispatches;
- checkpointed candidate identity never slides to another revision;
- cancellation and kill-switch termination.

### 16.3 Security tests

- foreign-workspace candidates never appear in choices or bindings;
- model-minted IDs and roles are rejected;
- raw UUIDs, revisions, ACL metadata, scores, evidence IDs, and runtime secrets never reach model-facing or public projections;
- an unauthorized candidate cannot be inferred from error shape.

### 16.4 Regression tests

- direct, fast-domain, people, write, and v1 paths remain unchanged;
- existing explicit scoped retrieval remains unchanged when the feature flag is off;
- compare and summarize behavior with the feature flag off remains byte-compatible with current contracts, including the current `retrieve-unscoped` refusal fallback;
- flag-on bootstrap replaces that fallback only for summarize/compare, while evaluate/multi-goal/cross-domain retain it;
- existing R37 recovery discovery still settles only discovered/supporting roles;
- scheduler, lease, synthesis, citation, plan-constructor, and append-owner guards still pass.

## 17. Rollout

1. **Calibration, tests, and shadow:** fit and version the calibration artifact from labeled data, keep the feature flag off, then run golden and shadow traffic with metadata-only metrics.
2. **Internal canary:** enable only for allowlisted workspaces.
3. **Staged eligible rollout:** 5% → 25% → 50% → 100% of v2-eligible traffic.

Promotion gates:

- zero workspace/ACL leaks;
- zero duplicate dispatches across crash/resume tests;
- calibration artifact hashes match the effective retrieval and reranker models;
- at least 95% selected-target accuracy on the labeled dataset, with semantic auto-selection and clarification rates reported separately;
- discovery p95 no greater than 15 seconds;
- clarification and typed-unavailable rates reviewed against baseline;
- no regression in grounded answer and citation gates.

The feature flag and existing v2 kill switch provide rollback. Disabling discovery restores the existing v2 behavior; it never changes v1.

## 18. Definition of Done

The feature is complete when:

1. With the flag on, summarize routes through complex discovery even with one bound document; with the flag off, current fast/fallback routing is unchanged.
2. Supported summarize and compare requests can succeed without supplied `document_ids` when confident authorized candidates exist.
3. Quoted documents are assigned roles from query meaning rather than automatically constraining the target.
4. Every discovery and research capability dispatch is backed by a validated, checkpointed plan.
5. `validate_plan_expansion` and the AST ownership guard prove that only the governed expansion owner can add target units and factual tasks.
6. Bootstrap selection and existing generic discovery share candidate identity but have disjoint settle/role authority.
7. Ambiguous selection suspends and resumes from the same checkpoint.
8. Selected revisions are ACL-validated, exactly pinned, leased, and provenance-audited.
9. The hard discovery budget is enforced under success, failure, retry, cancellation, and resume.
10. Versioned calibration, security, durability, regression, and rollout gates pass.
