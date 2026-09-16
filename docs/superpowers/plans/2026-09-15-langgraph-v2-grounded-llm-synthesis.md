# LangGraph v2 Grounded LLM Synthesis Implementation Plan

> **Spec authority:** `docs/superpowers/specs/2026-09-15-langgraph-v2-grounded-llm-synthesis-design.md` (commit `564cd67`, "clarify synthesis checkpoint and grounding invariants")
>
> **Reviewed:** two review rounds complete; all raised blockers resolved in spec §6.2, §10.3–10.5, §11.6, §13.1–13.5.

**Goal:** Replace the extractive `build_extractive_draft` production path with a bounded, checkpointed, claim-first LLM synthesis pipeline for document-backed factual answers: governed evidence selection → stable `E1..En` handle manifest → structured claim proposal → deterministic anchor validation → claim-first grounding → single-owner citation projection → server-rendered Markdown → citation-before-token SSE. At most two provider calls per synthesis operation, durable across crash/resume.

**Confirmed design decisions (this plan's forks, resolved with the user):**

- **Topology:** the synthesis state machine is a **compiled subgraph** mounted as the supervisor's `synthesize` node (same pattern as `complex_boundary`: compiled without a saver, inherits the supervisor's checkpointer — and therefore the shadow run's isolated saver). The supervisor edge becomes `evaluate → synthesize → finalizer`; the standalone `ground` node leaves the production topology (its assertion-splitter/string-mapping code remains only for extractive compatibility tests).
- **Reduce order channel:** `summarize_reduce_node` keeps only deterministic evidence preparation and writes the ReduceSpec-ordered use refs into the evaluation the outer supervisor sees. Because `EvidenceEvaluation` is a frozen contract with no use-order field today, this plan adds an **additive optional field** `synthesis_use_order: tuple[EvidenceUseRef, ...] | None = None` to `EvidenceEvaluation` (same `contract_version`, same additive-compat rule as §13.2). `_synthesis_input_of` prefers it when present, else falls back to task-result flatten order.
- **Output:** this document only; implementation proceeds task-by-task under TDD.

**Global constraints**

- Work only in `/home/AIRAG/.worktrees/langgraph-v2` on `feat/langgraph-v2`.
- Do not push, merge, deploy, restart Docker services, vLLM, or live infrastructure.
- TDD is mandatory: focused failing test first, record expected failure, implement minimum change, rerun.
- Run GitNexus upstream `impact` before modifying every function/class/method; report HIGH/CRITICAL before editing. Run `detect_changes --scope compare --base-ref main` before every commit/final completion.
- V2 remains sole authority for ACL, binding, revision pin, routing, scheduling, evidence, checkpointing, citation identity, and terminal status. The model receives only minimized evidence + opaque `E`-handles; it never sees internal UUIDs, plans, bindings, ACL, or checkpoint state.
- No production fallback to `build_extractive_draft()`; no same-request V1 fallback on synthesis failure; no new model role, traffic flag, or `.env.example` change.
- Spec §10.4 honesty: anchor validation is literal/canonical support, **not** semantic entailment. Do not add a second judge; do not claim compound-sentence detection beyond deterministic multi-sentence/paragraph rejection.

---

## Task 1 — `SynthesisCheckpoint` contract + additive `synthesis` slot

**Expected production seams:**

- `backend/app/services/agents/v2/contracts/synthesis.py` (new `SynthesisCheckpoint`, `HandleManifest`, `ParsedCandidate`, `GroundedArtifact` models)
- `backend/app/services/agents/v2/contracts/state.py` (`SupervisorV2State.synthesis`)
- `backend/app/services/agents/v2/supervisor_v2.py` (`_SLOT_MODELS`, `_NULLABLE_SLOTS`, `build_initial_v2_state`, `normalize_checkpoint_state` legacy branch)
- `backend/app/services/agents/v2/contracts/validation.py` (`_CHECKPOINT_REQUIRED_KEYS`, `validate_synthesis_checkpoint`)

**Expected tests:** `backend/tests/agents/v2/test_synthesis_checkpoint_contract.py`

### 1A. Contract models

`SynthesisCheckpoint` (frozen `ContractModel`, `contract_version="2.0"`):

```text
phase: prepared | attempt_reserved | candidate | grounded | failed
attempts_started: 0 | 1 | 2
handle_manifest: tuple[HandleEntry]        # handle "E1" -> EvidenceUseRef
candidate: ParsedCandidate | None          # phase=candidate
grounded: GroundedArtifact | None          # phase=grounded
failure_code: str | None                   # closed codes only
```

`ParsedCandidate`: server-assigned `claim-N` ids, claim text, cited `E`-handles, per-claim presentation kind. `GroundedArtifact`: grounded claims with resolved `EvidenceUseRef`s, server-rendered content, public citation projection (allowlisted projector output only), presentation kind per claim. Never: raw evidence, prompt text, unparsed output, reasoning, secrets, ACL state, stack traces.

### 1B. Slot wiring (spec §13.1 checklist, all required)

- `SupervisorV2State`: `synthesis: SynthesisCheckpoint | None` (nullable; `None` = canonical idle).
- `_SLOT_MODELS["synthesis"] = SynthesisCheckpoint`; `_NULLABLE_SLOTS` += `"synthesis"`.
- `_CHECKPOINT_REQUIRED_KEYS` += `"synthesis"`.
- `build_initial_v2_state()`: `synthesis=None`.
- `normalize_checkpoint_state()`: legacy root-`2.0` payload missing the key → insert `synthesis=None` **before** required-key validation; a current payload missing it outside that branch fails closed.
- Fresh-turn/context hygiene clears `synthesis` with stale terminal state.

Acceptance: spec §17.1 — fresh state has `synthesis=None`; old checkpoint normalizes; current checkpoint missing the key fails; all phases round-trip JSON serde + real `AsyncPostgresSaver`; fresh turn clears prior synthesis.

---

## Task 2 — Presentation policy + target-aware evidence selector

**Expected production seams:**

- `backend/app/services/agents/v2/synthesis/` (new package): `presentation.py` (`PresentationMode`, eligibility decision), `selection.py` (`SynthesisEvidenceSelector`)
- `backend/app/services/agents/v2/nodes/synthesize.py` (selector replaces `apply_budget_split` on the LLM path; FIFO helper stays for extractive compat)

**Expected tests:** `backend/tests/agents/v2/test_synthesis_presentation.py`, `test_synthesis_selection.py`

### 2A. PresentationPolicy

Deterministic `PresentationMode` from checkpointed route/plan + admitted evidence kinds: `document_grounded_llm` | `people_card` | `direct` | `typed_unavailable`. Rules per spec §6.1: document retrieval/section/summarize-final/complex-with-document-backed-claims → LLM; People → existing card path (never the synthesis model); KG-only or KG-claim-without-document-lineage → typed unavailable, never a fake document citation; derived eligible only when every cited derived use recursively projects to authorized locatable document lineage.

### 2B. Target-aware selector

Group admitted evidence by `target_id` + global/supporting group; preserve deterministic in-group order. Two passes: (1) coverage pass — ≥1 eligible item per required target with admitted evidence; (2) round-robin fill to budget. Reserve budget for system/schema, query, delimiters/handles, output tokens. A required target that had admitted evidence but survives with none → fail closed (`selection_missing_target`), never silently answer one side. Single-target keeps ranking order.

Acceptance: spec §17.2 + §17.4 — People makes zero LLM calls; KG-only typed-unsupported; two-target compare selects from both before extra quota; deterministic across retries; no FIFO starvation; derived overflow not mislabeled as semantic summary.

---

## Task 3 — Handle manifest + structured claim adapter + privacy-safe provider

**Expected production seams:**

- `backend/app/services/agents/v2/synthesis/handles.py` (manifest build/resolve)
- `backend/app/services/agents/v2/synthesis/adapter.py` (`StructuredLLMDraftBuilder`: prompt build, buffered call, strict parse)
- `backend/app/services/llm/` (synthesis provider factory: effective `main` config, content-suppressed tracing — NOT a new role)
- `backend/app/services/agent/runtime_selector.py` (ingress wires `RuntimeServices.answer_draft_builder`)

**Expected tests:** `backend/tests/agents/v2/test_synthesis_handles.py`, `test_synthesis_adapter.py`, `test_synthesis_tracing.py`

### 3A. HandleManifest

Server-built before the first provider call; checkpointed in `SynthesisCheckpoint`. `E1..En` → exact `EvidenceUseRef`, order = selected-evidence order, no plaintext, opaque to model. Identity invariant: repair reuses the same manifest; resume never rebinds a handle (denied `E2` fails rehydration, never becomes the next surviving use).

### 3B. Structured adapter

Prompt per spec §9.3 (Vietnamese default, one material assertion per claim, ≤3 handles/claim, evidence as untrusted quoted data, ignore embedded instructions, placeholder-only examples). Buffered call, `temperature=0`, bounded output, early-break on parseable JSON. Parser per §9.2: fenced JSON ok / prose rejected; unknown fields rejected; 1–12 claims; per-claim + total char limits; handles must exist in manifest (dedupe preserving order); obvious multi-sentence/paragraph claims rejected via existing splitter (`claim_multi_sentence`); duplicate normalized text rejected; server assigns `claim-N` ids; resolves handles through the manifest only.

### 3C. Privacy-safe provider

`get_main_provider_for_synthesis()` (conceptual): resolves the effective `main` connection/model from existing runtime config, applies a synthesis-specific tracing policy — allowlisted operational metadata only (role, provider/model, evidence count, char/token estimate, claim count, latency, usage, outcome code, repair flag, cancellation). Never: query/evidence/prompt/output/answer text, reasoning, internal IDs. Same suppression for Langfuse and the dataset trace collector.

Acceptance: spec §17.5 + §17.6 + §17.12 — manifest stability across repair/resume; injection cannot alter schema/authority; raw output never logged/traced; VN/EN language preserved.

---

## Task 4 — Anchor canonicalization + claim-first grounding + CitationProjector + renderer

**Expected production seams:**

- `backend/app/services/agents/v2/synthesis/anchors.py` (extraction + canonicalization + per-item support check)
- `backend/app/services/agents/v2/synthesis/grounding.py` (claim-first grounding; no rendered-Markdown parse-back)
- `backend/app/services/agents/v2/synthesis/citations.py` (`CitationProjector`, public handle generation)
- `backend/app/services/agents/v2/synthesis/render.py` (server-owned Markdown + marker insertion)

**Expected tests:** `backend/tests/agents/v2/test_synthesis_anchors.py`, `test_synthesis_grounding.py`, `test_citation_projector.py`

### 4A. Anchors (spec §10.3 — conservative, domain-aware)

Extract from each claim: monetary/numeric quantities, percentages, dates, durations/deadlines, `Điều/Khoản/Điểm/Chương/Mục` locators, official document numbers, configured closed literals. Canonicalization rules: grouping separators where unambiguous; VN scale words (`nghìn/ngàn`/`vạn`/`triệu`/`tỷ`) as multipliers on numeric tokens; currency/unit is part of the anchor; ambiguous decimal/grouping → fail conservative; `%` preserved; zero-padded dates/durations normalize only when semantically equivalent; `5 ngày làm việc` ≠ `5 ngày`; `Khoản 2 Điều 5` never weakened to `Điều 5`; doc numbers exact, never fuzzy. **Support rule:** canonical anchor must occur in ≥1 *individually cited* evidence item — never assembled across sources. Unsupported → repair path (`claim_anchor_unsupported`).

### 4B. Claim-first grounding

`ParsedClaim[] → resolve EvidenceUseRefs via manifest → validate claims → anchor guards → GroundedClaim[]`. Rehydrate cited exact uses under current authority when a checkpoint boundary was crossed. No rendered-text parse-back anywhere in production.

### 4C. CitationProjector (single owner)

`GroundedClaim → EvidenceUseRef → EvidenceRecord.source → PublicCitation`. `DocumentSourceIdentity` → authoritative doc/revision/locator stores under current scope. `DerivedSourceIdentity` → recursive expansion to authorized locatable lineage, dedup deterministic lineage order; bare derived citation invalid. `PeopleSourceIdentity`/`KnowledgeGraphSourceIdentity` → not projected (typed unavailable upstream). Public metadata per spec §11.3 allowlist (`citation_id`, 4-char `index` with ≥1 letter, `label`, `source_type`, `document_id`, `chunk_id`, excerpt, `source_file`, `page_no`, `heading_path`, `document_number`, `article_label`, `validity_status`, `superseded_by`); internal identities forbidden. Index assignment: deterministic, collision-resolved, first-claim-use then evidence order, ≤3 per claim, `citation_id`/`index` may share the opaque value.

### 4D. Renderer

Render each grounded claim once; markers inserted from resolved citations immediately before terminal punctuation (`[a3z9][b2m7]`, never `[a3z9, b2m7]`, no leading space, no references list). Summary claims → opening paragraphs; detail → bullets/paragraphs; caveat → bounded caveat. Rendering after grounding cannot break claim identity.

Acceptance: spec §17.7 (all anchor cases incl. `20.000.000 đồng`↔`20 triệu đồng`, `2 tỷ`, `500 nghìn`, distinct duration qualifiers, hierarchical locators) + §17.8 residual-risk fixture (A/B misattribution documented, not claimed as caught) + §17.11 (projector single-owner, V1 marker syntax, derived lineage, People/KG never disguised).

---

## Task 5 — Synthesis subgraph (state machine) + supervisor rewiring

**Expected production seams:**

- `backend/app/services/agents/v2/synthesis/graph.py` (new: `build_synthesis_subgraph()`, state, nodes)
- `backend/app/services/agents/v2/supervisor_v2.py` (`SUPERVISOR_V2_NODES`: `synthesize` → subgraph wrapper; remove `ground` from production edges; `_complex_branch` unchanged → `synthesize`)
- `backend/app/services/agents/v2/nodes/synthesize.py`, `grounding.py` (production paths replaced; extractive helpers remain for compat tests only)

**Expected tests:** `backend/tests/agents/v2/test_synthesis_graph.py`, `test_synthesis_resume.py`

### 5A. Subgraph topology

```text
prepare → reserve → generate → validate_ground ─┬─ success → finalize_artifact → END
                  ↑                             ├─ repairable & attempts<2 & deadline → reserve
                  │                             └─ else → failed → END
```

Compiled **without** a checkpointer (inherits supervisor's — shadow isolation preserved). Each durable phase transition is a node boundary:

- `prepare`: presentation policy → governed hydration → target-aware selection → build + checkpoint `HandleManifest` (phase `prepared`).
- `reserve`: `attempts_started += 1`, checkpoint **before** provider call (phase `attempt_reserved`). Deliberate two-checkpoint sequence (`prepared` then `attempt_reserved`) per spec §12.1 — latency accepted for restart safety.
- `generate`: buffered provider call via the privacy-safe builder; outcome checkpointed as parsed bounded candidate **or** closed failure code — never raw output (phase `candidate` or `failed`-eligible).
- `validate_ground`: manifest resolution → claim/schema validation → anchor guards → rehydrate cited uses if boundary crossed → citation projection → render → checkpoint grounded artifact (phase `grounded`) or route repair/failed.
- `finalize_artifact`: write `final_response`-bound artifact state for the supervisor merge.

### 5B. Resume rules (spec §13.5)

`synthesis=None` → prepare; `prepared` → reuse manifest, reserve attempt 1; `attempt_reserved` → consumed, closed interruption code, reserve repair only if `attempts_started < 2` + deadline; `candidate` → validate/ground checkpointed candidate, zero regeneration; `grounded` → zero model calls, rehydrate + revalidate current authority before emission; `failed` → same typed failure, zero calls. Any exact use failing current ACL/revision/expiry/tombstone/lineage/retention → fail closed, no handle substitution.

### 5C. Supervisor rewiring

`SUPERVISOR_V2_NODES["synthesize"]` becomes the wrapped subgraph boundary (parent→child mapping like `build_complex_research_state`; merge back `synthesis` slot + grounded artifact). Remove `ground` node + `synthesize → ground → finalizer` edges; add `synthesize → finalizer`. `_route_branch`/`_complex_branch` unchanged. `AnswerDraftChannel` may remain as in-process cache; `SynthesisCheckpoint` is authoritative.

Acceptance: spec §17.10 — two-checkpoint sequence observable before first provider call; crash during attempt 1 allows at most repair; crash during attempt 2 allows none; candidate resume zero regeneration; grounded resume zero calls; revoked uses fail without renumbering; ≤2 calls across crash/resume.

---

## Task 6 — Outer streaming adapter: citation-before-token + finalizer integration

**Expected production seams:**

- `backend/app/services/agent/streaming.py` (`stream_v2_turn_events` success branch)
- `backend/app/services/agents/v2/nodes/finalizer.py` (consume grounded artifact → `FinalResponse` with citations)

**Expected tests:** `backend/tests/agents/v2/test_synthesis_streaming.py`, `test_synthesis_finalizer.py`

### 6A. Citation frame ordering (spec §11.6)

In the success branch (currently `status(generating)` → `_chunk_prose` loop): emit `status(generating)` → one `citation` frame carrying the canonical projector output stored with the grounded artifact → token chunks → `complete` (repeating the **same** citation identity set — never rebuilt). Frontend `case "citation"` already exists (`useRAGChatStream.ts:593`); contract stays additive.

### 6B. Finalizer

`finalizer_node` consumes the checkpointed grounded artifact → `FinalResponse(status="success", content=rendered, citations=projector output)`. `synthesis_failed` → typed error terminal with the safe Vietnamese message persisted as nonblank assistant content. Non-sufficient/unsupported presentation → existing typed terminals; never `build_extractive_draft`.

Acceptance: spec §17.13 — tokens only after grounding/rendering + exactly one `complete`; citation frame before first token; `complete` repeats identical indexes; failure → no tokens + one typed `error`; safe message persists nonblank; cancellation → no late success.

---

## Task 7 — Summarize reduce single-owner refactor

**Expected production seams:**

- `backend/app/services/agents/v2/complex_research_graph.py` (`summarize_reduce_node`)
- `backend/app/services/agents/v2/contracts/evaluation.py` (additive `synthesis_use_order` field)
- `backend/app/services/agents/v2/nodes/synthesize.py` (`_synthesis_input_of` prefers ordered refs)

**Expected tests:** `backend/tests/agents/v2/test_summarize_single_owner.py`

### 7A. Strip reduce to evidence preparation

`summarize_reduce_node` must: verify every `ReduceSpec.map_task_ids` has a checkpointed result (missing → `ComplexResearchError`, unchanged); collect map-task `EvidenceUseRef`s in spec order; write them into the evaluation the outer supervisor sees via the new `synthesis_use_order` field; return. It must **not**: call the provider, create a user-visible draft, reserve attempts, own a handle manifest, or touch `answer_draft_channel`.

### 7B. Contract addition

`EvidenceEvaluation.synthesis_use_order: tuple[EvidenceUseRef, ...] | None = None` — additive optional, same `contract_version`, same normalization precedent as §13.2. `_synthesis_input_of` uses it when non-None; else task-result flatten (unchanged for non-summarize routes).

Acceptance: spec §17.3 — zero provider calls in reduce; exactly one synthesis state machine owns the final summary; no channel bypass of `SynthesisCheckpoint`; resume cannot produce reduce-call + outer-call.

---

## Task 8 — Documentation + gates

**Expected seams:** `CLAUDE.md`, `README.md`, `docs/harness.md`, `docs/pre-canary-handoff.md`

- `CLAUDE.md`: canonical V2 synthesis ownership — presentation policy, privacy-safe builder, handle manifest, nullable `synthesis` slot + compat rule, claim-first grounding, anchor canonicalization, single-owner citation flow, summarize single-owner rule.
- `README.md`: pointer only.
- `docs/harness.md`: focused test entrypoints (contract/selection/handle/checkpoint/tracing/synthesis/citation) + live acceptance commands.
- `docs/pre-canary-handoff.md`: synthesis failure, privacy tracing, target coverage, checkpoint compat/latency, summarize bypass guard, citation ordering, People/KG exclusions, rollback checks.

Run: full offline v2 suite (`tests/agents/v2`, `tests/api`, `tests/migrations/v2`, `tests/workers`), static guards, frontend build, `detect_changes --scope compare --base-ref main`. Commit docs + write final report under `.superpowers/sdd/2026-09-15-langgraph-v2-grounded-llm-synthesis/`.

---

## Cross-cutting acceptance (spec §20, condensed)

Two provider calls max per synthesis op incl. crash/resume · no token before grounded-artifact checkpoint · citation frame before first token · `complete` repeats identical citation set · every in-scope claim carries 1–3 V1-compatible citations · manifest never rebinds handles · People/KG never enter document synthesis · reduce never bypasses the state machine · tracing content-free · cancellation never yields late success · V1 unchanged · no extractive fallback reachable from production paths.
