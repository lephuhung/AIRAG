# Final fix report — LangGraph v2 Phase 4 semantic/adaptive execution

Sole-writer final fix wave for the `final-review.md` VERDICT FIX_REQUIRED.
FIX_BASE `c21f63f` (Task-12 review round 2 HEAD). Scope: the four **Important**
findings plus the cheap, tightly-coupled Moderate items; the rest are
documented as descoped/ledgered rather than re-opened.

Plan: `docs/superpowers/plans/2026-09-14-langgraph-v2-phase4-semantic-adaptive-execution.md`
Spec (binding): `docs/superpowers/specs/2026-09-14-langgraph-v2-semantic-adaptive-execution-design.md`
Contract spec (frozen-literal authority): `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`
Ledger: `.superpowers/sdd/2026-09-14-langgraph-v2-phase4-semantic-adaptive-execution/progress.md`

## Fixed findings

### I1 — `evaluate`/compliance unreachable in the production-wired arm (FIXED)

The v1 taxonomy has no evaluate intent, so a compliance query was
model-classified to `search` → `retrieve/document` → targetless retrieval,
and Task 12 had already lifted the v1 fallback for `evaluate`.

Fix (reviewer option (a), v2-only — no shared v1 prompt change):

- `backend/app/services/agents/v2/semantic/intent.py`: added `"evaluate"` to
  `_VALID_INTENTS`, added the conservative deterministic narrow scope
  `classify_evaluate()` keyed on strong compliance/legal-validity cues
  (`tuân thủ` / `compliance` / `tính pháp lý` / `evaluate` /
  `đánh giá tuân thủ|tính pháp lý|mức độ …`). The bare generic `đánh giá`
  keyword is deliberately absent, keeping the Task-3 ruling intact (a plain
  "đánh giá chung về …" question still reaches the model path).
  `IntentClassifier.classify` runs it after `classify_deterministic` and
  before the model.
- `backend/app/services/agents/v2/nodes/routing.py`: `_INTENT_ANALYSIS`
  maps `"evaluate" → ("evaluate", "document")`.

Proof (TDD, RED verified):

- `test_classify_evaluate_deterministic_scope` — positive + negative cues.
- `test_classify_evaluate_short_circuits_model` — no model call.
- `test_route_node_compliance_with_classifier_wired` — production shape:
  real `IntentClassifier` wired, a fake model that *would* emit `search`;
  asserts `complex_research/compliance_evaluation` and `provider.called is
  False`. RED before the fix: `work_type == "retrieve"` (targetless).

### I2 — Phase-4B document-identity adapter was dead code (FIXED)

`DocumentIdentityResolver` / `resolve_draft_identities` had no production
call site.

Fix (reviewer option "wire the resolver"):

- `backend/app/services/agents/supervisor_v2.py`: `DeterministicSemanticAdapter`
  gained optional `session_factory` + `workspace_ids`; `build_draft` now calls
  `resolve_draft_identities` for still-unresolved refs when the
  resolver/session/scope are wired (already-resolved preprocessor/ui/api
  refs pass through; the v2 binding resolver stays the only revision-pin
  authority). Unwired construction sites stay inert (default `None`).
- `backend/app/services/agent/runtime_selector.py`: `build_v2_ingress` wires
  `identity_resolver=DocumentIdentityResolver()` + session + trusted scope,
  and the duplicated "No identity_resolver here" comment block was removed.

Proof:

- `test_build_draft_wires_identity_resolution_when_configured` — spy proves
  `resolve_draft_identities` is invoked with the wired resolver/scope/session.
- `test_build_draft_skips_identity_resolution_without_wiring` — negative pin.

### I3 — Phase-4C coreference/focus has no runtime consumer (RELABELED AS INCOMPLETE)

Per the reviewer's explicit option, this is converted to an explicit
incomplete-status entry in the ledger rather than a passing task. The
end-to-end consumer (history → ACL-checked `KnownDocumentResource(source=
"conversation", …)` resources, or a coref-driven finalizer projection) is
pre-canary hardening work for a future approved plan, not this wave.

### I4 — typed `personal` routed to `complex_research` (FIXED)

`personal` was removed from `_INTENT_ANALYSIS` (`nodes/routing.py`) so it
falls back to the legacy deterministic fast path (BASE behavior) until a
frozen-reason ruling records typed `direct`. Route-level pins:
`Tôi là ai?` → `fast_domain/simple_kg_lookup`;
`Tôi đang công tác tại đâu?` → targetless fast path.

Proof: `test_typed_personal_falls_back_to_legacy_fast_path`,
`test_route_node_typed_personal_served_fast` (RED verified).

### Mod1 — frozen `RouteReason` extension unauthorized/undocumented (FIXED)

- Recorded `targetless_document_retrieval` as an approved erratum in the
  contract spec + Phase-0 mirror (`orchestrator_compat/frozen_contracts.py`).
- The three frozen-corpus targetless cases now carry
  `v2_reason_code="targetless_document_retrieval"`; `rag-general-compare`
  gains a real `decide_route` assertion (the previously-missing Phase-4A gate
  pin).

### Mod3 — frontend type-check red (FIXED)

The 6 new `TS2339` errors in
`frontend/src/hooks/__tests__/useRAGChatStream.contract.test.tsx` came from
the `let final: ChatMessage | null = null;` + closure-assignment narrowing.
Rewrote to capture the `sendMessage` result through `act` (and dropped the
now-unused `ChatMessage` import). `tsc --noEmit -p tsconfig.app.json` is back
to the BASE 8 pre-existing errors; `vitest run` stays 28/28.

### Mod5 — typed path bypassed the v1-owned write boundary (FIXED)

The typed-intent early return in `analyze_query` now re-runs `_write_intent`,
so a write request misclassified to a read intent keeps its `write` domain
(`simple_write_operation`). Proof: `test_typed_search_preserves_write_boundary`.

## Descoped / ledgered (no code change, per reviewer allowance)

- **Mod2** — no browser/component E2E harness exists in-repo; the
  event-inventory artifact was not produced. Descoped.
- **Mod4** — 2 of 3 Phase-4A gate cases are model-dependent and there is no
  recorded-model offline harness. Descoped.
- **Mod6** — shadow arm omits classifier/planner/replanner (shadow sampling
  off by default; evidence-quality note). Ledgered.
- **Mod7** — v1 wire-shape normalization at the session relay; accepted under
  the one-contract canary decision, flagged for the rollout runbook.
- **Minor 1–11** — remain ledgered with owners in `final-review.md`.

## Verification evidence

- `tests/agents/v2` minus `persistence`/`orchestrator_compat`:
  **958 passed / 3 skipped** (was 948/3 at FIX_BASE; +10 new tests).
- `tests/api/test_agent_canary_selection.py` + `tests/agents/test_v1_clarification_emit.py`
  + `tests/test_chat_public_contract_ddl.py` + `tests/api/test_agent_v2_ingress.py`:
  **79 passed**.
- Frontend `vitest run`: **28/28 passed**.
- Frontend `tsc --noEmit -p tsconfig.app.json`: **8 errors** (all pre-existing
  BASE errors in `ChatPanel.rollback.integration.test.tsx` / `AdminLLMConfigPage.tsx`);
  the 6 new `useRAGChatStream.contract.test.tsx` errors are gone.
- GitNexus `impact` (stale index at `ed7386b`, documented blast radius):
  `analyze_query` → 1 direct (`route_node`), LOW; `DeterministicSemanticAdapter`
  → 28 (4 direct), LOW.
- GitNexus `detect-changes` (working tree): 13 files / 13 symbols, 0 affected
  processes, risk **low**.

## Commit

- FIX_BASE `c21f63f` → final fix HEAD (this commit), subject:
  `fix(v2): final fix wave — evaluate reachability, resolver wiring, personal fallback, contract erratum`
- `package.json` / `package-lock.json` are pre-existing uncommitted residue
  (playwright + catppuccin) and were **not** staged.

## Concerns

1. **I3 remains end-to-end incomplete** (Phase 4C multi-turn coreference has
   no runtime consumer) — relabeled, not delivered. Requires ACL-checked
   `source="conversation"` history projection or an equivalent finalizer
   consumer before the Phase-4C production-parity gate.
2. The `evaluate` deterministic narrow scope is a new v2-only classifier; its
   cue vocabulary is conservative but not exhaustively fuzzed against the
   full Vietnamese query space. The bare-`đánh giá` exclusion preserves the
   Task-3 ruling at the cost of not short-circuiting "đánh giá chung về …".
3. I2 wiring makes `build_draft` perform v1 resolver work (DB + optional LLM +
   vector) on turns with unresolved refs; the resolver's per-request cache
   bounds it to one resolution per turn, and the D3 read-only contract is
   preserved, but first-turn latency for named-document queries now includes
   that resolution.
