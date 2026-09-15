# Follow-up fix report — LangGraph v2 Phase 4 (final re-review I5 + M8 + residue cleanup)

Sole-writer follow-up to the final re-review (`68e4a0d7…_output.md`, VERDICT FIX_REQUIRED scoped to I5/M8).

- BASE: `6662151` (`fix(v2): final fix wave — evaluate reachability, resolver wiring, personal fallback, contract erratum`)
- HEAD: `fix(v2): narrow compliance evaluation intake` (this commit)

## Scope

Only I5 (tighten the v2-only deterministic `evaluate` scope), M8 (record the ruling/erratum), and the approved workflow-residue cleanup (restore root `package.json`/`package-lock.json` to HEAD). No other prior finding re-opened.

## Changed files

- `backend/app/services/agents/v2/semantic/intent.py` — split the broad `_EVALUATE_SCOPE_RE` into `_EVALUATE_HEAD_RE` + `_EVALUATE_DOMAIN_RE` + `_EVALUATE_LEVEL_RE` + `_EVALUATE_YN_RE`; `classify_evaluate` now requires an assessment request.
- `backend/tests/agents/v2/test_intent_translation.py` — added `document.retrieve` to `FULL_CAPABILITIES`; added `test_classify_evaluate_informational_topic_not_assessment` and parametrized `test_route_node_informational_compliance_topic_stays_targetless`.
- `docs/superpowers/plans/2026-09-14-langgraph-v2-phase4-semantic-adaptive-execution.md` — Task 3 erratum (M8).
- `docs/superpowers/specs/2026-09-14-langgraph-v2-semantic-adaptive-execution-design.md` — §4.1 erratum (M8).
- `.superpowers/sdd/…/progress.md` — ledger follow-up entry (M8 + residue).

## I5 implementation (assessment-request cue split)

`classify_evaluate` now returns `evaluate` only for an explicit compliance/legal **assessment request**:

1. assessment head (`đánh giá`/`kiểm tra`/`rà soát`/`thẩm định`/`đối chiếu`/`xác định`/`review`/`assess`/`evaluate`) **AND** domain cue (`tuân thủ`/`compliance`/`tính pháp lý`/`pháp lý`); or
2. degree/level phrase (`mức độ tuân thủ`/`mức độ rủi ro`); or
3. yes/no form (`…tuân thủ … không?`).

Bare compliance-topic nouns and bare `đánh giá` return `None` (model path). The shared v1 taxonomy prompt is unchanged.

## RED (against HEAD `6662151`, before implementation)

```
FAILED tests/agents/v2/test_intent_translation.py::test_classify_evaluate_informational_topic_not_assessment
FAILED tests/agents/v2/test_intent_translation.py::test_route_node_informational_compliance_topic_stays_targetless[quy định về tuân thủ thuế là gì?]
FAILED tests/agents/v2/test_intent_translation.py::test_route_node_informational_compliance_topic_stays_targetless[chế tài xử phạt khi không tuân thủ quy định?]
FAILED tests/agents/v2/test_intent_translation.py::test_route_node_informational_compliance_topic_stays_targetless[hồ sơ tuân thủ gồm những giấy tờ gì?]
4 failed, 36 deselected in 1.33s
```
Representative failure: `assert 'evaluate' == 'retrieve'` (route test line 572) — the pre-fix broad scope classified `quy định về tuân thủ thuế là gì?` as `evaluate` instead of `retrieve`.

## GREEN (after implementation)

- `tests/agents/v2/test_intent_translation.py`: **40 passed** (36 + 4 new).
- `tests/agents/v2` minus `persistence`/`orchestrator_compat`: **962 passed / 3 skipped** (was 958/3).
- `tests/api/test_agent_canary_selection.py` + `tests/agents/test_v1_clarification_emit.py` + `tests/test_chat_public_contract_ddl.py`: **49 passed**.
- Focused routing + parity: `test_targetless_retrieval.py` + `golden/test_v1_intent_parity.py` + `test_intent_adapter.py`: **34 passed**.
- Manual cue-split probe: all positive assessment forms → `evaluate`; all informational negatives → `None` (see report concern section).

## GitNexus

- `impact classify_evaluate`: 1 direct (Semantic module), risk LOW.
- `impact IntentClassifier`: 24 impacted / 2 direct, risk LOW.
- `detect-changes --scope all`: 4 files / 13 symbols, 0 affected processes, risk **low**.

## Residue cleanup evidence

`git checkout -- package.json package-lock.json`; after restore `git status --short` shows only the four intended modified files (no `package.json`/`package-lock.json`). `frontend/package*` untouched.

## Self-review

- Negative pin exists and fails pre-fix (RED captured above).
- Positive evaluate cues unchanged (`Đánh giá tuân thủ…`, `Đánh giá tính pháp lý…`, `Tôi có tuân thủ … không?`, `Kiểm tra mức độ tuân thủ…`, plus `mức độ tuân thủ …` alone).
- Model advisory boundary preserved: informational queries consult the model (`provider.called is True`); evaluation queries short-circuit deterministically (`provider.called is False`); `decide_route` stays the route authority in both.
- The v1 shared taxonomy prompt is not modified.

## Residual risks

1. The cue vocabulary (heads/domains) is heuristic; pathological Vietnamese phrasing could still straddle assessment vs topic, but the split now requires an assessment head/degree/yes-no form, which is far narrower than the prior bare-noun match.
2. The `…tuân thủ … không?` yes/no window is bounded to 80 chars between `tuân thủ` and `không?`; longer sentences would fall through to the model path (advisory `search` → targetless) rather than evaluate — fail-open toward the fast path, not a security issue.
3. Not re-opened (pre-canary hardening per the re-review): M9 (real resolver chain unpinned), M10 (per-build DB session + undeadlined resolver work), Minor 1/2/3/5.
