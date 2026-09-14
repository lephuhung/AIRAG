# Live Phone-Lookup Fix Report (SDD)

- Base: `9ded201` (`fix(v2): persist document evidence revision`), worktree `/home/AIRAG/.worktrees/langgraph-v2`, engine Muse Spark 1.3 only.
- Commit: `fix(v2): route phone lookups to mongo` (narrow; task-owned files only).

## 1. Live defect

Authenticated `Tra cứu số điện thoại 0989755968` produced `QueryAnalysis(work_type=retrieve, domains=(document,))`
and dispatched a T1 `document.retrieve`. Direct Mongo `search_by_phone` finds **3 distinct person groups /
3 records** (schemas `bhxh`/`vnvc`). The user report (wrong/unverifiable) is correct.

## 2. Root cause

- `analyze_query` (`backend/app/services/agents/v2/nodes/routing.py`) derived the `people` domain **only**
  from `semantic.person_refs`, but `draft_from_preprocessing` always emits `person_refs=()`. A phone-number
  query therefore fell through to the ref-less default `domains={"document"}` → `retrieve` → complex
  `document.retrieve`. RED test proved it: `assert 'retrieve' == 'lookup'` failed pre-fix.
- `V1PeopleLookupService.lookup` always ran `search_by_name` and returned the **first** person only.
- `PeopleCapability.execute` persisted a **single** `EvidenceUse` (first-only shape).

## 3. Fix (minimal, no new scheduler / no domain agents)

1. **Routing** (`v2/nodes/routing.py::analyze_query`): in the ref-less fallthrough only, consult the
   deterministic `classify_supervisor_scope(normalized_query)`; `== "people"` → `domains={"people"}`
   (→ `lookup` → `fast_domain/simple_people_lookup` → T1 `people.lookup`). No regex duplicated.
   Document/section refs keep document semantics authoritative by construction (branch unreachable with refs),
   verified by `test_explicit_doc_phone_query_stays_document`.
2. **Adapter** (`agents/supervisor_v2.py`): new internal `PeopleLookupMatch(record_id, fields,
   required_fields)`; public `canonicalize_person_record` (bhxh `hoTen`/`soDienThoai`, vnvc
   `fullName`/`mobile`, … aliases → only `{name, phone, source}`; unrelated DOB/address/CCCD/BHXH dropped;
   already-canonical `name`/`phone` keys accepted so legacy name behavior is preserved);
   `stable_people_record_id` (`sha256(name|phone|source)[:16]`, non-PII, deterministic);
   `V1PeopleLookupService._lookup_many` dispatches by `people_intent_from_query`
   (phone→`search_by_phone`, CCCD→`search_by_cccd`, BHXH→`search_by_bhxh`, name→`search_by_name`,
   advanced→name fallback), dedupes grouped records (`_person_group` → identity key), one match per
   distinct person. Legacy `lookup()` contract is byte-identical (shadow R64 surface still `["lookup"]`).
3. **Capability** (`v2/capabilities/people.py`): `execute` probes the private multi-match seam and, when
   present, persists **one governed `EvidenceUse` per distinct person** (shared `acquisition_id`,
   `supporting`, `target_id=None`); empty → typed `not_found` (no document fallback — the fast plan owns
   a single task, so neither `document.retrieve` nor `document.search` can dispatch); malformed →
   `CONTRACT_MISMATCH`; ACL (`can_read_people` + `allowed_capabilities`) checked before any dispatch.
   No `Any`/`dict` annotations added (Task-2 AST guardrail stays green).

Flow preserved: proposal → validate → lease → checkpoint → shared `TaskScheduler` → capability.
Synthesis needed no change: extractive draft hydrates all admitted uses and cites all 3 people with sources.

## 4. Tests (TDD: RED → GREEN)

New: `backend/tests/agents/v2/fast_paths/test_phone_lookup_fix.py` (9 tests; `git add -f` — `backend/tests/` is gitignored but tracked files are force-added by convention):

| Test | Proves |
|---|---|
| `test_live_phone_query_routes_people_lookup_not_document` | RED→GREEN route fix (`lookup/people`, `fast_domain`, T1 `people.lookup`) |
| `test_live_phone_end_to_end_returns_all_three_people` | `search_by_phone` ×1, doc provider ×0, 3 distinct uses/hydrations/citations, all 3 names answered, no raw PII in checkpoint/evidence content |
| `test_phone_capability_uses_search_by_phone_not_search_by_name` | phone intent never runs name search |
| `test_phone_not_found_has_no_document_fallback` | `not_found` → `insufficient`, zero doc dispatch |
| `test_phone_lookup_denied_without_can_read_people` | strict ACL, service untouched |
| `test_explicit_doc_phone_query_stays_document` | scoped doc + phone text stays document path |
| `test_alias_normalization_and_grouped_dedupe` | bhxh/vnvc aliases, `_person_group` dedupe, stable non-PII `record_id`, fields ⊆ `{name,phone,source}` |
| `test_cccd_bhxh_dispatch_and_name_preserved` | CCCD/BHXH dispatch + legacy name route/behavior |
| `test_people_evidence_classification_floor_is_personal` | governor floor `personal` for `{name,phone,source}` (no sensitive fields stored) |

Mutation (all KILLED): M1 first-only service → 1 use ≠ 3; M2 forced document route → `domains==("document",)`;
M3 phone intent miswired to name search → phone spy silent, name spy called.

## 5. Verification

- New file: 9 passed. `fast_paths/`: 100 passed.
- Full `backend/tests/agents/v2/`: **1040 passed**, 4 skipped; 2 failures + 1 collection-ignore are the **known
  checkpoint-dependency environment gap** (`langgraph-checkpoint-postgres` not installed in this interpreter:
  `persistence/test_checkpoint.py` collection, `test_complex_subgraph_does_not_open_its_own_checkpointer`
  CWD-relative path read, compat-probe `AsyncPostgresSaver` import) — all untouched by this task, no
  checkpoint/persistence code changed.
- API + services (`test_agent_v2_ingress/streaming/canary_selection`, `test_governor_evidence_revision`,
  `backend/tests/services/`): 127 passed.
- No PII logging added: canonical evidence holds `{name, phone, source}` only; `record_id` is a hash;
  checkpoint JSON asserted free of DOB/address/BHXH/CCCD values and raw Mongo keys.

## 6. Impact analysis (GitNexus MCP unavailable in this runner — manual upstream review)

- `analyze_query` (LOW): additive branch in the ref-less fallthrough; callers (`route_node`, routing/compat
  tests) unaffected for ref-carrying queries. Full agents/v2 suite green.
- `V1PeopleLookupService.lookup` (LOW, dynamic): signature and legacy body unchanged; shadow wrapper +
  R64 surface test green.
- `PeopleCapability.execute` (LOW, dynamic): legacy single path byte-identical; multi path only for services
  exposing the new seam; capability-boundary/AST guardrails green.
- No other existing symbol edited.

## 7. Residual risks / notes for DeepSeek review

- Real-Mongo alias coverage beyond bhxh/vnvc (evn/lg/vacxin/cv19/uids) follows the same alias table but is
  stub-verified only; live BHYT/CCCD-adjacent fields intentionally excluded from evidence.
- `sensitive_personal` classification: stored fields (`name/phone/source`) floor at `personal` by the
  deterministic governor; nothing sensitive is persisted, so no `sensitive_personal` case arises by design.
- Protected dirt preserved untouched: `AGENTS.md`, `CLAUDE.md`, `backend/app/api/rag.py`,
  `backend/app/queue/publisher.py`, `backend/tests/workers/test_revision_pipeline.py`,
  `docker-compose.vllm.yml`, `backend/requirements-v2-benchmark.txt`, `docker-compose.v2test.yml`,
  `docs/reports/` (this report left untracked). No deploy/restart; vLLM untouched.
