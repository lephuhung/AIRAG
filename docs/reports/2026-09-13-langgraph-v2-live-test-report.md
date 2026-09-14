# LangGraph v2 — Live API Test Report

**Date:** 2026-09-13  
**Environment:** local Docker test stack (`hrag-backend`)  
**Authentication:** JWT obtained through `/api/v1/auth/login` for the user-supplied test account. Credentials and tokens are not recorded in this report.

## Scope

Verify real LangGraph v2 behavior over authenticated SSE requests, before proposing any further repair plan.

## Environment verification

- `hrag-backend` is healthy and compiles `supervisor_v2` against the shared checkpointer.
- v2 is enabled with 100% canary configuration in the test environment.
- The API account has access to the `Luật` workspace (7 indexed documents, 2,044 chunks) and `Nghị định` workspace (2 indexed documents, 301 chunks).

## Repairs made before live probes

`backend/app/services/agents/semantic_preprocessor.py`

1. Added missing `Document` import in `_resolve_abbreviations_db`.
2. Added missing SQLAlchemy `func` import in `_lookup_by_alias`.
3. Replaced illegal mutation of frozen `AbbreviationEntry` with immutable `model_copy` results.
4. Corrected `abrr_list` → `abbr_list` in LLM abbreviation fallback.

Regression coverage added at:

- `backend/tests/agents/test_semantic_preprocessor_db_resolution.py`

Validation after these repairs: 60 focused tests passed; static undefined-name check passed.

## Live probe results

| Probe | Workspace | SSE result | Finding |
|---|---|---|---|
| `NĐ là gì?` | Luật | 2 `status`, 1 `complete` | No silent response. `NĐ` is treated as an unresolved document reference and returns the clarification `Unresolved references: NĐ`. |
| Generic multi-document compare | Luật | 1 `status`, 1 `error` | Complex route reaches finalizer without a checkpointed `EvidenceEvaluation`; finalizer returns a generic error. |
| Compare using two multi-word law titles plus scoped `document_ids` | Luật | 2 `status`, 1 `complete` | Resolver extracts only `Luật An` / `Luật Bảo`; title grammar cannot bind multi-word document names. Scoped API attachments are intentionally not auto-bound. |
| `53/2022/NĐ-CP quy định những nội dung gì?` | Nghị định | 1 `status`, 1 `error` | Preprocessor emits overlapping document spans `(0,13)` and `(3,13)`; frozen `PreprocessingResult` validation rejects the state before routing. |

## Confirmed defects

1. **Overlapping document-reference extraction**: multiple extraction patterns can emit overlapping spans for one document number; no deduplication/precedence occurs before `PreprocessingResult` validation.
2. **Multi-word title extraction is incomplete**: the named-document grammar only captures one token after document-type keywords, preventing binding of ordinary Vietnamese law titles.
3. **Complex no-plan outcome is surfaced as a generic error**: a complex query without valid bound targets produces no plan/evaluation. The parent finalizer then rejects the missing evaluation instead of producing a typed clarification/insufficient outcome.
4. **v2 document discovery cannot admit indexed legacy documents without a revision identity**: in the known searchable `Luật` workspace, v1 retrieval returned three sources for one valid document ID, but `document_views.load_current_revision_identity()` returned `None`. The v2 adapter correctly fails closed, yielding no discovery candidate and preventing a factual plan from reading the corpus.

## Authenticated route/node matrix

`GET /auth/me` confirmed that the supplied API principal is active and `is_superadmin=true`. Every request below used a fresh JWT from this principal and a workspace inside its accessible scope.

| Route / component | Authenticated probe | SSE / HTTP outcome | Finding |
|---|---|---|---|
| Direct greeting | `Xin chào` | 2 `status`, 1 `token`, 1 `complete(success)` | Direct node is healthy. |
| Direct conversation | `Cảm ơn` | 2 `status`, 1 `token`, 1 `complete(success)` | Direct node is healthy. |
| Ref-less factual / user-reported case | `Mạng LAN trong BMNN quy định như thế nào` | 1 `status`, 1 terminal `error` | Reproduces the user-reported failure while authenticated as superadmin. Finalizer lacks a checkpointed `EvidenceEvaluation`. |
| Knowledge graph | `Tài liệu nào thuộc đơn vị nào?` | 1 `status`, 1 terminal `error` | The route reaches factual processing but returns typed `Đã xảy ra lỗi khi thu thập bằng chứng.` Evidence/capability admission needs separate tracing. |
| Write | `Soạn báo cáo ngắn về an toàn mạng` | 1 `status`, 1 terminal `error` | Expected current v2 limit: typed `Thao tác viết (write) chưa được hỗ trợ trong v2 giai đoạn này.` |
| Retrieval API (not LangGraph) | BMNN question in `Luật` workspace | HTTP 200; 5 chunks and 5 citations | Retrieval provider/corpus/ACL work. A later capability-level trace shows v2 runs retrieval too, but drops its sources because the matched document has no v2 current-revision identity. |
| v1 A/B admin harness | BMNN question, arm `v1` | HTTP 500 | Separate harness defect: `sources_accumulator` reads missing `ChatSourceChunk.chunk`, raising `AttributeError`. This does not establish a v1 answer-quality baseline. |

## Additional implementation coverage gaps

- `draft_from_preprocessing()` always writes `person_refs=()` and `section_refs=()`. Therefore a user request cannot currently route to `people.lookup` or `section.read` through the v2 semantic adapter, regardless of superadmin permission.
- In the live SuperAdmin runtime (388 authenticated workspaces, `can_read_people=true`), the registry exposes only `document.search` and `people.lookup`. `abbreviation.resolve` is excluded as `service unavailable`; the other declared capabilities are likewise absent from this deployment's request-scoped registry. Therefore runtime deployment gates, as well as semantic/binding reachability, must be checked before treating a capability as live.
- The admin A/B driver cannot be used as a gate until `sources_accumulator` handles the current `ChatSourceChunk` contract.

## Evidence-based repair themes (not yet a plan)

1. Make semantic extraction produce non-overlapping, bindable document references for real Vietnamese document titles and official number formats.
2. Define and test terminal behavior when complex planning has no valid target/plan, instead of flowing a missing evaluation into finalizer failure.
3. Trace and repair evidence admission for the knowledge-graph factual path.
4. Repair the v1 admin-evaluation source accumulator contract so A/B regression testing is usable.
5. Expose or deliberately defer People/Section extraction; capability grants alone do not make those paths testable.

No implementation plan for these themes has been proposed yet; it will be based on this completed authenticated matrix.

## Authenticated capability-level probes

To distinguish individual capability behavior from graph orchestration, a temporary in-container probe constructed `build_v2_ingress()` with the JWT-derived SuperAdmin user identity, all 388 authenticated workspace IDs, and `can_read_people=true`. It printed only aggregate/result status and rolled back the Evidence UoW.

| Capability / gate | Input | Result | Finding |
|---|---|---|---|
| Capability registry | SuperAdmin runtime, 388 workspace IDs | `document.search`, `people.lookup` only | The live request-scoped registry does **not** expose the full declared capability set. `abbreviation.resolve` explicitly raises `CapabilityUnavailable(service)`. |
| `document.search` | `Mạng LAN trong BMNN` across all authorized workspaces | `not_found`, 0 candidates, 0 evidence uses | The capability executes but produces no candidate. The scoped boundary probe below establishes this is a revision-identity admission failure, not an ACL or generic retrieval failure. |
| `people.lookup` | Non-matching sentinel query | `not_found`, `matched=false`, 0 evidence uses | Capability is correctly permission-admitted for SuperAdmin and fails safely. Some underlying schemas time out after 5s and are skipped, indicating a performance/indexing risk but no terminal capability error. |
| `document.read`, `section.read`, `knowledge_graph.query`, `memory.lookup`, `abbreviation.resolve` | Registry-only gate probe | each raises `CapabilityUnavailable(service)` | These capabilities are declared in code but not deployed into the live request-scoped registry. No provider invocation was attempted. In particular, the prior KG UI error cannot yet be diagnosed as a fault *inside* `knowledge_graph.query`: the supported capability path is currently gated out. |

These probes did not expose People records and did not persist evidence.

### Document-search boundary trace

The same JWT-derived authenticated scope was intersected with the single `Luật` workspace (confirmed to be within scope). The result remained `not_found`; therefore it is not caused by the 388-workspace fan-out. A direct, read-only trace of the exact adapter boundary then measured:

| Boundary | Aggregate observed result |
|---|---|
| `tools.search_documents` | 3 raw sources, all with a document ID; 1 unique valid document ID |
| `document_views.load_current_revision_identity` | 0 identities loaded; 1 identity missing; 0 `RevisionNotReady`; 0 unexpected errors |

This proves the adapter's candidate-dropping behavior is its designed pinning guard firing on corpus data that lacks a v2 revision identity. It must be resolved before authenticated factual routes can read that document through v2.

## Confirmed operating constraint

The operator confirmed that the deployment has already moved to **v2-only serving** and cannot return factual traffic to v1. A v1 fallback is therefore out of scope and would also violate the v2 immutable revision-pinning contract.

The required recovery path is a controlled, revision-aware reindex of the existing corpus: it must create and publish complete revision artifacts/build manifests before setting `Document.current_revision_id`. Merely assigning a revision pointer or repurposing legacy vector/object artifacts is unsafe and unsupported. Until a document has a published v2 revision, factual requests must terminate as a typed unavailable/insufficient outcome rather than a generic SSE error or an unpinned answer.
