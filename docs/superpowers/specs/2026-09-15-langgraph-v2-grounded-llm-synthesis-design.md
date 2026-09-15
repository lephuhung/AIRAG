# LangGraph v2 — Grounded LLM Answer Synthesis Design

**Date:** 2026-09-15
**Status:** Reviewed — ready for implementation planning
**Scope:** V2 document-backed factual answer synthesis, claim grounding, public citations, checkpoint recovery, privacy-safe tracing, terminal persistence, and production rollout

## 1. Goal

LangGraph v2 must turn admitted retrieval evidence into a concise, natural answer produced by the configured main LLM. It must not expose the current deterministic extractive draft—which concatenates retrieved legal chunks—as the normal production answer.

The generated answer must retain the authority boundaries already established in v2:

- V2 alone owns ACL, immutable revision binding, scheduling, evidence admission, evidence selection, grounding, public citation identity, checkpointing, and terminal status.
- The LLM is a bounded drafting component. It cannot authorize data, select tools, mutate plans, invent trusted identity, choose public citation IDs, or bypass validation.
- Every material factual claim must reference admitted current-run evidence through server-issued ephemeral handles.
- Server-side validation proves claim-to-evidence traceability and enforces deterministic high-risk factual consistency; it does not claim to be a formal semantic-entailment proof.
- No answer token crosses the public boundary until the complete candidate has passed schema validation, handle resolution, claim validation, citation projection, and server-side rendering.

The required user-visible result for a query such as:

> Đăng tải thông tin sai sự thật bị xử lý như thế nào?

is a concise Vietnamese explanation with directly supporting inline citations—not a dump of retrieved chunks and not an uncited model answer.

---

## 2. Current problem

The current production path is:

```text
sufficient EvidenceEvaluation
  -> hydrate SynthesisEvidence
  -> build_extractive_draft()
  -> ground_answer()
  -> FinalResponse
```

`DraftBuilder` exists as a protocol, but production callers do not inject an implementation. `synthesize_answer(..., draft_builder=None)` therefore uses `build_extractive_draft()`, whose output preserves source text rather than synthesizing an answer.

The overlap-grounding repair in commit `1d80b62` correctly prevents duplicated legal chunks from becoming spuriously ambiguous, but it intentionally does not add LLM composition. The observed result is grounded enough for the current deterministic contract yet still reads like raw provisions.

There are also four implementation constraints the LLM synthesis design must address explicitly:

1. the current generic main-LLM tracing wrapper captures full prompt and output content;
2. the current grounding path maps rendered `AnswerDraft.content` back to claims by string matching rather than grounding claims directly;
3. current synthesis budget splitting is FIFO and may starve one target in compare/multi-target work;
4. current public citation contracts are document-centric, while People and KG evidence have different or incomplete presentation semantics.

V1 already provides the target document-citation presentation behavior:

- each locatable source has a unique four-character alphanumeric index;
- the answer places markers such as `[a3z9][b2m7]` immediately after the sentence they support;
- the SSE stream supplies locatable source metadata;
- the frontend resolves each marker into a clickable citation badge and opens the corresponding document/chunk.

V2 must preserve that user experience while retaining stronger server-owned evidence and identity controls.

---

## 3. Approved decisions

The following decisions are fixed for this design:

1. **Structured claim generation:** the LLM returns a strict JSON claim proposal, not free-form prose with self-issued citation IDs.
2. **Main model configuration:** synthesis uses the effective `main` LLM connection/model from existing runtime configuration. It does not add a new model role or admin configuration surface.
3. **Privacy-safe synthesis tracing:** synthesis must not use the current generic full-content Langfuse/dataset tracing behavior unchanged. The synthesis call records content-free operational telemetry only.
4. **Stable evidence-handle manifest:** `E1..En` are backed by a checkpointed server-owned mapping to exact `EvidenceUseRef` identities before any model call. Handles are never re-numbered to different evidence on resume.
5. **Target-aware evidence selection:** prompt evidence is selected deterministically across required targets; FIFO truncation cannot silently starve a compare/multi-target branch.
6. **Claim-first grounding, render-last:** claims are validated and grounded before Markdown/citation rendering. The server never renders prose and then parses it backward to rediscover claim identity.
7. **Deterministic high-risk factual guards:** quantities, dates, legal locators, official document numbers, and comparable literal anchors in generated claims must be supported by the cited evidence after canonical normalization.
8. **Validate before stream:** the complete candidate is buffered, parsed, grounded, citation-projected, and rendered before the first answer token is emitted.
9. **One repair attempt:** at most two model calls are allowed—one initial attempt and one bounded repair attempt.
10. **No production extractive fallback:** after both attempts fail, the turn fails closed with a safe typed terminal error. Raw chunks are not returned as a fallback answer.
11. **Document-backed citation scope first:** this phase enables LLM synthesis only where every factual claim can resolve to an authorized document-backed public citation. People remains on its existing presentation path; KG synthesis is deferred until a first-class public KG locator/citation contract exists.
12. **Single citation projection owner:** one server component resolves governed evidence identities into public locatable citation metadata used by rendering, SSE, persistence, and reload.
13. **Existing V2 rollout control:** rollback uses the existing V2 canary/arm control; no synthesis-specific traffic flag is introduced.

---

## 4. Non-goals

This work does not:

- change People-record answer presentation; People evidence has a distinct privacy/card contract and is not sent to the answer-synthesis model in this phase;
- introduce KG LLM synthesis before a public KG citation/navigation contract exists;
- add a second independent LLM judge;
- claim formal semantic entailment between arbitrary natural-language claims and evidence;
- move ACL, binding, revision pinning, planning, routing, scheduling, evidence admission, or public citation ownership into the model;
- give the model internal evidence UUIDs, document UUIDs, workspace IDs, run IDs, task IDs, binding IDs, revision IDs, or checkpoint IDs;
- stream speculative text followed by `token_rollback`;
- add a new model role or admin configuration surface;
- reintroduce V1 supervisor control fields or a second agent graph;
- change direct greeting, clarification, denied, or evidence-insufficient routes into model calls;
- treat the current raw-concatenation overflow artifact as a semantic summary merely because it is stored as derived evidence;
- repair unrelated OTLP exporter or Telegram configuration;
- restart or reconfigure vLLM as part of implementation.

---

## 5. Canonical architecture

```text
Request / conversation
        |
        v
V2 semantic + deterministic route
        |
        v
V2 plan / scheduler / capabilities
        |
        v
EvidenceEvaluation == sufficient
        |
        v
Governed hydration under current ACL/revision/retention
        |
        v
Server PresentationPolicy
        |-------------------------------> People/card path (no synthesis LLM)
        |
        v
Document-backed synthesis eligible?
        | no -> typed presentation/synthesis unavailable
        | yes
        v
Target-aware SynthesisEvidenceSelector
        |
        v
checkpoint stable HandleManifest
E1 -> exact EvidenceUseRef(A)
E2 -> exact EvidenceUseRef(B)
...
        |
        v
reserve attempt in checkpoint (attempts_started += 1)
        |
        v
StructuredLLMDraftBuilder
(effective main model; privacy-safe tracing; buffered)
        |
        v
checkpoint parsed claim proposal/error code
        |
        v
resolve E-handles through SAME HandleManifest
        |
        v
claim schema + high-risk factual validation
        |
        v
GroundedClaim[]
        |
        v
CitationProjector
(document / derived->document lineage only)
        |
        v
server renders Markdown + [a3z9] markers
        |
        v
checkpoint grounded artifact
        |
        v
citation SSE -> token SSE -> complete
```

The graph remains the only orchestration authority. The LLM adapter is a request-scoped runtime service. Evidence selection, handle identity, grounding, citation projection, and rendering remain server-owned.

---

## 6. Presentation policy and synthesis eligibility

### 6.1 Server-owned presentation strategy

A deterministic server policy selects presentation behavior from the checkpointed route/plan plus admitted evidence kinds. The model never chooses this strategy.

Conceptually:

```text
PresentationMode
  document_grounded_llm
  people_card
  direct
  typed_unavailable
```

Rules for this phase:

- document retrieval, document section reads, document summarize final/reduce output, and complex final answers whose supporting claims can all resolve to document-backed sources -> `document_grounded_llm`;
- People lookup -> existing `people_card`/public People transport; no synthesis model call;
- direct/clarification/denied/insufficient -> existing non-LLM presentation;
- KG-only or mixed evidence that requires a KG claim without a document-backed locatable lineage -> typed unavailable for LLM synthesis in this phase, not a fake document citation;
- derived evidence is eligible only when every cited derived use can recursively project to currently authorized locatable document lineage.

A complex workflow may use People internally to discover documents. That does not force People records into the synthesis prompt: only final admitted evidence selected to support user-visible claims crosses the synthesis boundary.

### 6.2 Why this boundary is required

The supervisor graph currently sends factual fast-path work through `execute -> evaluate -> synthesize -> ground -> finalizer`. The production implementation must therefore branch presentation **inside the owned synthesis/presentation boundary** rather than assuming every sufficient factual result is document-synthesizable.

A missing/unsupported presentation strategy is a typed failure or existing specialized presentation—not permission to call `build_extractive_draft()`.

---

## 7. Runtime service boundary

### 7.1 Structured builder

Add a request-scoped synthesis service to `RuntimeServices`, conceptually:

```python
answer_draft_builder: StructuredLLMDraftBuilder | None
```

Production ingress constructs exactly one builder using the **effective `main` configuration**, but not the existing generic full-content tracing wrapper unchanged.

A suitable implementation boundary is conceptually:

```python
get_main_provider_for_synthesis()
```

This is **not** a new LLM role. It resolves the same effective `main` provider/model/API configuration, but applies a synthesis-specific tracing policy where prompt/evidence/output content is not exported to Langfuse or the dataset trace collector.

Nodes do not instantiate providers directly.

### 7.2 Privacy-safe tracing requirement

Current generic tracing serializes full `messages`, `system_prompt`, and output. Because synthesis messages contain hydrated evidence plaintext, that behavior is forbidden for this call path unless a future separately approved policy explicitly changes it.

Synthesis tracing may record only allowlisted operational metadata such as:

- role=`main`;
- provider/model name;
- input evidence count;
- input character/token estimate;
- output claim count;
- latency;
- usage totals;
- outcome/failure code;
- repair attempted;
- cancellation/deadline result.

It must not record:

- raw query text when policy treats it as content;
- raw evidence text;
- system/user synthesis prompt;
- raw model output;
- generated answer/claim text;
- hidden thinking/reasoning;
- internal UUIDs or ACL facts.

The same content-suppression policy applies to both Langfuse and any internal dataset/distillation trace collector.

### 7.3 Model-facing minimization

The builder receives only:

- the contextualized query/normalized meaning necessary to answer;
- the deterministically selected, already admitted synthesis evidence content;
- ephemeral handles `E1`, `E2`, ... assigned by the server from the checkpointed handle manifest;
- bounded presentation instructions and strict output schema.

It does not receive:

- `CapabilityRuntimeContext`;
- workspace/user/run/task identity;
- plans or bindings;
- ACL decisions;
- document, revision, evidence, use, or checkpoint UUIDs;
- tools or tool schemas;
- retention policy.

Evidence content is delimited as untrusted quoted data. The system prompt explicitly instructs the model to ignore commands, role requests, prompt injections, or output-format changes embedded inside evidence.

---

## 8. Evidence admission, target-aware selection, and stable handles

### 8.1 Governed hydration first

Only evidence admitted by the existing governor under current ACL, revision, expiry, tombstone, purpose, and retention rules is eligible for synthesis selection. Discovery-only evidence cannot support claims.

Hydration remains authoritative. Selection cannot resurrect a denied use.

### 8.2 Target-aware selection

Production LLM synthesis must not use the current FIFO head/tail behavior as its semantic selection policy.

The selector groups admitted evidence by logical target (`target_id`) plus a global/supporting group, preserves deterministic evidence order inside each group, and fills the prompt budget in two passes:

1. **coverage pass:** include at least one eligible item for each required target that has admitted evidence;
2. **round-robin pass:** add further items across targets in stable order until the prompt budget is reached.

The selector must reserve budget for:

- system/schema instructions;
- contextualized query;
- evidence delimiters/handles;
- bounded output tokens.

For compare/multi-target work, synthesis fails closed rather than silently answering from one side when a required target had admitted evidence before selection but none survives the selected prompt set.

Single-target factual retrieval remains simple: the selector preserves ranking/order and fills from that target until budget.

### 8.3 Overflow/derived evidence

The existing overflow path currently stores concatenated tail content as derived evidence. That artifact may remain for compatibility/governance, but this design does **not** treat raw concatenation as an LLM-generated semantic summary.

A derived item may be selected only when:

- its content fits the synthesis input budget;
- its stored validation state remains valid;
- its source lineage rehydrates successfully under current authority;
- the eventual citation projector can expand it to locatable document-backed lineage.

A future true compaction/summarization subsystem requires its own faithfulness contract and is outside this design.

### 8.4 Stable HandleManifest

Before the first provider call, the server creates and checkpoints a stable manifest:

```text
HandleManifest
  E1 -> EvidenceUseRef(use_A)
  E2 -> EvidenceUseRef(use_B)
  E3 -> EvidenceUseRef(use_C)
```

Properties:

- each handle maps to exactly one exact current-run `EvidenceUseRef`;
- handle order follows the deterministic selected-evidence order;
- the manifest contains no evidence plaintext;
- the manifest is internal/checkpointed and never public/model-authoritative beyond the opaque E-label;
- repair uses the same manifest and handle numbering;
- resume never reassigns an existing handle to a different use.

Example forbidden behavior:

```text
initial: E1=A, E2=B, E3=C
resume:  B denied
WRONG:   E1=A, E2=C
RIGHT:   E2 still means B -> exact rehydration fails -> fail closed
```

This identity invariant is mandatory. Schema-valid handle reuse must never change provenance across retry/resume.

---

## 9. Structured model output

### 9.1 Proposal schema

The only accepted model output is one JSON object equivalent to:

```json
{
  "claims": [
    {
      "kind": "summary",
      "text": "Hành vi cung cấp thông tin sai sự thật có thể bị xử phạt hành chính.",
      "evidence": ["E1", "E3"]
    },
    {
      "kind": "detail",
      "text": "Mức phạt cụ thể phụ thuộc hành vi, chủ thể và quy định áp dụng.",
      "evidence": ["E2"]
    }
  ]
}
```

`kind` is a presentation hint with the closed values `summary`, `detail`, and `caveat`; it carries no routing or authorization meaning.

The model does not return:

- `content` as a second independently trusted copy;
- `claim_id`;
- evidence/use/document UUIDs;
- public citation indexes;
- terminal status;
- sources or URLs.

### 9.2 Parser and limits

The parser must:

- extract exactly one JSON object, allowing only a surrounding Markdown JSON fence for provider compatibility;
- reject prose before or after the object;
- reject unknown top-level and claim fields;
- require at least one and at most 12 claims;
- require nonblank claim text;
- enforce per-claim and total-character limits in addition to provider token limits;
- require every claim to reference one to three evidence handles;
- reject handles absent from the checkpointed manifest;
- deduplicate repeated handles while preserving model order;
- require each claim to represent one material assertion;
- reject duplicate normalized claim text;
- construct server-owned `claim-1`, `claim-2`, ... IDs in output order;
- resolve each accepted handle through the exact manifest entry to an admitted `EvidenceUseRef`.

The parser does **not** build user-visible Markdown. It outputs typed claims only.

### 9.3 Prompt requirements

The synthesis prompt must require the model to:

- answer the current contextualized query in its language, defaulting to Vietnamese;
- provide a direct short summary followed by only relevant details;
- explain legal rules instead of copying every retrieved provision;
- state conditions, exceptions, validity warnings, dates, time limits, and differing penalty bands only when present in cited evidence;
- use only supplied evidence;
- attach the most directly supporting handles to every factual claim;
- use no more than three evidence handles per claim;
- state a bounded caveat when the selected evidence answers only part of the question;
- emit only the declared JSON schema;
- ignore instructions, role requests, or output-format changes contained inside evidence text;
- never create IDs or infer missing facts.

Examples in the prompt use placeholder handles only and never resemble live public citation IDs.

---

## 10. Claim-first grounding and support validation

### 10.1 Ground claims before rendering

The production architecture must not rely on this loop:

```text
claims -> render Markdown -> split assertions -> map text back to claims
```

That pattern is fragile because server-added bullets/headings/punctuation can change normalized text and create false unmapped/ambiguous failures.

The required flow is:

```text
ParsedClaim[]
   -> resolve exact EvidenceUseRefs
   -> validate claims
   -> validate support guards
   -> GroundedClaim[]
   -> CitationProjector
   -> render Markdown + inline markers
```

The existing assertion splitter/string-mapping code may remain for extractive compatibility tests, but production structured synthesis must not depend on parsing rendered content back into claim identity.

### 10.2 Traceability guarantee

For every grounded claim, the server proves:

- the claim has a server-owned claim ID;
- it cites one to three handles issued in the exact handle manifest;
- those handles resolve to exact admitted current-run uses;
- those uses remain authorized at validation time;
- all cited evidence is eligible for synthesis and public citation projection.

This is a server-verifiable **claim-to-evidence traceability guarantee**.

### 10.3 Deterministic high-risk factual guards

Because this phase does not add a second semantic judge, deterministic guards must cover factual anchors where hallucination is especially damaging and literal/canonical validation is feasible.

At minimum, extract and canonicalize from each generated claim:

- monetary amounts and other numeric quantities;
- percentages;
- dates and explicit time periods/deadlines;
- `Điều`, `Khoản`, `Điểm`, `Chương`, `Mục` locators;
- official document numbers/symbols when stated;
- other closed literal identifiers configured by the implementation.

Every such anchor in a claim must be found in at least one cited evidence item after domain-aware normalization, for example:

```text
20.000.000 đồng <-> 20 triệu đồng
05 ngày <-> 5 ngày
Điều 5 <-> điều 5
```

If a high-risk anchor is unsupported, the candidate enters the bounded repair path. It never reaches public rendering.

These guards intentionally do not claim to prove arbitrary semantic entailment. The model remains constrained to supplied evidence, and unsupported semantic claims that contain no deterministically checkable anchor are a residual model-quality risk measured by offline/live evaluation.

### 10.4 One material assertion per claim

Each claim must contain one material assertion so evidence mapping and inline citation placement stay unambiguous. Compound claims with independently supportable propositions are schema-invalid/repairable rather than rendered as one citation-bearing sentence.

---

## 11. Citation projection and server-owned rendering

### 11.1 Single CitationProjector owner

Introduce one server-owned citation projection boundary consumed by synthesis rendering, SSE, persistence, and history reload.

Conceptually:

```text
GroundedClaim
  -> EvidenceUseRef
  -> EvidenceRecord.source
  -> CitationProjector
  -> PublicCitation
```

No other layer independently fabricates citation metadata.

### 11.2 Supported source projection in this phase

`DocumentSourceIdentity` projects through authoritative document/revision/locator stores under current scope to a public document citation.

`DerivedSourceIdentity` recursively expands to currently authorized source lineage. The projector deduplicates locatable document sources in deterministic lineage order. A bare derived citation is invalid.

`PeopleSourceIdentity` is not projected into document synthesis; People keeps its separate public presentation.

`KnowledgeGraphSourceIdentity` is not accepted for grounded LLM synthesis in this phase unless a later first-class public KG citation/navigation contract is implemented. It must not be disguised as a document citation.

### 11.3 Public citation metadata

The projector returns an allowlisted public document citation sufficient for the existing frontend, including where available:

- `citation_id`;
- four-character `index`;
- `label`;
- `source_type`;
- advisory retrieval score if policy retains it;
- `document_id`;
- `chunk_id`;
- authorized content excerpt;
- `source_file`;
- `page_no` and `heading_path`;
- `document_number` and `article_label`;
- `validity_status` and `superseded_by`.

Internal evidence/use/task/run/workspace/binding/revision/checkpoint identities remain forbidden by the public transport allowlist.

All location/title/validity metadata comes from authoritative server stores and immutable source locators, never from model output.

### 11.4 Public handle generation

After claims are grounded and citations are resolvable, the server deterministically assigns a unique four-character alphanumeric index to each first-used public source.

Requirements:

- include at least one letter for V1 sanitizer/frontend compatibility;
- deterministic collision resolution;
- repeated use of one public source reuses the same index;
- ordering follows first claim use, then cited evidence order;
- at most three indexes render for one claim;
- public index is not an internal evidence/document authority token.

For the public projection, `citation_id` and `index` may share the same opaque four-character value.

### 11.5 Render after grounding

The server renders each grounded claim exactly once. Citation markers are inserted from the claim's resolved public citations immediately before terminal sentence punctuation:

```markdown
Hành vi này có thể bị xử phạt hành chính[a3z9][b2m7].
```

The renderer then applies presentation structure:

- summary claims -> concise opening paragraph(s);
- detail claims -> readable Markdown bullets or paragraphs;
- caveat claims -> bounded caveat paragraph/list item.

Because rendering occurs **after** grounding, adding bullet prefixes or Markdown structure cannot break claim identity.

The renderer never groups identifiers as `[a3z9, b2m7]`, never leaves a space before the first marker, and never emits a separate references list unless the public product contract changes later.

---

## 12. Initial attempt, repair, and failure

### 12.1 Prepare + reserve

Before attempt one:

1. Hydrate evidence under current authority.
2. Select prompt evidence with the target-aware selector.
3. Create the stable handle manifest.
4. Checkpoint the prepared manifest.
5. Reserve attempt one by incrementing `attempts_started` and checkpoint **before** the provider call.

### 12.2 Initial attempt

For attempt one:

1. Build the minimized prompt from the prepared manifest and selected evidence.
2. Call the privacy-safe effective-main provider with `temperature=0` and bounded output tokens.
3. Parse the structured claim proposal.
4. Checkpoint only the safely parsed bounded candidate or closed failure code; never raw output.
5. Resolve handles through the exact checkpointed manifest.
6. Run claim/schema/high-risk-anchor validation.
7. Rehydrate cited exact uses under current authority if a checkpoint boundary was crossed.
8. Project citations.
9. Render the fully grounded answer.
10. Checkpoint the grounded artifact before public answer emission.

A candidate is not successful merely because the provider returned text. Schema, manifest, authorization, support-guard, citation-projection, and rendering checks must all pass.

### 12.3 Repair attempt

One repair call is allowed when the first attempt fails because of:

- malformed or extra output;
- schema/size violations;
- missing, unknown, or excessive evidence handles;
- duplicate/compound claims;
- unsupported deterministic high-risk anchors;
- citation projection failure that can be corrected by citing another already-supplied handle;
- a transient first provider failure when enough turn deadline remains.

The repair prompt uses the **same handle manifest and evidence set**. Handles are not re-numbered.

It contains:

- the same minimized query and `E1..En` evidence set;
- the prior structured proposal only when safely parsed;
- closed, user-independent error codes and affected claim indexes;
- no stack traces, UUIDs, ACL facts, database details, or raw internal exception messages.

The second result passes through the complete parser, manifest resolution, support validation, citation projection, and rendering pipeline again. There is no third model call.

### 12.4 Fail-closed terminal

When the second attempt fails—or the deadline leaves no room for repair—the turn emits exactly one public terminal:

```text
event: error
code: synthesis_failed
message: Không thể tổng hợp câu trả lời đã được kiểm chứng. Vui lòng thử lại.
```

No answer tokens, raw evidence, partial proposal, or internal diagnostic precedes that terminal.

The safe message is persisted as a nonblank assistant message so history reload cannot recreate a blank-assistant row. Internal reasons remain telemetry-only.

There is no same-request fallback to V1 and no production fallback to `build_extractive_draft()`.

---

## 13. Checkpoint and resume semantics

### 13.1 Bounded synthesis state machine

LLM output is not guaranteed deterministic even at temperature zero. Runtime-only `AnswerDraftChannel` cannot be the sole owner because a worker restart could otherwise cause uncontrolled regeneration or extractive re-derivation.

Synthesis therefore runs as a small bounded checkpointed state machine:

```text
idle
  -> prepared(handle_manifest)
  -> attempt_reserved
  -> candidate
  -> grounded
  -> public terminal

candidate/validation failure
  -> reserve repair once
  -> attempt_reserved
  -> candidate
  -> grounded | failed
```

`attempts_started` increments and checkpoints **before** every provider call. A crash during generation consumes that attempt.

### 13.2 SynthesisCheckpoint

Add one additive checkpoint slot, conceptually `synthesis`, backed by a versioned `SynthesisCheckpoint` containing:

```text
contract_version
phase: idle | prepared | attempt_reserved | candidate | grounded | failed
attempts_started: 0 | 1 | 2
handle_manifest: tuple[handle -> EvidenceUseRef]
parsed candidate claims + E-handles (phase=candidate)
grounded claims + exact EvidenceUseRefs (phase=grounded)
server-rendered final content when grounded
presentation kind per claim
closed failure codes only
```

The checkpoint never contains:

- raw evidence plaintext;
- prompt text;
- unparsed model output;
- hidden reasoning;
- provider secrets;
- ACL decisions/state;
- stack traces;
- public source excerpts.

Internal exact `EvidenceUseRef` identities are permitted in the synthesis checkpoint because they are required to preserve provenance across resume and never cross the model/public boundary.

### 13.3 Resume rules

- `idle`: prepare normally.
- `prepared`: reuse the exact manifest; reserve the first attempt if none has started and deadline remains.
- `attempt_reserved`: the reserved call is considered consumed. Resume records a closed interruption code and may reserve the repair only when `attempts_started < 2` and deadline remains.
- `candidate`: resolve handles only through the checkpointed manifest and validate/ground without regenerating.
- `grounded`: perform zero model calls. Rehydrate exact cited uses and re-run current authorization/citation projection checks before public emission/re-emission.
- `failed`: return the same safe typed failure with zero model calls.
- If any exact manifest/cited use no longer passes current ACL, revision, expiry, tombstone, lineage, or retention checks, fail closed. Do not substitute another use into the same E-handle.
- If a grounded artifact no longer passes current authority, fail closed; do not regenerate around changed authority.
- A fresh turn resets the synthesis checkpoint together with stale terminal state.
- Old checkpoints with no synthesis slot normalize to `idle`.
- A legacy mid-turn checkpoint cannot use extractive production fallback; it enters the bounded state machine only if its authoritative state can be safely normalized, otherwise returns the typed synthesis failure.

The runtime-only `AnswerDraftChannel` may remain as an in-process cache, but `SynthesisCheckpoint` is authoritative for attempts, handles, and recovery.

No SQL schema migration is required for the additive LangGraph state slot; compatibility and real Postgres serde tests remain mandatory.

---

## 14. SSE, persistence, deadline, and cancellation

### 14.1 Successful public order

A successful buffered turn has the public order:

```text
status(generating)
citation(...all public citations...)
token(...validated rendered answer chunks...)
complete(answer + citations + status=success)
```

The first answer token is therefore grounded and citation-resolvable when it arrives.

The rendered answer and the same public citation metadata are persisted so an immediate hard reload reconstructs identical clickable badges. Persistence consumes the same `CitationProjector` output; it does not independently rebuild public source identity.

### 14.2 Deadline and cancellation

- Both model attempts share the existing turn deadline; they do not each receive a fresh timeout budget.
- The builder checks remaining time before starting repair.
- Provider calls use async cancellation where supported.
- User cancellation stops an in-flight synthesis call and prevents any later success terminal.
- Cancelled turns preserve existing cancellation semantics and never become `synthesis_failed` merely because cancellation interrupted the provider.
- The stream emits exactly one of `complete`, `error`, or `cancelled`.
- Because tokens are emitted only after grounded artifact success, model failure cannot leave speculative prose in the UI or DB.

---

## 15. Security and privacy properties

The implementation must preserve these invariants:

- only governor-admitted evidence may be selected for the prompt;
- current ACL is checked on initial hydration and resume/revalidation;
- target-bound evidence still requires authoritative plan/binding/revision relationships;
- discovery-only evidence cannot support claims;
- the selected evidence set is server-owned and target-aware;
- E-handles are opaque model-facing aliases backed by a stable checkpointed exact-use manifest;
- the model cannot add evidence by naming a handle outside the manifest;
- the model cannot cause an existing E-handle to resolve to a different use after retry/resume;
- the model cannot issue public citation indexes;
- raw prompt, query/evidence plaintext, generated answer text, model reasoning, and raw invalid output do not enter synthesis telemetry/tracing sinks;
- evidence text is explicitly treated as untrusted prompt data;
- provider selection remains server configuration, never request-controlled;
- no model output can alter plan, route, capability, ACL, lease, binding, revision, or checkpoint identity;
- derived evidence is never public-cited without recursively authorized locatable lineage;
- People data does not cross the synthesis-model boundary in this phase;
- KG evidence does not cross this document-grounded synthesis boundary unless a future first-class KG public citation contract is added.

This design guarantees governed claim-to-evidence traceability plus deterministic validation of defined high-risk factual anchors. It explicitly does **not** claim formal semantic entailment for arbitrary prose.

---

## 16. Observability

Synthesis emits structured content-free telemetry:

- `synthesis_attempt_count`;
- `synthesis_latency_ms`;
- `synthesis_outcome`;
- `synthesis_failure_code`;
- `repair_attempted`;
- `selected_evidence_count`;
- `selected_target_count`;
- `claim_count`;
- `citation_count`;
- provider/model role=`main` from the effective runtime snapshot;
- cancellation/deadline outcome.

Allowed closed failure codes include values such as:

```text
presentation_unsupported_source
selection_missing_target
provider_error
provider_timeout
malformed_json
schema_invalid
unknown_evidence_handle
handle_manifest_mismatch
claim_limit_exceeded
claim_compound
claim_anchor_unsupported
citation_unresolvable
resume_revalidation_failed
deadline_exhausted
```

No failure code embeds user/evidence content or internal identifiers.

The synthesis path must use content-suppressed tracing rather than the current generic main-LLM full-content trace behavior. OTLP exporter failures remain an operational configuration problem and cannot change synthesis terminal outcome.

---

## 17. Test strategy

Implementation follows TDD.

### 17.1 Presentation-policy tests

- People lookup does not call the synthesis LLM and preserves existing public People presentation;
- document-backed factual retrieval selects document synthesis;
- derived document lineage is accepted only when every cited lineage source revalidates;
- KG-only synthesis is typed unsupported in this phase rather than fabricated as a document citation;
- direct/clarify/denied/insufficient paths make zero synthesis model calls.

### 17.2 Evidence-selection tests

- single-target ranking/order remains stable;
- two-target comparison selects evidence from both targets before filling extra quota;
- multi-target selection is deterministic across retries;
- a required target cannot be silently starved by FIFO budget exhaustion;
- prompt + evidence + reserved output stay within configured synthesis budget;
- raw-concatenation derived overflow is not mislabeled/tested as a semantic summary.

### 17.3 Handle-manifest tests

- `E1..En` maps to exact expected `EvidenceUseRef` identities;
- repair preserves the same manifest and numbering;
- candidate resume resolves through the persisted manifest;
- if `E2` becomes unauthorized, `E2` fails rehydration rather than being rebound to the next surviving use;
- manifest contains no evidence plaintext and never crosses public/model transport except for opaque E-labels.

### 17.4 Structured adapter tests

- one valid proposal constructs the expected ordered parsed claims;
- handles resolve to exact admitted uses through the manifest;
- unknown/fabricated handles fail;
- empty claims, duplicate normalized claims, extra fields, oversized claims, excessive handles, and compound claims fail;
- fenced JSON is accepted; surrounding prose is rejected;
- evidence prompt injection cannot alter accepted schema or authority;
- provider/model raw output is never logged/traced on failure;
- Vietnamese and English output preserves query language.

### 17.5 Support-validation tests

- amount normalization supports equivalent forms such as `20.000.000 đồng` and `20 triệu đồng`;
- dates/time periods normalize deterministically;
- legal locators/document numbers in a claim must be present in cited evidence after canonicalization;
- unsupported high-risk anchors trigger repair/failure;
- claim grounding does not depend on parsing server-rendered Markdown;
- adding bullet/list formatting after grounding cannot create an unmapped claim.

### 17.6 Repair tests

- malformed first output followed by valid repair succeeds in exactly two calls;
- unsupported anchor on first draft followed by valid repair succeeds;
- transient first provider error may repair when time remains;
- two invalid outputs produce `synthesis_failed`;
- no third call occurs;
- no repair starts after deadline exhaustion;
- cancellation propagates without repair or late success.

### 17.7 Checkpoint tests

- every synthesis checkpoint phase round-trips through JSON serde and real `AsyncPostgresSaver`;
- prepared/candidate checkpoint carries stable handle->use references and no evidence plaintext;
- checkpoint contains no prompt, unparsed output, reasoning, or public excerpt;
- attempt reservation is durable before each provider call;
- a crash during attempt one can start at most the repair; a crash during attempt two starts no further call;
- candidate resume performs zero regeneration before validation;
- grounded resume performs zero model calls;
- resume rejects revoked/expired/tombstoned exact uses without handle renumbering;
- old checkpoints missing the slot normalize successfully;
- a fresh turn clears the prior synthesis artifact;
- retry/resume cannot exceed the two-attempt budget.

### 17.8 Citation projector/parity tests

- one citation projector owns runtime render, SSE, persistence, and reload projection;
- public indexes are four-character alphanumeric values containing a letter;
- collision resolution is deterministic;
- answer markers use V1 syntax and occur immediately before terminal punctuation;
- multiple sources render as `[a3z9][b2m7]`, never grouped;
- fabricated/unmatched markers cannot cross the public boundary;
- `citation` is emitted before the first token;
- citation payload contains locatable allowlisted metadata and no internal keys;
- derived evidence expands recursively to authorized document lineage;
- People/KG cannot be disguised as document citations;
- the existing frontend converts markers to clickable badges;
- clicking opens the correct document/chunk;
- hard reload preserves the same answer markers and citation metadata.

### 17.9 Tracing/privacy tests

- synthesis resolves the effective `main` provider/model configuration;
- Langfuse synthesis observation contains only approved metadata and no query/evidence/answer plaintext;
- internal dataset/distillation collector receives no synthesis prompt/evidence/answer content;
- failure paths do not log raw invalid model output;
- tracing exporter failure never changes the synthesis result.

### 17.10 API/streaming/persistence tests

- successful output emits tokens only after grounding/rendering and exactly one `complete` terminal;
- model/validation failure emits no tokens and exactly one typed `error`;
- the safe failure message persists as nonblank assistant content;
- cancellation emits no late success;
- session title/summary persistence sees the final synthesized answer where those downstream policies permit it;
- public contract remains additive/version-stamped;
- V1 serving remains unchanged.

### 17.11 Live acceptance

Against the configured effective main provider and a disposable/test session, the canonical document-backed query must:

- return a concise synthesized Vietnamese explanation;
- avoid dumping retrieved chunks;
- state only evidence-backed conduct, sanctions, conditions, dates, amounts, and caveats under the defined validation guarantee;
- place at least one clickable citation after every material factual claim;
- open the correct document/chunk from each citation;
- retain citations after hard reload;
- produce no blank assistant row, raw evidence dump, raw internal citation ID, leaked synthesis evidence in tracing, or multiple terminal event.

A multi-target acceptance case must additionally prove both targets survive prompt selection and appear in the grounded claim set when the answer requires both.

---

## 18. Rollout and rollback

No new synthesis traffic feature flag is added.

Rollout uses the existing V2 arm controls and pre-canary gates:

1. run the full offline V2, API, frontend, static, checkpoint, privacy/tracing, and public-contract suites;
2. run disposable Postgres checkpoint/resume tests with the dependency-complete image;
3. run main-provider live smoke tests with content-safe fixtures;
4. verify trace sinks contain no synthesis prompt/evidence/answer plaintext;
5. verify People and unsupported KG paths do not enter document synthesis;
6. promote through existing canary stages while observing synthesis outcomes, target coverage, citation coverage, latency, repair rate, and cancellation;
7. roll back affected traffic to V1 through existing persisted rollout control if gates fail.

A synthesis failure never triggers an implicit same-request V1 or extractive fallback. Rollback is an operator-owned arm decision, preserving attribution and authority boundaries.

Operational rollout must not restart vLLM engines. Backend/frontend recreation, if required, follows the existing handoff runbook.

---

## 19. Documentation impact

The implementation change must update architecture documentation in the same change:

- `CLAUDE.md`: canonical V2 synthesis ownership, presentation policy, runtime builder, stable handle manifest, checkpoint artifact, claim-first grounding, privacy-safe tracing, and citation flow;
- `README.md`: pointer to the canonical architecture section, without duplicating it;
- `docs/harness.md`: focused selection/handle/checkpoint/tracing/synthesis/citation tests and live acceptance commands;
- `docs/pre-canary-handoff.md`: synthesis failure, privacy tracing, target coverage, citation, checkpoint, People/KG exclusions, and rollback checks.

No `.env.example` model-role change is expected because the design reuses the effective `main` configuration. Any tracing-policy implementation should reuse existing settings where possible rather than creating a second provider stack.

---

## 20. Acceptance criteria

The design is complete only when all of the following hold:

1. Production V2 document-backed factual success uses a structured LLM builder based on the effective `main` model configuration.
2. Production does not call `build_extractive_draft()` as a factual fallback.
3. People presentation remains outside the synthesis model in this phase; KG-only claims are not synthesized until a first-class public KG citation contract exists.
4. Only currently admitted evidence selected by a deterministic target-aware selector reaches the model.
5. Before every model call, the checkpoint owns an exact stable `E-handle -> EvidenceUseRef` manifest; retry/resume never rebinds an existing handle to different evidence.
6. The model receives only minimized evidence plus opaque E-handles and never receives internal trusted identities.
7. The backend owns claim IDs, exact use resolution, support guards, public citation indexes, terminal status, and Markdown rendering.
8. Grounding is claim-first/render-last; production does not parse rendered Markdown back into claim identity.
9. Every claim references one to three exact admitted uses, and every defined high-risk factual anchor in a claim is present in at least one cited evidence item after canonical normalization.
10. At most two model calls occur per synthesis operation, including across crash/resume.
11. No answer token is emitted before complete validation, citation projection, rendering, and grounded-artifact checkpointing.
12. Every in-scope document-backed material factual claim has one to three V1-compatible clickable inline citations.
13. One `CitationProjector` owns source-to-public-citation projection for rendering, SSE, persistence, and reload.
14. Citation metadata opens the correct currently authorized document/chunk and survives hard reload.
15. Invalid model output or failed support/citation validation after repair fails closed with `synthesis_failed` and a nonblank persisted safe message.
16. Checkpoint resume with a grounded artifact performs no new model generation; current exact evidence access is rechecked and revoked uses fail closed without handle renumbering.
17. Synthesis tracing is content-free: query/evidence/prompt/generated-answer plaintext is not exported through Langfuse or the dataset trace collector by this path.
18. Cancellation cannot produce a late success.
19. Existing ACL, binding, revision pin, scheduler, evidence governance, checkpoint, and canary authority remain unchanged.
20. V1 public citation behavior and frontend compatibility are regression-tested.
21. Multi-target synthesis cannot silently starve a required target under prompt budget.
22. Independent review reports no Critical or Important blocker before runtime promotion.
