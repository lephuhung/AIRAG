# Legal-Structure Chunking Unification + Khoản/Điểm Tracking — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every active revision-worker ingestion path produces article-contained
chunks for legal documents: no chunk crosses an Điều boundary, while an oversized
Điều may produce multiple chunks. Khoản/Điểm ownership is carried on every emitted
chunk, including continuation chunks, so `search_document_section` can resolve
composite references such as `khoản 2 Điều 8` and `điểm a khoản 2 Điều 8` without
inventing parent-child relationships.

**Architecture:** Extend the deterministic structure engine in
`heading_path.py` with subdivision markers, parent-aware references, and a
sequence-aware metadata derivation pass. `LegalDocumentChunker` remains the only
legal chunker and is used by both OCR/legacy and Docling legal paths. Internal
models and the revision structure artifact carry typed lists; only the Chroma
write boundary converts lists to pipe-separated strings. Docling uses a separate,
flat, page-marked chunking export while preserving its existing display markdown.
No model or LLM participates in structure parsing.

**Tech stack:** Python 3.11, Docling/HybridChunker,
`langchain-text-splitters`, ChromaDB, pytest.

## Scope and terminology

- **Article-contained** means one chunk never contains text from two different
  Điều. It does not mean one chunk per Điều: long Điều may be split further.
- The active ingestion path is `parse_worker → structure artifact → embed_worker`.
- The inline `HRAGService.process_document()` fallback is currently broken because
  it calls the nonexistent `DeepDocumentParser.parse()`. Repairing that fallback is
  out of scope, but its Chroma metadata writer must still be kept schema-compatible
  so it does not create divergent metadata if the fallback is repaired later.
- This plan changes the v1 `search_document_section` label-based lookup. The v2
  `section.read` capability uses stable `ContentLocator` structure IDs and is not
  changed to consume free-text Khoản/Điểm labels.

## Load-bearing invariants

1. `heading_path` remains `"Phần > Chương > Mục > Điều N"`. Khoản/Điểm never enter
   `_LEVELS` and never alter exact heading-path matching.
2. Every factual subdivision relation is represented as an atomic parent-aware
   reference; independent `khoan_nos` and `diem_labels` are convenience indexes,
   not sufficient proof that a điểm belongs to a khoản.
3. Internal and artifact types use lists. Pipe-joining happens only when writing
   Chroma metadata.
4. New legal chunks carry `subdivision_schema_version=1`, including chunks whose
   subdivision lists are empty. Old chunks/artifacts default to version `0`.
5. A continuation chunk inherits the Khoản/Điểm interval active at its first
   character. Metadata is not derived by scanning each chunk independently.
6. A Khoản interval is never cut unless that single interval exceeds
   `HRAG_LEGAL_CHUNK_MAX_CHARS`. Whole adjacent Khoản intervals may be packed into
   one chunk when the parent-aware references remain intact.
7. `char_start`/`char_end` are exact slices of the chunking source. Docling legal
   offsets are relative to the flat chunking export, not its display HTML.
8. Non-legal documents (`<3` accepted Điều headings) retain their existing
   chunkers: HybridChunker for Docling and DocumentChunker for legacy/OCR.
9. Backfill is metadata-only, dry-run by default, revision-aware, idempotent, and
   never drops existing metadata keys.
10. No LLM/model calls occur in subdivision detection, chunking, or backfill.

## Canonical metadata contract

### Parser output

```python
@dataclass(frozen=True)
class Subdivision:
    kind: Literal["khoan", "diem"]
    label: str
    start: int
    article_no: str
    parent_khoan: str | None = None


@dataclass(frozen=True)
class SubdivisionMetadata:
    khoan_nos: tuple[str, ...] = ()
    diem_labels: tuple[str, ...] = ()
    subdivision_refs: tuple[str, ...] = ()
```

Canonical references:

```text
khoan:2
khoan:2/diem:a
```

A điểm accepted without an enclosing Khoản, if the detector policy later permits
one, must use `diem:a`; it must never be fabricated as a child of another Khoản.

### Chunk and structure artifact

`EnrichedChunk` and `ChunkRecord` add:

```python
khoan_nos: list[str] = field(default_factory=list)
diem_labels: list[str] = field(default_factory=list)
subdivision_refs: list[str] = field(default_factory=list)
subdivision_schema_version: int = 0
```

Old structure artifacts without these keys parse to empty lists and version `0`.
The change is additive, so `STRUCTURE_ARTIFACT_VERSION` and
`VECTOR_ARTIFACT_VERSION` remain unchanged; do not invalidate published revisions.

### Chroma metadata

```python
{
    "khoan_nos": "1|2",
    "diem_labels": "a|b",
    "subdivision_refs": "khoan:1|khoan:1/diem:a|khoan:2|khoan:2/diem:a|khoan:2/diem:b",
    "subdivision_schema_version": 1,
}
```

The empty representation is `""` for the three string indexes. Search must use
`subdivision_refs` for `điểm + khoản`; it must not infer parentage by intersecting
the independent indexes.

---

### Task 0: Confirm blast radius and freeze acceptance tests

**Required before editing symbols:**

- `find_headings`
- `LegalDocumentChunker` / `LegalDocumentChunker.split_text`
- `_parse_with_docling`
- `_chunk_document`
- `_parse_legacy`
- `derive_heading_paths`
- `EnrichedChunk`
- `ChunkRecord`
- `build_structure_artifact`
- `parse_structure_artifact`
- `load_revision_chunk_payloads`
- `search_document_section`

GitNexus currently reports `ChunkRecord` as **HIGH** risk (15 direct dependents,
53 impacted symbols). Stop and report if refreshed impact remains HIGH/CRITICAL;
do not silently proceed.

- [ ] Run upstream `impact` for every symbol above and record direct callers and
      affected flows.
- [ ] Confirm the required acceptance tests in Tasks 1–6 are RED for the intended
      reason before implementation.
- [ ] Do not modify unrelated dirty worktree files.

---

### Task 1: Parent-aware deterministic subdivision engine

**Files:**

- Modify: `backend/app/services/parsing/heading_path.py`
- Modify: `backend/tests/services/test_legal_chunker.py`

**Public interfaces:**

```python
@dataclass(frozen=True)
class SubdivisionParseStats:
    khoan_candidates: int = 0
    khoan_accepted: int = 0
    diem_candidates: int = 0
    diem_accepted: int = 0
    ambiguous_rejected: int = 0


@dataclass(frozen=True)
class SubdivisionParseResult:
    subdivisions: tuple[Subdivision, ...]
    stats: SubdivisionParseStats


def parse_subdivisions(text: str) -> SubdivisionParseResult: ...
def find_subdivisions(text: str) -> list[Subdivision]: ...
def derive_subdivision_metadata(
    chunk_texts: Sequence[str],
    heading_paths: Sequence[Sequence[str]] | None = None,
) -> list[SubdivisionMetadata]: ...
```

`find_subdivisions` is the convenience wrapper over `parse_subdivisions`.
`derive_subdivision_metadata` processes chunks in document order and carries the
active Điều/Khoản/Điểm state across chunk boundaries. Optional `heading_paths`
provide article context for old chunks whose body no longer repeats the heading.

#### Detection policy

- Khoản candidate: `^[ \t]{0,3}(\d{1,2})\.[ \t]+\S` at line start, value
  `1..30`.
- A new Điều heading resets Khoản and Điểm state.
- The first accepted Khoản in an Điều must be `1`. After a sequence is open,
  forward jumps are allowed to tolerate a dropped OCR line; equal/decreasing
  candidates are rejected without changing state.
- Điểm candidate: `^[ \t]{0,3}([a-zđ])\)[ \t]+\S`, using Vietnamese order
  `a,b,c,d,đ,e,g,h,i,k,l,m,n,o,p,q,r,s,t,u,v,x,y`.
- Điểm state resets on each accepted Khoản and each Điều. The first accepted điểm
  in a Khoản must be `a`; forward jumps are allowed only after the sequence opens.
- A điểm without an active Khoản is rejected in version 1. This is conservative
  and can be revisited only with corpus evidence.
- Markdown block quotes/nested list prefixes and indentation greater than three
  spaces are not top-level legal subdivisions.
- The monotonic rules reduce, but cannot eliminate, ambiguity with ordinary
  numbered lists. Rejected-candidate statistics are observability signals, not a
  claim of ground-truth miss rate.

#### Tests first

- [ ] Basic Điều → Khoản → Điểm, including `đ)` and `i)`.
- [ ] Parent references are exact:

```python
assert metadata.subdivision_refs == (
    "khoan:1",
    "khoan:1/diem:a",
    "khoan:2",
    "khoan:2/diem:a",
    "khoan:2/diem:b",
)
```

- [ ] State resets at every Điều and every accepted Khoản.
- [ ] `1., 3.` is accepted after sequence open; first candidate `3.` is rejected.
- [ ] Mid-sentence `khoản 2 Điều 7`, `2024.`, decimal/thousand values, nested
      indentation, block quotes, and decreasing/reset list numbers are rejected.
- [ ] Negative candidates are tested inside an Điều, not only outside the gate.
- [ ] `derive_subdivision_metadata` carries Khoản/Điểm to a continuation chunk
      with no marker.
- [ ] `heading_paths` can establish the current Điều for old marker-less chunks.
- [ ] Run:

```bash
cd backend && pytest tests/services/test_legal_chunker.py -q
```

---

### Task 2: Khoản-aware splitting and inherited metadata

**Files:**

- Modify: `backend/app/services/embedding/chunker.py`
- Modify: `backend/tests/services/test_legal_chunker.py`

#### Algorithm

1. Keep current Phần/Chương/Mục/Điều section slicing; no section crosses an Điều
   boundary.
2. For an oversized Điều, parse all subdivision markers against the full section
   before creating child chunks.
3. Convert accepted Khoản markers into exact `[start, end)` intervals. The Điều
   heading/preamble before Khoản 1 is its own segment and may be packed with the
   first whole Khoản when the result fits.
4. Greedily pack only whole segments up to `max_chars`.
5. If one Khoản interval exceeds `max_chars`, run `_sub_splitter` only inside that
   interval.
6. After all exact chunk slices are known, call
   `derive_subdivision_metadata` over the ordered chunk contents. Copy typed lists
   plus `subdivision_schema_version=1` into each `TextChunk.metadata`.
7. The OCR/legacy wrapper must explicitly copy these values from
   `TextChunk.metadata` into `EnrichedChunk`; do not rely on `extra_metadata` or
   rescanning individual pieces.

#### Tests first

- [ ] No chunk contains headers for two different Điều.
- [ ] No Khoản interval is cut unless that interval alone exceeds `max_chars`.
- [ ] Whole adjacent Khoản may share a chunk, and all parent-aware refs are exact.
- [ ] Every fallback subchunk of one long Khoản inherits its `khoan_nos`.
- [ ] Every fallback subchunk inside a long điểm inherits both
      `khoan_nos` and `subdivision_refs`.
- [ ] The cross-product regression is covered: a chunk containing
      `khoan:1/diem:a` and `khoan:2/diem:b` must not claim
      `khoan:2/diem:a`.
- [ ] For every output chunk:

```python
assert original[c.char_start:c.char_end] == c.content
```

- [ ] Non-legal inputs still use the existing caller-selected chunker.
- [ ] Run:

```bash
cd backend && pytest tests/services/test_legal_chunker.py -q
```

---

### Task 3: Docling legal path uses a separate flat chunking export

**Files:**

- Modify: `backend/app/services/parsing/deep_document_parser.py`
- Create or modify: `backend/tests/services/test_deep_document_parser.py`
- Modify if integration coverage fits better:
  `backend/tests/workers/test_revision_pipeline.py`

#### Display/chunk source separation

Preserve current display behavior, including the existing layout-export
`try/except` fallback. The following is schematic and must not remove that recovery
path:

```python
display_markdown = (
    self._export_layout_markdown(doc, pic_url_list)
    if settings.HRAG_DOCLING_PRESERVE_LAYOUT
    else self._inject_image_references(self._export_markdown(doc), pic_url_list)
)
```

Add a dedicated flat export for structure detection and legal chunking:

```python
chunk_markdown, page_spans = self._export_chunk_markdown(doc)
```

`_export_chunk_markdown` must request a unique internal page-break placeholder,
enumerate it into `<!-- page N -->` markers, and return exact
`(page_no, start, end)` spans. Do not treat arbitrary markdown `---` rules as page
breaks. If the installed Docling version rejects `page_break_placeholder`, log a
warning, keep legal chunking, return no inferred page spans, and use `page_no=0`
rather than fabricating page numbers.

#### Legal branch

```python
if (
    settings.HRAG_LEGAL_CHUNKING
    and LegalDocumentChunker.has_legal_structure(chunk_markdown)
):
    chunks = self._chunk_legal_markdown(...)
else:
    chunks = self._chunk_document(...)
```

For legal chunks:

- `parser` remains `"docling"`;
- `page_no` is the first page whose span intersects
  `[char_start, char_end)`;
- image/table refs are assigned once to the first chunk intersecting their page;
- a chunk spanning multiple pages is eligible for assets on every intersected
  page, not only the page containing `char_start`;
- `has_table` is true when the chunk contains a syntactically valid markdown pipe
  table or receives a table ref from any intersected page;
- `derive_heading_paths` and subdivision metadata operate on flat chunk content;
- display markdown remains unchanged and is returned as `ParsedDocument.markdown`;
- emit one counts-only `[legal-chunking]` log line using `document_id`, not chunk
  text or prompt-like content.

Factor page-asset assignment so HybridChunker and legal chunks share the
first-assignment policy without changing non-legal behavior.

#### Tests first

- [ ] Legal flat Docling export yields article-contained chunks.
- [ ] `HRAG_DOCLING_PRESERVE_LAYOUT=False` and `True` choose the same legal
      chunking source and preserve their respective display markdown.
- [ ] Layout HTML is never fed to subdivision regexes.
- [ ] A long Điều spanning two pages receives correct start page and assets from
      both intersected pages.
- [ ] A real horizontal rule is not interpreted as a page break.
- [ ] Unsupported `page_break_placeholder` yields legal chunks with `page_no=0`,
      no guessed ranges, and no crash.
- [ ] Non-legal Docling input still invokes `_chunk_document` unchanged.
- [ ] Smoke-test a checked-in legal fixture if one exists; do not require a live
      GPU/model service.
- [ ] Run:

```bash
cd backend && pytest tests/services/test_deep_document_parser.py tests/services/test_legal_chunker.py -q
```

---

### Task 4: Persist subdivision metadata through every authoritative boundary

**Files:**

- Modify: `backend/app/services/models/parsed_document.py`
- Modify: `backend/app/services/parsing/deep_document_parser.py`
- Modify: `backend/app/workers/parse_worker.py`
- Modify: `backend/app/services/agents/v2/persistence/document_views.py`
- Modify: `backend/app/workers/utils.py`
- Modify: `backend/app/workers/embed_worker.py`
- Modify: `backend/app/services/retrieval/hrag_service.py` metadata writer only;
  do not repair its `parser.parse()` bug in this task
- Modify tests:
  - `backend/tests/agents/v2/persistence/test_revision_artifacts.py`
  - `backend/tests/workers/test_revision_pipeline.py`

#### Required plumbing

- [ ] Add typed list fields and schema version to `EnrichedChunk`.
- [ ] Legacy/OCR and Docling legal wrappers copy `TextChunk.metadata` into those
      fields.
- [ ] `parse_worker` copies all four fields into `ChunkRecord` and
      `raw_chunks_json`.
- [ ] `ChunkRecord.as_dict()` writes them; `parse_structure_artifact()` reads them
      with additive old-artifact defaults.
- [ ] `load_revision_chunk_payloads()` includes them in the explicit dict it
      returns. This is mandatory because embed worker prefers the structure
      artifact over `raw_chunks_json`.
- [ ] `embed_worker` pipe-joins the three typed lists and writes
      `subdivision_schema_version`.
- [ ] `HRAGService` uses the same Chroma encoding so a future repair cannot revive
      a divergent writer.
- [ ] Caption re-embedding remains metadata-neutral; verify it updates documents
      and embeddings only.
- [ ] Do not bump global artifact/vector versions for this additive change.

#### Tests first

- [ ] New structure artifact round-trip preserves all fields.
- [ ] Old artifact JSON without fields loads empty lists and version `0`.
- [ ] `load_revision_chunk_payloads()` preserves fields from the authoritative
      structure artifact.
- [ ] Embed worker prefers the structure artifact and writes exact Chroma values.
- [ ] Empty typed lists become empty Chroma strings, not missing keys, for schema
      version 1 legal chunks.
- [ ] Existing non-legal/version-0 payloads remain accepted.
- [ ] Run:

```bash
cd backend && pytest \
  tests/agents/v2/persistence/test_revision_artifacts.py \
  tests/workers/test_revision_pipeline.py -q
```

---

### Task 5: Composite `search_document_section` resolution

**Files:**

- Modify: `backend/app/services/agent/tools.py`
- Modify: `backend/tests/agents/v2/persistence/test_revision_live_callers.py`
- Modify or create pure tool tests near:
  `backend/tests/services/test_agent_tools_workspace_scope.py`
- Extend live gated coverage:
  `backend/tests/retrieval/test_section_retrieval.py`

#### Reference parsing

Parse independently of word order and punctuation:

```python
article_no = re.search(r"(?i)\bđiều\s+(\d+[a-z]?)\b", ref)
khoan_no = re.search(r"(?i)\bkhoản\s+(\d{1,2})\b", ref)
diem_label = re.search(r"(?i)\bđiểm\s+([a-zđ])\b", ref)
```

Normalize labels to lowercase. Version 1 supports one Điều, one Khoản, and one
Điểm per lookup; multi-reference expressions such as `khoản 2 và khoản 3` remain
outside this task and must not be silently reduced to one target.

#### Two-pass matching per document/revision

1. Collect Điều-level candidates for one `(collection, document_id, revision_id)`.
2. If no subdivision was requested, return all Điều candidates as today.
3. For `Khoản` only, require exact membership in `khoan_nos`.
4. For `Điểm + Khoản`, require the single atomic token
   `khoan:{n}/diem:{label}` in `subdivision_refs`.
5. For `Điểm` without Khoản, require exact membership in `diem_labels` but do not
   infer a parent.
6. If every Điều candidate has `subdivision_schema_version=0` or lacks the key,
   treat the corpus as pre-backfill and fall back to the Điều-level candidates.
7. If any Điều candidate declares schema version 1, an empty/nonmatching
   subdivision result is authoritative: return not-found for that structural
   lookup. Do not replace it with unrelated semantic top-k chunks.
8. Keep fallback state local to the current document/revision. Do not use the
   function-wide `all_chunks` list to decide whether another store may fall back.
9. Preserve current revision/workspace hard filters and output ordering.

#### Tests first

- [ ] `Điều 8` returns all article chunks regardless of Khoản.
- [ ] `khoản 2 Điều 8` returns only exact Khoản candidates.
- [ ] `điểm a khoản 2 Điều 8` requires
      `khoan:2/diem:a`; independent `2` and `a` in different paths do not match.
- [ ] `điểm a Điều 8` uses `diem_labels` without inventing a Khoản.
- [ ] Old version-0 Điều chunks fall back to Điều-level content.
- [ ] Version-1 chunks with no requested subdivision return not-found, not broad
      Điều content and not semantic top-k.
- [ ] Mixed old/new documents make fallback decisions independently.
- [ ] Multiple revision-qualified stores preserve exact revision filters.
- [ ] Unsupported multi-reference input is rejected or routed to the existing
      semantic behavior explicitly; it is never partially parsed as one target.
- [ ] Run:

```bash
cd backend && pytest \
  tests/services/test_agent_tools_workspace_scope.py \
  tests/agents/v2/persistence/test_revision_live_callers.py -q
```

Live retrieval remains opt-in:

```bash
cd backend && RETRIEVAL_EVAL=1 pytest tests/retrieval/test_section_retrieval.py -q
```

Do not require or restart the Compose/vLLM stack from this worktree session.

---

### Task 6: Safe metadata backfill and parser observability

**Files:**

- Create: `backend/scripts/backfill_subdivision_nos.py`
- Modify: `backend/app/services/parsing/heading_path.py` only if stats helpers need
  a public seam
- Modify: `backend/app/services/parsing/deep_document_parser.py` for one
  counts-only per-document log
- Modify: `docs/embedding.md`
- Create or modify script tests under `backend/tests/scripts/`

#### Backfill contract

- Dry-run is the default; writes require explicit `--apply`.
- Require at least one scope selector: `--workspace`, `--collection`, or
  `--document-id`.
- Enumerate Chroma collections/namespaces, fetch ids + documents + full metadata,
  and group by `(document_id, revision_id-or-legacy)`.
- Sort each group by `ordinal`, falling back to `chunk_index`.
- Pass ordered chunk texts and heading paths to
  `derive_subdivision_metadata`; never call `find_subdivisions` independently per
  chunk.
- Merge new keys into each complete existing metadata dict before
  `update_metadatas`; never submit partial metadata that could erase document,
  revision, workspace, or source identity.
- Batch writes with a bounded batch size.
- Set `subdivision_schema_version=1` only for groups processed successfully from
  start to finish. On malformed ordering or parse failure, leave the entire group
  unchanged and report it.
- Idempotent reruns produce zero additional changes.
- Print counts only: collections, document/revision groups, chunks scanned,
  changed, unchanged, skipped, ambiguous, and failed. Never print chunk text.
- The implementation task creates and tests the script but does not execute a
  production backfill.

The missing documented `backfill_heading_path.py` is not recreated as part of
this task. Update stale documentation separately rather than silently expanding
this migration.

#### Observability

Log structured counts from `SubdivisionParseStats` once per legal document:

```text
[legal-structure] document_id=... khoan_candidates=... khoan_accepted=...
diem_candidates=... diem_accepted=... ambiguous_rejected=...
```

Call this an acceptance/rejection ratio, not a ground-truth miss ratio. A future
model repair pass requires labeled corpus evaluation, not merely a high rejection
count.

#### Tests first

- [ ] Dry-run performs no `update_metadatas` call.
- [ ] `--apply` preserves every pre-existing metadata key.
- [ ] Continuation chunks inherit the prior marker during backfill.
- [ ] Groups are isolated by revision and document.
- [ ] One malformed group is skipped atomically while other groups continue.
- [ ] Second run is idempotent.
- [ ] Run the focused script tests; do not connect to production Chroma.

---

### Task 7: Documentation and final verification

**Files:**

- Modify: `docs/embedding.md`
- Modify: `.env.example` only if an existing legal-chunking flag/default is
  changed. Prefer the existing `HRAG_LEGAL_CHUNKING` and
  `HRAG_LEGAL_CHUNK_MAX_CHARS`; add no new flag by default.

- [ ] Document both active legal ingestion paths, the flat Docling chunking
      export, the four metadata keys, schema-version fallback, and the safe
      backfill command.
- [ ] Correct the stale reference to the nonexistent `backfill_heading_path`
      script; do not claim a command exists when it does not.
- [ ] Run focused tests from Tasks 1–6.
- [ ] Run the narrow worker/persistence regression set covering authoritative
      artifacts and embed metadata.
- [ ] Run `detect_changes({scope: "compare", base_ref: "main"})` before commit.
- [ ] Review every changed symbol and affected flow. Confirm no agent ownership,
      rollout, synthesis, or v2 capability execution invariant changed.
- [ ] Do not run live canary, production backfill, or restart model services from
      this worktree.

## Explicit non-goals

- OCR-path table structure/TableFormer on scans.
- LLM/model-based structure extraction or repair.
- Repairing `HRAGService.process_document()` / `DeepDocumentParser.parse()`.
- Creating the missing historical `backfill_heading_path.py` script.
- Changing `heading_path` format or exact-match semantics.
- Adding free-text labels to v2 `ContentLocator` contracts.
- Multi-reference expressions such as `khoản 2 và khoản 3 Điều 8`.
- `DocumentStatus.OCRING` cleanup.
