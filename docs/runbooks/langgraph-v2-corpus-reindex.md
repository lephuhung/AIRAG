# Runbook: v2 revision-aware corpus reindex

Purpose: publish v2 revision artifacts for legacy documents so
`document_views.load_current_revision_identity()` returns an identity and v2
factual routes can read the corpus. A document without a published revision
MUST keep failing closed; this runbook never weakens that guard.

## Constraints

- NEVER set `Document.current_revision_id` by hand. Only the revision publish
  CAS sets it.
- NEVER restart the vLLM engines (`hrag-vllm-ocr`, `hrag-vllm-memory`).
- Reindex is copy-on-write: the published revision stays current and readable
  until the replacement publishes. Reclamation is GC's job.

## 1. Pre-check (read-only)

```bash
# v2 schema applied?
docker exec hrag-postgres psql -U postgres -d hrag -c "select version from v2_schema_version;"

# legacy documents in the target workspaces (namespace per workspace id)
docker exec hrag-postgres psql -U postgres -d hrag -c \
  "select id, workspace_id, document_number, document_title, status, current_revision_id
     from documents
    where source_deleted_at is null and current_revision_id is null
    order by workspace_id;"
```

List workspace UUIDs through the API (`GET /api/v1/workspaces`) with a
superadmin JWT, or resolve the workspace by name from the pre-check output.

## 2. Execute

```bash
# Obtain $TOKEN first: mint a short-lived superadmin JWT in-container per
docs/auth.md ("Mint a JWT without a password"); never print or commit it.
# Run curl INSIDE the container: host :8080 is the nginx proxy (it 301s) and
hrag-backend publishes no host port, so localhost:8080 only serves the API
# in-container.
docker exec hrag-backend curl -s -X POST "http://localhost:8080/api/v1/rag/reindex-workspace/$WS" -H "Authorization: Bearer $TOKEN" | jq
```

Run the smallest workspace first. The endpoint queues each document
(`allocate_reindex_revision` → new monotonic generation) and returns
`document_count`; the parse/embed/kg/caption workers build and publish.

## 3. Verify

```bash
# pointer + published revision + build artifacts. LEFT JOINs (not an inner
# join) so an empty result means "no live documents in the workspace" while
# rows with published=false mean "documents exist but nothing published yet".
docker exec hrag-postgres psql -U postgres -d hrag -c \
  "select d.id, d.current_revision_id,
          (r.status = 'published') as published, r.status, b.build_profile,
          b.markdown_artifact_key, b.structure_artifact_key,
          b.embedding_namespace, b.embedding_model_hash
     from documents d
     left join document_revisions r on r.revision_id = d.current_revision_id
     left join document_revision_builds b on b.revision_id = r.revision_id
    where d.workspace_id = '$WS' and d.source_deleted_at is null;"
```

- Every doc has a row with non-null `current_revision_id` and
  `published = true` (`status = 'published'`).
- `document_revision_builds` has the profile's required artifacts + embedding manifest.

Boundary check (in-container, read-only):

```bash
docker exec hrag-backend python -c "
import asyncio, uuid
from app.core.database import async_session_maker
from app.services.agents.v2.persistence import document_views
async def main():
    async with async_session_maker() as db:
        ident = await document_views.load_current_revision_identity(db, uuid.UUID('$DOC_ID'))
        print('identity:', ident)
asyncio.run(main())
"
```

Then re-run the authenticated factual SSE probe in the reindexed workspace; it
must return an answer or a typed `insufficient`, never a generic error.

## 4. Abort / rollback

If verification fails, abandon the unpublished revisions so the previous
pointer stays current; never edit the pointer:

```bash
docker exec hrag-backend python -c "
import asyncio, uuid
from app.core.database import async_session_maker
from app.services.agents.v2.persistence.document_revisions import DocumentRevisionsRepository
async def main():
    async with async_session_maker() as db:
        repo = DocumentRevisionsRepository(db)
        for rev in await repo.list_non_published_for_update(uuid.UUID('$DOC_ID')):
            await repo.abandon_revision(rev, reason='reindex verification failed')
        await db.commit()
asyncio.run(main())
"
```

Confirm `current_revision_id` is unchanged and the document still serves its old
revision. Investigate before retrying.
