"""
Dọn các chunk MỒ CÔI trong ChromaDB — chunk có ``document_id`` trỏ tới document
ĐÃ BỊ XOÁ/RE-UPLOAD (UUID không còn trong bảng ``documents`` của Postgres).

Triệu chứng: câu trả lời trích những "nguồn không có thật" (vd index "npdk",
file 53-cp.signed.pdf) mà click vào không mở được — vì frontend ``useDocument``
fetch document_id chết → fallback "Nguồn <index>". Orphan còn chiếm slot top-k
nên đẩy chunk thật ra ngoài → giảm chất lượng trả lời.

Nguồn gốc: re-upload tạo document_id MỚI, còn chunk cũ trong collection của
workspace không phải lúc nào cũng bị xoá (xoá cũ trước khi có cleanup, hoặc
lệch workspace). Lỗi xoá hiện tại đã dọn vector (app/api/documents.py
delete_document), nên đây chủ yếu là tàn dư.

Chạy (cần Postgres + ChromaDB):
    python -m scripts.purge_orphan_vectors --dry-run   # chỉ in
    python -m scripts.purge_orphan_vectors             # xoá thật

Idempotent: chỉ xoá chunk có document_id KHÔNG tồn tại trong Postgres. Giữ
nguyên document_id nil (00000000-… của nguồn KG) và id còn sống.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.document import Document
from app.services.vector_store import VectorStore, get_chroma_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("purge_orphan_vectors")

_NIL_UUID = str(uuid.UUID(int=0))


async def _existing_document_ids() -> set[str]:
    async with async_session_maker() as db:
        rows = await db.execute(select(Document.id))
        return {str(r[0]) for r in rows.all()}


def _collection_document_ids(col) -> dict[str, int]:
    """Map document_id → số chunk, đọc toàn bộ metadata của một collection."""
    counts: dict[str, int] = {}
    got = col.get(include=["metadatas"])
    for md in got.get("metadatas", []) or []:
        did = str((md or {}).get("document_id") or "")
        if did:
            counts[did] = counts.get(did, 0) + 1
    return counts


async def purge(dry_run: bool = False) -> None:
    existing = await _existing_document_ids()
    logger.info(f"Postgres có {len(existing)} document còn sống")

    client = get_chroma_client()
    collections = [c for c in client.list_collections()
                   if c.name.startswith(VectorStore.COLLECTION_PREFIX)]
    logger.info(f"ChromaDB có {len(collections)} collection (prefix '{VectorStore.COLLECTION_PREFIX}')")

    total_orphan_docs = total_orphan_chunks = 0

    for c in collections:
        ws_id = c.name[len(VectorStore.COLLECTION_PREFIX):]
        try:
            store = VectorStore(uuid.UUID(ws_id))
        except ValueError:
            logger.warning(f"  collection {c.name}: tên workspace không phải UUID — bỏ qua")
            continue

        counts = _collection_document_ids(store.collection)
        orphans = {
            did: n for did, n in counts.items()
            if did != _NIL_UUID and did not in existing
        }
        if not orphans:
            logger.info(f"  {c.name}: {sum(counts.values())} chunk, 0 orphan ✓")
            continue

        n_chunks = sum(orphans.values())
        total_orphan_docs += len(orphans)
        total_orphan_chunks += n_chunks
        logger.info(
            f"  {c.name}: {len(orphans)} document_id mồ côi / {n_chunks} chunk "
            f"(trên tổng {sum(counts.values())})"
        )
        for did, n in sorted(orphans.items(), key=lambda x: -x[1]):
            logger.info(f"      - {did}: {n} chunk")
            if not dry_run:
                try:
                    store.delete_by_document_id(uuid.UUID(did))
                except Exception as e:
                    logger.warning(f"        xoá lỗi: {e}")

    logger.info(
        f"Xong. Orphan: {total_orphan_docs} document_id / {total_orphan_chunks} chunk"
        + ("  [DRY-RUN — chưa xoá]" if dry_run else "  — đã xoá")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dọn chunk mồ côi trong ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in, không xoá")
    args = parser.parse_args()
    asyncio.run(purge(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
