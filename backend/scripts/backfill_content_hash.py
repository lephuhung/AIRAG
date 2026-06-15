"""
Backfill ``documents.content_hash`` cho các document cũ.

Cơ chế chống upload trùng (xem app/api/documents.py) so khớp file bằng cột
``content_hash`` (SHA256 của file gốc). Cột này chỉ được tính lúc upload, nên các
document đã tồn tại TRƯỚC khi bật tính năng đều có ``content_hash = NULL`` và sẽ
không bao giờ được nhận diện là trùng. Script này duyệt các document đó, tải file
gốc từ MinIO, tính SHA256 và ghi lại vào DB.

Chạy (cần Postgres + MinIO đang chạy):
    python -m scripts.backfill_content_hash            # thực thi
    python -m scripts.backfill_content_hash --dry-run  # chỉ in, không ghi DB

An toàn để chạy lại nhiều lần (idempotent): chỉ xử lý document có
``content_hash IS NULL`` và ``upload_s3_key IS NOT NULL``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.document import Document
from app.services.storage_service import get_storage_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_content_hash")

# Ghi DB sau mỗi N document để không giữ transaction quá lâu.
COMMIT_EVERY = 50


async def backfill(dry_run: bool = False) -> None:
    storage = get_storage_service()

    async with async_session_maker() as db:
        result = await db.execute(
            select(Document).where(
                Document.content_hash.is_(None),
                Document.upload_s3_key.is_not(None),
            )
        )
        docs = result.scalars().all()

    total = len(docs)
    logger.info(f"Tìm thấy {total} document cần backfill content_hash")
    if total == 0:
        return

    updated = 0
    skipped = 0
    failed = 0

    async with async_session_maker() as db:
        for i, doc in enumerate(docs, 1):
            # Lấy lại bản ghi trong session hiện tại để ghi được.
            document = await db.get(Document, doc.id)
            if document is None or document.content_hash is not None:
                skipped += 1
                continue

            key = document.upload_s3_key
            try:
                raw = await storage.download_file(key)
            except Exception as e:
                logger.warning(
                    f"[{i}/{total}] doc={document.id} — không tải được file "
                    f"'{key}': {e} — bỏ qua"
                )
                failed += 1
                continue

            content_hash = hashlib.sha256(raw).hexdigest()

            if dry_run:
                logger.info(
                    f"[{i}/{total}] doc={document.id} -> {content_hash[:12]} "
                    f"(dry-run, không ghi)"
                )
                updated += 1
                continue

            document.content_hash = content_hash
            updated += 1
            logger.info(
                f"[{i}/{total}] doc={document.id} -> content_hash={content_hash[:12]}"
            )

            if updated % COMMIT_EVERY == 0:
                await db.commit()
                logger.info(f"... đã commit {updated} document")

        if not dry_run:
            await db.commit()

    logger.info(
        f"Hoàn tất: updated={updated}, skipped={skipped}, failed={failed} "
        f"(tổng {total}){' [DRY-RUN]' if dry_run else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill documents.content_hash")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ tính và in hash, không ghi vào DB",
    )
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
