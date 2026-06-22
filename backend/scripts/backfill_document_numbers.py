"""
Backfill ``documents.document_number`` (và ``published_date`` nếu trống) cho các
document cũ bị trích xuất metadata SAI.

Bối cảnh: extractor metadata (LLM nhỏ trong ``document_type_classifier.py``) đôi
khi để LỌT TIÊU ĐỀ vào trường số hiệu — ví dụ Luật An ninh mạng bị gán
``document_number = "Luật An ninh mạng"`` thay vì ``"24/2018/QH14"``. Hệ quả:
``resolve_document_reference`` không match được theo số hiệu → rơi vào vector
fallback → chọn nhầm văn bản (vd Nghị định 53/2022/NĐ-CP). Lỗi gốc đã được vá ở
classifier (loại số hiệu không có chữ số + regex khôi phục từ header); script này
sửa các DÒNG ĐÃ TỒN TẠI bằng cùng logic regex, đọc lại markdown gốc từ MinIO.

Chạy (cần Postgres + MinIO đang chạy):
    python -m scripts.backfill_document_numbers --dry-run   # chỉ in, không ghi
    python -m scripts.backfill_document_numbers             # thực thi

An toàn để chạy lại (idempotent): chỉ đụng tới document có ``document_number``
NULL hoặc KHÔNG chứa chữ số (= không phải số hiệu hợp lệ).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.document import Document
from app.services.document_type_classifier import _recover_doc_number
from app.services.storage_service import get_storage_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_document_numbers")

COMMIT_EVERY = 50

# Ngày ban hành: "Hà Nội, ngày 12 tháng 6 năm 2018" hoặc "ngày 12/6/2018".
_DATE_WORDS = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
_DATE_SLASH = re.compile(r"ngày\s+(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", re.IGNORECASE)


def _is_valid_number(n: str | None) -> bool:
    return bool(n) and any(ch.isdigit() for ch in n)


def _recover_date(header: str) -> str | None:
    # Same guard as the number: a citation date in "Căn cứ …" is not THIS doc's
    # publication date, so only look before the preamble.
    region = header
    _cut = re.search(r"căn\s+cứ", region, re.IGNORECASE)
    if _cut:
        region = region[: _cut.start()]
    m = _DATE_WORDS.search(region) or _DATE_SLASH.search(region)
    if not m:
        return None
    d, mo, y = m.group(1), m.group(2), m.group(3)
    return f"{int(d):02d}/{int(mo):02d}/{y}"


async def backfill(dry_run: bool = False) -> None:
    storage = get_storage_service()

    async with async_session_maker() as db:
        result = await db.execute(select(Document))
        docs = result.scalars().all()

    # Lọc các document có số hiệu KHÔNG hợp lệ (null hoặc không có chữ số).
    targets = [d for d in docs if not _is_valid_number(d.document_number)]
    logger.info(f"Tổng {len(docs)} document; {len(targets)} có document_number không hợp lệ")
    if not targets:
        return

    fixed_num = fixed_date = skipped = failed = 0

    async with async_session_maker() as db:
        for i, doc in enumerate(targets, 1):
            try:
                key = doc.markdown_s3_key or storage._make_key(doc.workspace_id, doc.id)
                try:
                    md = await storage.download_markdown(key)
                except Exception as e:
                    logger.warning(f"[{i}/{len(targets)}] doc={doc.id} không tải được markdown ({key}): {e}")
                    failed += 1
                    continue
                header = (md or "")[:1500]

                new_num = _recover_doc_number(md or "")
                new_date = _recover_date(header) if not doc.published_date else None

                if not new_num and not new_date:
                    logger.info(
                        f"[{i}/{len(targets)}] doc={doc.id} title={doc.document_title!r} "
                        f"— không khôi phục được số hiệu từ header (cũ={doc.document_number!r})"
                    )
                    skipped += 1
                    continue

                logger.info(
                    f"[{i}/{len(targets)}] doc={doc.id} title={doc.document_title!r}: "
                    f"number {doc.document_number!r} -> {new_num!r}"
                    + (f", date None -> {new_date!r}" if new_date else "")
                )

                if not dry_run:
                    fresh = await db.get(Document, doc.id)
                    if fresh is None:
                        continue
                    if new_num:
                        fresh.document_number = new_num
                        fixed_num += 1
                    if new_date:
                        fresh.published_date = new_date
                        fixed_date += 1
                    if i % COMMIT_EVERY == 0:
                        await db.commit()
                else:
                    if new_num:
                        fixed_num += 1
                    if new_date:
                        fixed_date += 1
            except Exception as e:
                logger.warning(f"[{i}/{len(targets)}] doc={doc.id} lỗi: {e}")
                failed += 1

        if not dry_run:
            await db.commit()

    logger.info(
        f"Xong. Sửa số hiệu: {fixed_num}, sửa ngày: {fixed_date}, "
        f"bỏ qua (không khôi phục được): {skipped}, lỗi: {failed}"
        + ("  [DRY-RUN — chưa ghi DB]" if dry_run else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill document_number/published_date sai trong KB")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in, không ghi DB")
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
