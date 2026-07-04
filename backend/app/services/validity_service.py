"""
Suy TRẠNG THÁI HIỆU LỰC của văn bản trong kho từ kết quả ``validity_extractor``.

Gọi ở parse_worker (sau khi classifier set ``document_number``) và ở
``scripts/backfill_validity.py`` cho kho cũ. Xử lý CẢ HAI chiều thời gian:

- chiều xuôi: văn bản mới B tuyên bố "thay thế/bãi bỏ/hết hiệu lực A" →
  nếu A đã có trong kho, đánh dấu A ngay;
- chiều ngược: A được upload SAU B → lúc parse A, quét ``validity_events``
  của các văn bản có sẵn xem có ai từng tuyên bố nhắm vào số hiệu của A.

Phạm vi so khớp: TRONG CÙNG WORKSPACE — kho là per-workspace, đánh dấu chéo
workspace vừa rò rỉ thông tin giữa tenant vừa sai ngữ cảnh kho.

Trạng thái (documents.validity_status):
    unknown            → chưa có dữ kiện
    effective          → trích được điều khoản hiệu lực, chưa ai thay thế
    partially_amended  → bị văn bản khác sửa/bãi bỏ MỘT PHẦN
    superseded         → bị thay thế/hết hiệu lực TOÀN PHẦN
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.services.validity_extractor import extract_validity

logger = logging.getLogger(__name__)

# Sự kiện scope=full hạ văn bản đích xuống superseded; partial chỉ hạ xuống
# partially_amended và không bao giờ ghi đè superseded.
_FULL_KINDS = {"thay_the", "bai_bo", "het_hieu_luc"}


async def apply_validity(
    db: AsyncSession, document: Document, markdown: str, commit: bool = True
) -> None:
    """Trích hiệu lực từ ``markdown`` của ``document`` và lan trạng thái.

    Best-effort như AuditService: lỗi ở đây không được làm fail pipeline parse —
    caller nên bọc try/except. Tự commit trừ khi ``commit=False`` (dry-run của
    backfill: caller flush + rollback để xem trước thay đổi).
    """
    info = extract_validity(markdown)
    events = [e.as_dict() for e in info.events]

    document.validity_events = events
    if info.effective_date:
        document.effective_date = (
            document.published_date
            if info.effective_date == "sign_date" and document.published_date
            else info.effective_date
        )
        if document.validity_status in (None, "unknown"):
            document.validity_status = "effective"

    # Chiều xuôi: sự kiện của document → các văn bản đích đã có trong kho
    for event in events:
        target = await _find_by_number(db, event["target_number"], document.workspace_id)
        if target is None or target.id == document.id:
            continue
        _mark_target(target, event, source=document)

    # Chiều ngược: văn bản có sẵn từng tuyên bố nhắm vào document này
    if document.document_number:
        rows = (await db.execute(
            select(Document).where(
                Document.workspace_id == document.workspace_id,
                Document.id != document.id,
                Document.validity_events.isnot(None),
            )
        )).scalars().all()
        for other in rows:
            for event in other.validity_events or []:
                if _same_number(event.get("target_number"), document.document_number):
                    _mark_target(document, event, source=other)

    if commit:
        await db.commit()
    logger.info(
        f"[validity] doc={document.id} number={document.document_number!r} "
        f"status={document.validity_status} effective={document.effective_date} "
        f"events={len(info.events)}"
    )


def _mark_target(target: Document, event: dict, source: Document) -> None:
    """Áp một tuyên bố hiệu lực lên văn bản đích (không tự commit)."""
    if event.get("scope") == "full" and event.get("kind") in _FULL_KINDS:
        target.validity_status = "superseded"
        target.superseded_by_number = source.document_number
        target.superseded_by_document_id = source.id
        logger.info(
            f"[validity] {target.document_number!r} ({target.id}) superseded by "
            f"{source.document_number!r} ({event['kind']})"
        )
    elif target.validity_status != "superseded":
        target.validity_status = "partially_amended"


async def _find_by_number(
    db: AsyncSession, number: str | None, workspace_id
) -> Document | None:
    if not number:
        return None
    return (await db.execute(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.document_number.ilike(number),
        ).limit(1)
    )).scalar_one_or_none()


def _same_number(a: str | None, b: str | None) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())
