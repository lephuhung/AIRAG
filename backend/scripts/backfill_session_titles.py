"""
Backfill ``chat_sessions.title`` cho các session cũ bị kẹt ở tiêu đề tạm.

Bối cảnh (xem app/api/chat_session.py): tiêu đề phiên chat đáng lẽ được nâng cấp
thành nhãn chủ đề (``topic_label``) do LLM sinh ra sau lượt hỏi đáp đầu tiên. Với
các phiên CŨ (tạo trước khi vá), bước nâng cấp này bị bỏ qua nên tiêu đề kẹt ở lát
cắt thô 30 ký tự của tin nhắn đầu (vd. ``"@OPhim cho tôi thông tin về cá"``).

Script này sửa các phiên đó: với mỗi phiên đã có ``topic_label`` lưu sẵn trong
``chat_exchange_summaries`` (lượt đầu tiên), gán lại ``title = topic_label``.

Vì hệ thống KHÔNG có chức năng đổi tên session thủ công (không có endpoint PUT/PATCH
nào, frontend cũng không gọi), nên MỌI title trong DB đều là tự sinh — ta có thể
ghi đè thẳng mà không sợ đè nhầm tên người dùng tự đặt. Không cần dò/chuẩn hoá gì.

An toàn:
  - Bỏ qua phiên chưa có summary (không có nhãn để dùng) — in cảnh báo.
  - Idempotent: phiên đã trùng nhãn thì bỏ qua; chạy lại nhiều lần không gây hại.

Chạy (cần Postgres đang chạy):
    python -m scripts.backfill_session_titles                 # thực thi
    python -m scripts.backfill_session_titles --dry-run       # chỉ in, không ghi
    python -m scripts.backfill_session_titles --session-id <uuid>   # chỉ 1 phiên
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.chat_session import ChatSession
from app.models.exchange_summary import ExchangeSummary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_session_titles")

# Ghi DB sau mỗi N phiên để không giữ transaction quá lâu.
COMMIT_EVERY = 50


async def _first_topic_label(db, session_id) -> str | None:
    """Nhãn chủ đề của lượt hỏi đáp đầu tiên (exchange_index nhỏ nhất)."""
    res = await db.execute(
        select(ExchangeSummary.topic_label)
        .where(ExchangeSummary.session_id == session_id)
        .order_by(ExchangeSummary.exchange_index.asc())
        .limit(1)
    )
    label = res.scalar_one_or_none()
    return label.strip() if label else None


async def backfill(dry_run: bool = False, only_session_id: str | None = None) -> None:
    async with async_session_maker() as db:
        stmt = select(ChatSession)
        if only_session_id:
            stmt = stmt.where(ChatSession.id == only_session_id)
        result = await db.execute(stmt)
        sessions = result.scalars().all()

    total = len(sessions)
    logger.info(f"Tìm thấy {total} phiên cần kiểm tra")
    if total == 0:
        return

    updated = 0
    skipped_no_summary = 0
    skipped_already_ok = 0

    async with async_session_maker() as db:
        for i, s in enumerate(sessions, 1):
            session = await db.get(ChatSession, s.id)
            if session is None:
                continue

            topic = await _first_topic_label(db, session.id)
            if not topic:
                skipped_no_summary += 1
                logger.info(
                    f"[{i}/{total}] session={session.id} — chưa có topic_label, bỏ qua "
                    f"(title hiện tại: {session.title!r})"
                )
                continue

            if session.title == topic:
                skipped_already_ok += 1
                continue

            # topic_label được DB giới hạn String(255); cắt an toàn phòng khi dài.
            new_title = topic[:255]

            if dry_run:
                logger.info(
                    f"[{i}/{total}] session={session.id} (dry-run) "
                    f"{session.title!r} -> {new_title!r}"
                )
                updated += 1
                continue

            logger.info(
                f"[{i}/{total}] session={session.id} "
                f"{session.title!r} -> {new_title!r}"
            )
            session.title = new_title
            updated += 1

            if updated % COMMIT_EVERY == 0:
                await db.commit()
                logger.info(f"... đã commit {updated} phiên")

        if not dry_run:
            await db.commit()

    logger.info(
        f"Hoàn tất: updated={updated}, skipped_no_summary={skipped_no_summary}, "
        f"skipped_already_ok={skipped_already_ok} "
        f"(tổng {total}){' [DRY-RUN]' if dry_run else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill chat_sessions.title từ topic_label của lượt đầu"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in thay đổi, không ghi vào DB",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Chỉ xử lý một session id cụ thể (mặc định: tất cả)",
    )
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run, only_session_id=args.session_id))


if __name__ == "__main__":
    main()
