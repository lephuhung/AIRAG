"""
Telegram channel adapter.

Bridges Telegram chats to the AIRAG agent. A chat must first be *linked* to a
real AIRAG account (via a one-time code minted on the web); once linked, every
message runs the LangGraph supervisor agent AS that user, so workspace/tenant
permissions are inherited automatically — no separate authorization logic here.

The webhook route (`app/api/integrations.py`) only validates the secret token and
hands the raw Telegram update to `process_update()`, which it schedules in the
background so Telegram gets an immediate 200 (avoiding delivery retries).
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta

import httpx
from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

# The active bot token for the current update, populated by process_update() from
# the DB-backed TelegramBotConfig. Falls back to the legacy .env value if unset.
_bot_token_var: ContextVar[str | None] = ContextVar("_tg_bot_token", default=None)

# Telegram hard limit per message; leave headroom for the streaming "▌" cursor.
TG_MAX_CHARS = 4096
TG_EDIT_BUDGET = 3900
# Streaming edit cadence. Telegram flood-limits edits to a SINGLE message to
# roughly ~1/sec — editing faster trips 429 (retry_after) which paradoxically
# makes the stream SLOWER. ~1s is the smooth-but-safe floor; a rejected edit is
# simply skipped (we never block waiting on a retry).
EDIT_MIN_INTERVAL_S = 1.0

HELP_TEXT = (
    "🤖 *AIRAG Bot*\n\n"
    "Các lệnh:\n"
    "• Gửi câu hỏi bất kỳ để hỏi kho tri thức\n"
    "• /link <mã> — liên kết với tài khoản AIRAG (lấy mã trên web)\n"
    "• /workspace — xem & chọn không gian làm việc\n"
    "• /new — bắt đầu cuộc trò chuyện mới\n"
    "• /whoami — xem tài khoản đang liên kết\n"
    "• /unlink — huỷ liên kết\n"
    "• /help — trợ giúp"
)


# ─────────────────────────── Telegram Bot API client ───────────────────────────

def _current_token() -> str | None:
    """Bot token for the current context, falling back to the legacy .env value."""
    return _bot_token_var.get() or (settings.TELEGRAM_BOT_TOKEN or None)


async def raw_api(method: str, payload: dict, token: str) -> dict:
    """Low-level call returning Telegram's *full* response (incl. error description).

    Used by the admin setup endpoints (getMe / setWebhook / getWebhookInfo) which
    need the raw `ok`/`description` to report status back to the UI.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/{method}", json=payload
            )
        return resp.json()
    except Exception as e:
        return {"ok": False, "description": f"request failed: {e}"}


async def _call(method: str, payload: dict, token: str | None = None) -> dict | None:
    """Call a Telegram Bot API method. Returns the `result` dict or None on error.

    Never blocks on flood control: a 429 just returns None so the streaming loop
    skips this edit and moves on (blocking on ``retry_after`` mid-stream stalls the
    whole answer). The next scheduled edit catches up with the accumulated text.
    """
    tok = token or _current_token()
    if not tok:
        logger.warning("[telegram] bot token not configured — skipping %s", method)
        return None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{tok}/{method}", json=payload
            )
            data = resp.json()
        if not data.get("ok"):
            # 'message is not modified' is benign during streaming edits.
            desc = str(data.get("description", ""))
            if "not modified" not in desc:
                logger.warning("[telegram] %s failed: %s", method, desc)
            return None
        return data.get("result")
    except Exception as e:  # network / json errors must never crash the worker
        logger.warning("[telegram] %s error: %s", method, e)
        return None


async def send_message(chat_id: str, text: str, parse_mode: str | None = None) -> int | None:
    """Send a message; returns the new message_id (or None)."""
    payload: dict = {"chat_id": chat_id, "text": _clip(text), "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    result = await _call("sendMessage", payload)
    return result.get("message_id") if result else None


async def edit_message(
    chat_id: str, message_id: int, text: str, parse_mode: str | None = None
) -> bool:
    """Edit a message; returns True if Telegram accepted the edit."""
    payload: dict = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": _clip(text),
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return await _call("editMessageText", payload) is not None


async def send_chat_action(chat_id: str, action: str = "typing") -> None:
    await _call("sendChatAction", {"chat_id": chat_id, "action": action})


def _clip(text: str, limit: int = TG_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# ──────────────────────────────── Update routing ───────────────────────────────

async def process_update(update: dict) -> None:
    """Entry point: route a single Telegram update. Owns its own DB session."""
    message = update.get("message") or update.get("edited_message")
    if not message:
        return  # ignore callback queries / channel posts / etc. for now

    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return

    from_user = message.get("from") or {}
    tg_username = from_user.get("username")
    tg_user_id = str(from_user.get("id")) if from_user.get("id") is not None else None

    async with AsyncSessionLocal() as db:
        # Resolve the bot token from the DB config (admin-managed, not .env).
        cfg = await get_bot_config(db)
        token = (cfg.bot_token if cfg else None) or (settings.TELEGRAM_BOT_TOKEN or None)
        if cfg is not None and not cfg.enabled:
            logger.info("[telegram] bot disabled — dropping update")
            return
        if not token:
            logger.warning("[telegram] no bot token configured — dropping update")
            return
        _bot_token_var.set(token)

        try:
            if text.startswith("/"):
                await _handle_command(db, chat_id, text, tg_username, tg_user_id)
            else:
                await _handle_question(db, chat_id, text, tg_user_id)
        except Exception as e:  # last-resort guard
            logger.exception("[telegram] update handling failed: %s", e)
            await send_message(chat_id, "⚠️ Có lỗi xảy ra khi xử lý yêu cầu của bạn.")


async def _handle_command(
    db, chat_id: str, text: str, tg_username: str | None, tg_user_id: str | None = None
) -> None:
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower().lstrip("/").split("@")[0]  # strip /cmd@BotName
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "start":
        if arg:
            await _redeem_code(db, chat_id, arg, tg_username, tg_user_id)
        else:
            await send_message(
                chat_id,
                "👋 Chào mừng đến với AIRAG!\n\n"
                "Để bắt đầu, hãy liên kết tài khoản: mở web AIRAG → Cài đặt → "
                "Liên kết Telegram để lấy mã, rồi gửi:\n`/link <mã>`",
                parse_mode="Markdown",
            )
        return

    if cmd == "link":
        if not arg:
            await send_message(chat_id, "Cú pháp: /link <mã lấy từ web>")
        else:
            await _redeem_code(db, chat_id, arg, tg_username, tg_user_id)
        return

    if cmd == "help":
        await send_message(chat_id, HELP_TEXT, parse_mode="Markdown")
        return

    # All remaining commands require a linked account.
    link = await _get_link(db, chat_id)
    if not link:
        await send_message(chat_id, "Bạn chưa liên kết tài khoản. Dùng /link <mã> để bắt đầu.")
        return

    if cmd == "whoami":
        await _cmd_whoami(db, chat_id, link)
    elif cmd == "new":
        link.active_session_id = None
        await db.commit()
        await send_message(chat_id, "🆕 Đã bắt đầu cuộc trò chuyện mới.")
    elif cmd == "workspace":
        await _cmd_workspace(db, chat_id, link, arg)
    elif cmd == "unlink":
        await db.delete(link)
        await db.commit()
        await send_message(chat_id, "🔌 Đã huỷ liên kết tài khoản khỏi chat này.")
    else:
        await send_message(chat_id, "Lệnh không hợp lệ. Gõ /help để xem hướng dẫn.")


# ──────────────────────────────── Linking flow ─────────────────────────────────

async def _redeem_code(
    db, chat_id: str, code: str, tg_username: str | None, tg_user_id: str | None = None
) -> None:
    from app.models.integration import TelegramLink, TelegramLinkCode
    from app.models.user import User

    code = code.strip().upper()
    result = await db.execute(
        select(TelegramLinkCode).where(TelegramLinkCode.code == code)
    )
    row = result.scalar_one_or_none()

    now = datetime.utcnow()
    if row is None or row.used or row.expires_at < now:
        await send_message(chat_id, "❌ Mã không hợp lệ hoặc đã hết hạn. Hãy lấy mã mới trên web.")
        return

    user = (await db.execute(select(User).where(User.id == row.user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        await send_message(chat_id, "❌ Tài khoản không hợp lệ hoặc chưa được kích hoạt.")
        return

    # Upsert the link (a chat may relink to a different account).
    link = await _get_link(db, chat_id)
    if link is None:
        link = TelegramLink(telegram_chat_id=chat_id, user_id=user.id)
        db.add(link)
    else:
        link.user_id = user.id
        link.active_session_id = None
        link.active_workspace_id = None
    link.telegram_username = tg_username
    if tg_user_id:
        link.telegram_user_id = tg_user_id
    row.used = True
    await db.commit()

    await send_message(
        chat_id,
        f"✅ Đã liên kết với *{user.email}*.\nGửi câu hỏi để bắt đầu, hoặc /workspace để chọn không gian.",
        parse_mode="Markdown",
    )


# ──────────────────────────────── Commands ─────────────────────────────────────

async def _cmd_whoami(db, chat_id: str, link) -> None:
    from app.models.knowledge_base import KnowledgeBase
    from app.models.user import User

    user = (await db.execute(select(User).where(User.id == link.user_id))).scalar_one_or_none()
    ws_name = "Tất cả (toàn bộ không gian có quyền)"
    if link.active_workspace_id:
        kb = (
            await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == link.active_workspace_id)
            )
        ).scalar_one_or_none()
        if kb:
            ws_name = kb.name
    email = user.email if user else "(không rõ)"
    await send_message(
        chat_id,
        f"👤 Tài khoản: *{email}*\n📁 Không gian: *{ws_name}*",
        parse_mode="Markdown",
    )


async def _cmd_workspace(db, chat_id: str, link, arg: str) -> None:
    from app.api.chat_agent import _get_accessible_workspaces
    from app.models.knowledge_base import KnowledgeBase
    from app.models.user import User

    user = (await db.execute(select(User).where(User.id == link.user_id))).scalar_one_or_none()
    if user is None:
        await send_message(chat_id, "❌ Tài khoản không còn tồn tại.")
        return

    ws_ids = await _get_accessible_workspaces(db, user)
    kbs = (
        await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(ws_ids)).order_by(KnowledgeBase.name)
        )
    ).scalars().all() if ws_ids else []

    # Selection: "/workspace all" or "/workspace <n>".
    if arg:
        if arg.lower() in ("all", "0", "tất cả", "tat ca"):
            link.active_workspace_id = None
            await db.commit()
            await send_message(chat_id, "✅ Sẽ tìm trên *tất cả* không gian có quyền.", parse_mode="Markdown")
            return
        try:
            idx = int(arg) - 1
            chosen = kbs[idx]
        except (ValueError, IndexError):
            await send_message(chat_id, "Số thứ tự không hợp lệ. Gõ /workspace để xem danh sách.")
            return
        link.active_workspace_id = chosen.id
        await db.commit()
        await send_message(chat_id, f"✅ Đã chọn không gian: *{chosen.name}*", parse_mode="Markdown")
        return

    # No arg → list them.
    if not kbs:
        await send_message(chat_id, "Bạn chưa có không gian nào.")
        return
    lines = ["📁 *Không gian khả dụng* (chọn bằng `/workspace <số>`):", "", "0. Tất cả"]
    for i, kb in enumerate(kbs, start=1):
        mark = " ✅" if link.active_workspace_id == kb.id else ""
        lines.append(f"{i}. {kb.name}{mark}")
    await send_message(chat_id, "\n".join(lines), parse_mode="Markdown")


# ──────────────────────────────── Q&A flow ─────────────────────────────────────

async def _handle_question(db, chat_id: str, question: str, tg_user_id: str | None = None) -> None:
    from app.api.chat_agent import _get_accessible_workspaces
    from app.services.agents.supervisor import get_supervisor_graph
    from app.services.agent.streaming import build_initial_state, stream_agent_events
    from app.models.chat_message import ChatMessage
    from app.models.user import User
    from app.prompts.chat import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT

    link = await _get_link(db, chat_id)
    if not link:
        await send_message(
            chat_id,
            "Bạn chưa liên kết tài khoản. Lấy mã trên web rồi gửi /link <mã>.",
        )
        return

    # Backfill the telegram user id for links created before it was tracked.
    if tg_user_id and not link.telegram_user_id:
        link.telegram_user_id = tg_user_id
        await db.commit()

    user = (await db.execute(select(User).where(User.id == link.user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        await send_message(chat_id, "❌ Tài khoản liên kết không còn hợp lệ. Hãy /link lại.")
        return

    # Resolve workspace scope (single active, or all accessible).
    if link.active_workspace_id:
        workspace_ids = [link.active_workspace_id]
    else:
        workspace_ids = await _get_accessible_workspaces(db, user)
    if not workspace_ids:
        await send_message(chat_id, "Bạn chưa có quyền truy cập không gian nào để tìm kiếm.")
        return

    # Ensure a session exists for continuity. After an idle gap this rolls over to
    # a fresh session — let the user know so they understand the prior context was
    # dropped (and that /new exists for an explicit reset).
    session, rolled_over = await _ensure_session(db, link, user, question)
    if rolled_over:
        await send_message(chat_id, "🆕 Đã lâu không trò chuyện nên mình bắt đầu chủ đề mới (gõ /new để chủ động tạo mới).")

    # Load short history for the agent (last few turns of this session).
    history = await _load_history(db, session.id, limit=10)

    # Persist the user turn.
    db.add(
        ChatMessage(
            session_id=session.id,
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            role="user",
            content=question,
            user_id=user.id,
        )
    )
    await db.commit()

    # Placeholder message we'll keep editing as tokens stream in.
    await send_chat_action(chat_id, "typing")
    placeholder_id = await send_message(chat_id, "💭 Đang tìm trong tài liệu…")

    system_prompt = DEFAULT_SYSTEM_PROMPT + HARD_SYSTEM_PROMPT
    acc: list[str] = []
    final_answer = ""
    final_sources: list[dict] = []
    last_edit = 0.0
    last_preview = ""               # skip no-op edits ("message is not modified")

    try:
        initial_state = build_initial_state(
            workspace_ids=workspace_ids,
            message=question,
            history=history,
            system_prompt=system_prompt,
            enable_thinking=False,
            db=db,
            user_id=user.id,
            session_id=str(session.id),
            document_ids=None,
        )
        graph = get_supervisor_graph()
        async for ev in stream_agent_events(graph, initial_state, channel="telegram"):
            etype = ev.get("event")
            data = ev.get("data") or {}
            if etype == "token":
                acc.append(data.get("text", ""))
                now = time.monotonic()
                if placeholder_id and now - last_edit >= EDIT_MIN_INTERVAL_S:
                    partial = "".join(acc).strip()
                    if partial:
                        # Plain text while streaming (markdown stripped) so no raw
                        # ** flickers; the final message gets full HTML formatting.
                        preview = _strip_markdown(_strip_citations(partial))
                        if preview != last_preview:
                            # Advance the clock on every attempt (success or 429) so a
                            # rejected edit just waits for the next slot — no hammering,
                            # no blocking. The accumulated text catches up next edit.
                            last_edit = now
                            if await edit_message(
                                chat_id, placeholder_id, _clip(preview + " ▌", TG_EDIT_BUDGET)
                            ):
                                last_preview = preview
            elif etype == "sources":
                final_sources = data.get("sources", []) or final_sources
            elif etype == "complete":
                final_answer = (data.get("answer") or "").strip()
                final_sources = data.get("sources", []) or final_sources
            elif etype == "error":
                final_answer = f"⚠️ {data.get('message', 'Đã xảy ra lỗi.')}"
    except Exception as e:
        logger.exception("[telegram] agent stream failed: %s", e)
        final_answer = final_answer or "⚠️ Lỗi khi tạo câu trả lời."

    if not final_answer:
        final_answer = "".join(acc).strip() or "Xin lỗi, tôi chưa tạo được câu trả lời."

    # The in-process agent stream yields ChatSourceChunk pydantic objects (the SSE
    # web path serializes them to dicts first); normalize so the JSONB persistence
    # below sees plain JSON-able dicts. We still keep the sources in DB history,
    # but they are NOT shown in Telegram.
    final_sources = [_source_to_dict(s) for s in final_sources]

    # Telegram has no clickable citation UI, so the inline [a3x9]/[MEM-..]/[IMG-..]
    # markers and a source list are just noise — strip them for a clean message.
    body = _strip_citations(final_answer)
    await _deliver_final(chat_id, placeholder_id, body)

    # Persist the assistant turn (best-effort).
    try:
        db.add(
            ChatMessage(
                session_id=session.id,
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                role="assistant",
                content=final_answer,
                sources=final_sources or None,
                user_id=user.id,
            )
        )
        await db.commit()
    except Exception:
        await db.rollback()


async def _ensure_session(db, link, user, first_message: str) -> tuple:
    """Return ``(session, rolled_over)`` for this chat.

    ``rolled_over`` is True only when an existing active session was replaced
    because it went idle (so the caller can show a "new topic" hint). It stays
    False for the very first conversation of a chat — there's nothing to roll over.
    """
    from app.models.chat_session import ChatSession

    if link.active_session_id:
        session = (
            await db.execute(
                select(ChatSession).where(
                    ChatSession.id == link.active_session_id,
                    ChatSession.user_id == user.id,
                )
            )
        ).scalar_one_or_none()
        # Reuse the active session only while it's still "warm". After
        # TELEGRAM_SESSION_IDLE_MINUTES of silence we transparently start a fresh
        # one so a new (unrelated) question isn't answered with the previous
        # conversation's context. The session vanishing also falls through here.
        if session and not await _session_is_idle(db, session.id):
            return session, False
        rolled_over = session is not None  # had a real session that went idle
    else:
        rolled_over = False

    title = (first_message[:60] + "…") if len(first_message) > 60 else first_message
    session = ChatSession(title=title or "Telegram", user_id=user.id, source="telegram")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    link.active_session_id = session.id
    await db.commit()
    return session, rolled_over


async def _session_is_idle(db, session_id) -> bool:
    """True if the session's most recent message is older than the idle window.

    Telegram has no separate-conversation UI, so an active session would otherwise
    accumulate forever and old turns would bleed into new answers. After
    ``TELEGRAM_SESSION_IDLE_MINUTES`` of silence the chat transparently rolls over
    to a fresh session. A brand-new/empty session is never considered idle.
    """
    from app.models.chat_message import ChatMessage

    idle_minutes = settings.TELEGRAM_SESSION_IDLE_MINUTES
    if idle_minutes <= 0:
        return False  # auto-expiry disabled
    last_at = (
        await db.execute(
            select(ChatMessage.created_at)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if last_at is None:
        return False  # no messages yet — keep the session
    return datetime.utcnow() - last_at > timedelta(minutes=idle_minutes)


async def _load_history(db, session_id, limit: int = 10) -> list[dict]:
    from app.models.chat_message import ChatMessage

    rows = (
        await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    rows = list(reversed(rows))
    return [{"role": r.role, "content": r.content} for r in rows if r.content]


def _source_to_dict(s) -> dict:
    """Coerce a source (dict or ChatSourceChunk pydantic model) to a plain dict.

    The in-process Telegram agent stream yields pydantic ``ChatSourceChunk``
    objects, while the SSE web path receives already-serialized dicts.
    ``model_dump(mode="json")`` also turns the ``document_id`` UUID into a string
    so the value is safe for the ``ChatMessage.sources`` JSONB column.
    """
    if isinstance(s, dict):
        return s
    dump = getattr(s, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


# Inline citation tokens the answer embeds for the web UI's clickable badges:
#   [a3x9]  [a3x9, b2m7]  [IMG-p4f2]  [MEM-xxx]  [1]  (mirrors ChatPanel CITATION_RE)
# Telegram can't render those, so we strip them from the message text.
_CITATION_RE = re.compile(
    r"\[\s*(?:(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+)"
    r"(?:\s*,\s*(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+))*|\d+)"
    r"(?:\s*,\s*(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+|\d+))*\s*\]"
)

# Some answers cite the raw document/chunk UUID, e.g. [25a7521c-3968-4b13-bb83-...]
# (optionally grouped/comma-separated). These are too long for _CITATION_RE, so
# strip them separately. Also catches a bare UUID accidentally left inline.
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUID_CITATION_RE = re.compile(rf"\[\s*{_UUID}(?:\s*,\s*{_UUID})*\s*\]|\b{_UUID}\b")


def _strip_citations(text: str) -> str:
    """Remove inline citation markers and tidy the resulting whitespace.

    Citations are useful only in the web UI (clickable badges); in Telegram they
    are plain-text noise, so the bot answers without any source references.
    """
    if not text:
        return text
    text = _UUID_CITATION_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"[ \t]+([.,;:!?…)])", r"\1", text)  # drop space left before punctuation
    text = re.sub(r"\(\s+\)", "", text)  # empty parens left by removed refs
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse double spaces
    text = re.sub(r" +\n", "\n", text)  # trailing spaces before newline
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    return text.strip()


# ───────────────────────── Markdown → Telegram formatting ───────────────────────
# The agent answers in CommonMark (**bold**, # headings, - lists, `code`, links),
# which Telegram does NOT render — it would show the literal ** etc. We convert the
# common subset to Telegram-flavoured HTML (parse_mode="HTML"), and fall back to a
# plain-text rendering if Telegram ever rejects the HTML.

# Leave room for the added <b>/<i>/… tags so a converted chunk stays < TG_MAX_CHARS.
TG_HTML_CHUNK = 3500


def _md_to_telegram_html(md: str) -> str:
    """Convert a CommonMark subset to Telegram HTML (<b>/<i>/<s>/<code>/<pre>/<a>)."""
    if not md:
        return ""

    # 1) Stash code blocks/spans so their contents aren't treated as markdown.
    stash: list[str] = []

    def _stash(rendered: str) -> str:
        stash.append(rendered)
        return f"\x00{len(stash) - 1}\x00"

    md = re.sub(
        r"```[^\n]*\n?(.*?)```",
        lambda m: _stash(f"<pre>{_html.escape(m.group(1).rstrip())}</pre>"),
        md,
        flags=re.DOTALL,
    )
    md = re.sub(
        r"`([^`\n]+)`",
        lambda m: _stash(f"<code>{_html.escape(m.group(1))}</code>"),
        md,
    )

    # 2) Escape the remaining text, then apply inline + block formatting.
    md = _html.escape(md)

    # links [text](url)
    md = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        md,
    )
    # bold then strike (run before single-char italic so ** isn't eaten by *)
    md = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md, flags=re.DOTALL)
    md = re.sub(r"__(.+?)__", r"<b>\1</b>", md, flags=re.DOTALL)
    md = re.sub(r"~~(.+?)~~", r"<s>\1</s>", md, flags=re.DOTALL)
    # italic *x* / _x_ (guards avoid list bullets and intra-word underscores)
    md = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", md)
    md = re.sub(r"(?<![_\w])_(?!\s)(.+?)(?<!\s)_(?![_\w])", r"<i>\1</i>", md)

    # block-level, line by line
    out: list[str] = []
    for line in md.split("\n"):
        if re.match(r"\s{0,3}([-*_])(\s*\1){2,}\s*$", line):
            out.append("──────────")  # horizontal rule
            continue
        h = re.match(r"\s{0,3}#{1,6}\s+(.*)$", line)
        if h:
            content = h.group(1).strip()
            out.append(content if "<b>" in content else f"<b>{content}</b>")
            continue
        b = re.match(r"(\s*)[-*+]\s+(.*)$", line)
        if b:
            out.append(f"{b.group(1)}• {b.group(2)}")
            continue
        out.append(line)
    md = "\n".join(out)

    # 3) Restore stashed code spans.
    md = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], md)
    return md


def _strip_markdown(md: str) -> str:
    """Plain-text rendering of the markdown subset (streaming + HTML fallback)."""
    if not md:
        return md
    md = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", md, flags=re.DOTALL)
    md = re.sub(r"`([^`\n]+)`", r"\1", md)
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md, flags=re.DOTALL)
    md = re.sub(r"__(.+?)__", r"\1", md, flags=re.DOTALL)
    md = re.sub(r"~~(.+?)~~", r"\1", md, flags=re.DOTALL)
    md = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"\1", md)
    md = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", md)
    md = re.sub(r"^\s{0,3}#{1,6}\s+", "", md, flags=re.MULTILINE)
    md = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", md, flags=re.MULTILINE)
    return md


def _chunk_markdown(text: str, budget: int) -> list[str]:
    """Split on line boundaries so a chunk never cuts mid-line (keeps tags intact)."""
    if len(text) <= budget:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > budget:  # a single very long line — hard split
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:budget])
            line = line[budget:]
        if len(cur) + len(line) + 1 > budget:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


async def _send_formatted(chat_id: str, md_text: str) -> None:
    """Send markdown as Telegram HTML, falling back to plain text on rejection."""
    if await send_message(chat_id, _md_to_telegram_html(md_text), parse_mode="HTML") is None:
        await send_message(chat_id, _strip_markdown(md_text))


async def _edit_formatted(chat_id: str, message_id: int, md_text: str) -> None:
    if not await edit_message(chat_id, message_id, _md_to_telegram_html(md_text), parse_mode="HTML"):
        await edit_message(chat_id, message_id, _strip_markdown(md_text))


async def _deliver_final(chat_id: str, placeholder_id: int | None, body: str) -> None:
    """Write the final answer (rendered as Telegram HTML), chunking if too long."""
    chunks = _chunk_markdown(body, TG_HTML_CHUNK)
    if placeholder_id:
        await _edit_formatted(chat_id, placeholder_id, chunks[0])
    else:
        await _send_formatted(chat_id, chunks[0])
    for chunk in chunks[1:]:
        await _send_formatted(chat_id, chunk)


# ─────────────────────────────── Helpers / DB ──────────────────────────────────

async def _get_link(db, chat_id: str):
    from app.models.integration import TelegramLink

    return (
        await db.execute(
            select(TelegramLink).where(TelegramLink.telegram_chat_id == chat_id)
        )
    ).scalar_one_or_none()


async def get_bot_config(db):
    """Return the singleton TelegramBotConfig row (id == 1), or None if unset."""
    from app.models.integration import TelegramBotConfig

    return (
        await db.execute(
            select(TelegramBotConfig).where(TelegramBotConfig.id == 1)
        )
    ).scalar_one_or_none()


# ─────────────────────────── Admin setup (web side) ─────────────────────────────

async def fetch_bot_identity(token: str) -> dict | None:
    """getMe — returns {id, username, ...} for the given token, or None if invalid."""
    data = await raw_api("getMe", {}, token)
    return data.get("result") if data.get("ok") else None


async def register_webhook(token: str, url: str, secret: str) -> dict:
    """setWebhook — returns the raw Telegram response (ok + description)."""
    payload: dict = {"url": url, "secret_token": secret, "drop_pending_updates": False}
    return await raw_api("setWebhook", payload, token)


async def fetch_webhook_info(token: str) -> dict:
    """getWebhookInfo — returns the raw Telegram response."""
    return await raw_api("getWebhookInfo", {}, token)


# ─────────────────────────── Link-code minting (web side) ───────────────────────

async def create_link_code(db, user_id: uuid.UUID) -> tuple[str, datetime]:
    """Mint a fresh one-time link code for a user. Returns (code, expires_at)."""
    from app.core.security import generate_link_code
    from app.models.integration import TelegramLinkCode

    code = generate_link_code()
    expires_at = datetime.utcnow() + timedelta(minutes=settings.TELEGRAM_LINK_CODE_TTL_MINUTES)
    db.add(TelegramLinkCode(code=code, user_id=user_id, expires_at=expires_at))
    await db.commit()
    return code, expires_at


def build_deep_link(code: str, bot_username: str | None) -> str | None:
    """t.me deep link that auto-fills /start <code>, if the bot username is known."""
    username = bot_username or settings.TELEGRAM_BOT_USERNAME
    if not username:
        return None
    return f"https://t.me/{username}?start={code}"
