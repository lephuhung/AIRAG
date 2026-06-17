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

import logging
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
# Don't hammer Telegram's edit endpoint while streaming tokens.
EDIT_MIN_INTERVAL_S = 1.2

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
    """Call a Telegram Bot API method. Returns the `result` dict or None on error."""
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


async def edit_message(chat_id: str, message_id: int, text: str, parse_mode: str | None = None) -> None:
    payload: dict = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": _clip(text),
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    await _call("editMessageText", payload)


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

    # Ensure a session exists for continuity.
    session = await _ensure_session(db, link, user, question)

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
        async for ev in stream_agent_events(graph, initial_state):
            etype = ev.get("event")
            data = ev.get("data") or {}
            if etype == "token":
                acc.append(data.get("text", ""))
                now = time.monotonic()
                if placeholder_id and now - last_edit >= EDIT_MIN_INTERVAL_S:
                    last_edit = now
                    partial = "".join(acc).strip()
                    if partial:
                        await edit_message(chat_id, placeholder_id, _clip(partial + " ▌", TG_EDIT_BUDGET))
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

    body = final_answer + _render_sources(final_sources)
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


async def _ensure_session(db, link, user, first_message: str):
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
        if session:
            return session

    title = (first_message[:60] + "…") if len(first_message) > 60 else first_message
    session = ChatSession(title=title or "Telegram", user_id=user.id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    link.active_session_id = session.id
    await db.commit()
    return session


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


def _render_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["", "📚 Nguồn:"]
    for s in sources[:5]:
        idx = s.get("index", "")
        fname = s.get("source_file") or s.get("document_id") or ""
        page = s.get("page_no")
        loc = f" (tr.{page})" if page else ""
        tag = f"[{idx}] " if idx else "• "
        lines.append(f"{tag}{fname}{loc}")
    return "\n".join(lines)


async def _deliver_final(chat_id: str, placeholder_id: int | None, body: str) -> None:
    """Write the final answer, editing the placeholder and chunking if too long."""
    if len(body) <= TG_MAX_CHARS:
        if placeholder_id:
            await edit_message(chat_id, placeholder_id, body)
        else:
            await send_message(chat_id, body)
        return

    chunks = [body[i : i + TG_EDIT_BUDGET] for i in range(0, len(body), TG_EDIT_BUDGET)]
    if placeholder_id:
        await edit_message(chat_id, placeholder_id, chunks[0])
    else:
        await send_message(chat_id, chunks[0])
    for chunk in chunks[1:]:
        await send_message(chat_id, chunk)


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
