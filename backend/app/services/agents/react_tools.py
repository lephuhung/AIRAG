"""
ReAct Tool Registry (RAG group) — two-tier tool definitions
============================================================

Step 2 of the ReAct-executor migration (see docs/react_executor_design.md).

The RAG tools in ``app/services/agent/tools.py`` mix two kinds of arguments:
  - **semantic** args the LLM should choose   → query, reference, section, entity, ...
  - **context** args supplied by the runtime  → db, workspace_ids, document_ids,
                                                 existing_citation_ids, user_id, ...

This module separates them:

  * ``RAG_TOOL_SCHEMAS`` — OpenAI ``tools=`` schemas exposing ONLY semantic args.
    The ``description``/``enum`` text is where domain routing rules live (replacing
    regex like ``_NAMED_DOC_PATTERN`` / ``_MULTI_DOC_PATTERN``).
  * ``TOOL_REGISTRY``    — name → async adapter(args, ctx). Each adapter injects the
    context args from :class:`ToolContext`, calls the real tool, and normalises the
    result into a common envelope :func:`tool_result`.

Three new agentic tools live here too: ``recall_memory``, ``save_memory``,
``ask_user`` — so the LLM decides *what to recall / store / clarify* instead of
regex heuristics.

The executor node (step 3) consumes this registry; this module has no graph wiring.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Tool context (runtime args the LLM never sees) + result envelope
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ToolContext:
    """Runtime context injected into every tool call by the executor.

    ``db`` is intentionally NOT stored here: it lives in a ContextVar
    (``get_current_db()``) and is fetched at call time, matching how the legacy
    rag_agent obtains its session.
    """

    workspace_ids: list = field(default_factory=list)
    document_ids: list | None = None           # mutated by resolve_document_reference (auto-scope)
    uploaded_document_ids: list | None = None   # files the user attached this turn — CONSTANT,
                                                 # never mutated by resolve. Lets search_documents
                                                 # tell "the uploaded file" apart from "a resolved
                                                 # named doc" and from "the whole knowledge base".
    user_id: uuid.UUID | None = None
    session_id: str | None = None
    existing_citation_ids: dict = field(default_factory=dict)  # keyed by ChatSourceChunk.index
    top_k: int = 8
    state: dict = field(default_factory=dict)  # SupervisorState — for push_event / ask_user

    def remember_citations(self, sources: list) -> None:
        """Record citation indices so later searches don't reuse the same IDs."""
        for s in sources or []:
            idx = getattr(s, "index", None)
            if idx is None and isinstance(s, dict):
                idx = s.get("index")
            if idx is not None:
                self.existing_citation_ids[str(idx)] = True


def tool_result(
    summary: str,
    *,
    sources: list | None = None,
    images: list | None = None,
    data: dict | None = None,
) -> dict:
    """Common envelope every adapter returns.

    - ``summary`` : text fed back to the LLM as the tool result (what was found).
    - ``sources`` : ChatSourceChunk list the executor accumulates for citations.
    - ``images``  : ChatImageRef list.
    - ``data``    : extra structured payload (candidates, document_ids, found, ...).
    """
    return {
        "summary": summary or "",
        "sources": sources or [],
        "images": images or [],
        "data": data or {},
    }


def _db():
    from app.services.agent.streaming import get_current_db
    return get_current_db()


# ──────────────────────────────────────────────────────────────────────────
# Adapters — RAG retrieval tools
# ──────────────────────────────────────────────────────────────────────────

async def _adapt_search_documents(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import search_documents

    query = (args.get("query") or "").strip()
    mode = args.get("mode") or "hybrid"
    if mode not in ("vector", "kg", "hybrid"):
        mode = "hybrid"
    if not query:
        return tool_result("Lỗi: thiếu 'query' cho search_documents.")

    # ── Scope resolution (Phase 1: separate "uploaded file" from "knowledge base") ──
    # scope="knowledge_base" → search the WHOLE corpus, ignore any attached/resolved
    #   doc filter. This is the escape hatch that makes "đối chiếu file với KB" and
    #   "hỏi đáp xuyên file + KB" possible (previously impossible: a non-empty
    #   document_ids ALWAYS forced a scoped search, so the KB was never reached).
    # scope="uploaded" → restrict to the files the user attached this turn.
    # scope omitted → backward-compatible: scope to whatever doc context exists
    #   (resolved named doc, else attached file), else search the KB.
    scope = (args.get("scope") or "").strip().lower() or None
    if scope == "knowledge_base":
        doc_ids, scoped = None, False
    elif scope == "uploaded":
        doc_ids = ctx.uploaded_document_ids or ctx.document_ids
        scoped = bool(doc_ids)
    else:
        doc_ids = ctx.document_ids
        scoped = bool(doc_ids)

    res = await search_documents(
        query=query,
        top_k=ctx.top_k,
        workspace_ids=ctx.workspace_ids,
        existing_citation_ids=set(ctx.existing_citation_ids.keys()),
        db=_db(),
        document_ids=doc_ids,
        search_mode=mode,
        scoped_to_documents=scoped,
    )
    sources = res.get("sources", []) or []
    images = res.get("images", []) or []
    ctx.remember_citations(sources)
    text = (res.get("context_text") or "").strip()
    if not text and not sources:
        text = f"Không tìm thấy nội dung phù hợp cho '{query}'."
    summary = text or f"Tìm thấy {len(sources)} đoạn liên quan đến '{query}'."
    return tool_result(summary, sources=sources, images=images,
                       data={"kg_summaries": res.get("kg_summaries", [])})


async def _adapt_resolve_document_reference(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import resolve_document_reference

    reference = (args.get("reference") or "").strip()
    if not reference:
        return tool_result("Lỗi: thiếu 'reference' cho resolve_document_reference.")

    # Pass the FULL user question as `topic` so resolve can use its subject content
    # (not just the bare doc reference the LLM extracted) to find/disambiguate the
    # right document — the question itself carries strong signal.
    topic = (ctx.state.get("rewritten_query") or ctx.state.get("original_query") or "").strip()

    res = await resolve_document_reference(
        reference=reference,
        workspace_ids=ctx.workspace_ids,
        db=_db(),
        topic=topic or None,
    )
    candidates = res.get("candidates", []) or []
    ambiguous = bool(res.get("ambiguous"))
    # Auto-scope subsequent searches when exactly one unambiguous match — the
    # LLM does not need to handle the UUID; the next search_documents/section
    # call is automatically restricted to this document.
    if candidates and not ambiguous:
        try:
            ctx.document_ids = [uuid.UUID(str(candidates[0]["document_id"]))]
        except (ValueError, KeyError, TypeError):
            ctx.document_ids = [candidates[0].get("document_id")] if candidates else None
    return tool_result(
        res.get("message", ""),
        data={"candidates": candidates, "ambiguous": ambiguous,
              "resolved_document_ids": [str(d) for d in (ctx.document_ids or [])]},
    )


async def _adapt_search_document_section(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import search_document_section

    section = (args.get("section_reference") or "").strip()
    if not section:
        return tool_result("Lỗi: thiếu 'section_reference' (vd 'Điều 5', 'Chương II').")

    res = await search_document_section(
        section_reference=section,
        workspace_ids=ctx.workspace_ids,
        document_ids=ctx.document_ids,
    )
    sources = res.get("sources", []) or []
    ctx.remember_citations(sources)
    return tool_result(res.get("text", ""), sources=sources)


async def _adapt_query_knowledge_graph(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import query_knowledge_graph
    from app.api.chat_agent import _generate_citation_id
    from app.schemas.rag import ChatSourceChunk

    entity = (args.get("entity") or "").strip()
    if not entity:
        return tool_result("Lỗi: thiếu 'entity' cho query_knowledge_graph.")
    res = await query_knowledge_graph(entity=entity, workspace_ids=ctx.workspace_ids, db=_db())
    text = (res.get("text") or "").strip()
    if not text:
        return tool_result(
            f"Không tìm thấy thông tin về '{entity}' trong knowledge graph.",
            data={"kg": True},
        )

    # Register the KG answer as a REAL citable source so the model cites a valid
    # id (e.g. [a3z9]) instead of inventing a marker like [idKG] that the frontend
    # cannot resolve and leaks as raw text. KG output isn't tied to one document
    # → use a nil document_id; the frontend renders kg sources as a "KG-…" chip.
    cid = _generate_citation_id(set(ctx.existing_citation_ids.keys()))
    chunk = ChatSourceChunk(
        index=cid,
        chunk_id=f"kg_{cid}",
        content=text,
        document_id=uuid.UUID(int=0),
        page_no=0,
        heading_path=[],
        score=1.0,
        source_type="kg",
        source_file=None,
    )
    ctx.remember_citations([chunk])
    summary = f"Nguồn [{cid}] (Knowledge Graph) cho '{entity}':\n{text}"
    return tool_result(summary, sources=[chunk], data={"kg": True})


async def _adapt_list_documents(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import list_documents

    res = await list_documents(workspace_ids=ctx.workspace_ids, db=_db())
    return tool_result(res.get("text", ""), data={"document_count": res.get("document_count", 0)})


async def _adapt_search_abbreviation(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import search_abbreviation

    abbr = (args.get("abbreviation") or "").strip()
    if not abbr:
        return tool_result("Lỗi: thiếu 'abbreviation' cho search_abbreviation.")
    res = await search_abbreviation(abbreviation=abbr, workspace_ids=ctx.workspace_ids, db=_db())
    return tool_result(res.get("text", ""), data={"found": res.get("found", False)})


async def _adapt_search_documents_number(args: dict, ctx: ToolContext) -> dict:
    from app.services.agent.tools import search_documents_number

    query = (args.get("query") or "").strip()
    if not query:
        return tool_result("Lỗi: thiếu 'query' (số văn bản) cho search_documents_number.")
    res = await search_documents_number(query=query, workspace_ids=ctx.workspace_ids, db=_db())
    return tool_result(res.get("text", ""), data={"documents": res.get("documents", [])})


async def _adapt_get_document_content(args: dict, ctx: ToolContext) -> dict:
    """Fetch full parsed content of the currently-resolved document(s).

    Used for summarize / full-read tasks. Requires a document to be resolved
    first (via resolve_document_reference) so ``ctx.document_ids`` is populated.
    """
    from app.services.agent.tools import get_documents_content

    if not ctx.document_ids:
        return tool_result(
            "Chưa xác định được văn bản. Hãy gọi resolve_document_reference trước "
            "để chọn văn bản, rồi mới lấy nội dung."
        )
    res = await get_documents_content(document_ids=ctx.document_ids, db=_db())
    docs = res.get("documents", []) or []
    parts = []
    for d in docs:
        if d.get("content"):
            parts.append(f"### {d.get('filename', d.get('id'))}\n{d['content']}")
        elif d.get("error"):
            parts.append(f"### {d.get('filename', d.get('id'))}\n(lỗi: {d['error']})")
    summary = "\n\n".join(parts) if parts else "Không lấy được nội dung văn bản."
    return tool_result(summary, data={"total_count": res.get("total_count", 0)})


async def _adapt_read_uploaded_document(args: dict, ctx: ToolContext) -> dict:
    """Return the FULL parsed text of the file(s) the user attached this turn.

    Unlike get_document_content (which needs resolve_document_reference first),
    this reads ``ctx.uploaded_document_ids`` directly — used for summarising,
    proofreading (spell/grammar), or reading the attachment before cross-checking
    it against the knowledge base.
    """
    from app.services.agent.tools import get_documents_content

    doc_ids = ctx.uploaded_document_ids or []
    if not doc_ids:
        return tool_result(
            "Người dùng chưa đính kèm văn bản nào trong lượt này."
        )
    res = await get_documents_content(document_ids=doc_ids, db=_db())
    docs = res.get("documents", []) or []
    parts = []
    for d in docs:
        if d.get("content"):
            parts.append(f"### {d.get('filename', d.get('id'))}\n{d['content']}")
        elif d.get("error"):
            parts.append(f"### {d.get('filename', d.get('id'))}\n(lỗi: {d['error']})")
    summary = "\n\n".join(parts) if parts else "Không đọc được nội dung văn bản đính kèm."
    return tool_result(summary, data={"total_count": res.get("total_count", 0)})


async def _adapt_check_document_format(args: dict, ctx: ToolContext) -> dict:
    """Check the FORMATTING/thể thức of an attached Word (.docx) file.

    Deterministic: downloads the original .docx from MinIO, extracts margins/
    fonts/line-spacing, and compares them against Vietnamese administrative
    standards (NĐ 30/2020). No RAG — returns a structured issue report.
    """
    from app.services.agent.tools import get_document_format
    from app.services.agents.docx_formatter_tools import analyze_format_issues

    doc_ids = ctx.uploaded_document_ids or ctx.document_ids or []
    if not doc_ids:
        return tool_result(
            "Chưa có file đính kèm để kiểm tra thể thức. Hãy yêu cầu người dùng "
            "tải lên một file Word (.docx)."
        )
    res = await get_document_format(document_ids=doc_ids, db=_db())
    out_parts = []
    for d in res.get("documents", []) or []:
        name = d.get("filename") or d.get("id")
        if d.get("error"):
            out_parts.append(f"### {name}\n(không kiểm tra được: {d['error']})")
            continue
        fmt = d.get("format_data") or {}
        issues = analyze_format_issues(fmt)
        if not issues:
            out_parts.append(f"### {name}\nKhông phát hiện lỗi thể thức rõ ràng so với chuẩn.")
            continue
        lines = [f"### {name} — {len(issues)} vấn đề thể thức"]
        for it in issues:
            sev = it.get("severity", "")
            lines.append(
                f"- [{sev}] {it.get('detail', '')} → {it.get('suggestion', '')}"
            )
        out_parts.append("\n".join(lines))
    summary = "\n\n".join(out_parts) if out_parts else "Không có dữ liệu thể thức."
    return tool_result(summary, data={"format_checked": True})


async def _adapt_summarize_long_document(args: dict, ctx: ToolContext) -> dict:
    """Summarise the attached file(s) with MAP-REDUCE so LONG documents are not
    truncated (read_uploaded_document caps at 48k chars).

    Reads the FULL markdown, splits it into windows, summarises each window in
    parallel (map), then combines the partial summaries (reduce). Optional
    ``focus`` narrows the summary to a topic (e.g. 'các điều khoản xử phạt').
    """
    import asyncio
    from sqlalchemy import select
    from app.models.document import Document
    from app.services.storage_service import get_storage_service
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.chunker import chunk_text

    raw_ids = ctx.uploaded_document_ids or ctx.document_ids or []
    doc_ids: list = []
    for d in raw_ids:
        try:
            doc_ids.append(d if isinstance(d, uuid.UUID) else uuid.UUID(str(d)))
        except (ValueError, TypeError):
            continue
    if not doc_ids:
        return tool_result("Chưa có file đính kèm để tóm tắt.")

    focus = (args.get("focus") or "").strip()
    SINGLE_PASS_LIMIT = 40000   # chars — below this, one LLM call suffices (no map-reduce)
    WINDOW = 12000              # chars per map window (~3k tokens)
    MAX_WINDOWS = 20            # safety cap on a single huge document

    storage = get_storage_service()
    llm = get_llm_provider()

    def _text(out) -> str:
        return out if isinstance(out, str) else getattr(out, "content", str(out))

    async def _summarize_block(text: str, label: str) -> str:
        instr = (
            f"Tóm tắt phần văn bản sau, TẬP TRUNG vào: {focus}.\n\n" if focus
            else "Tóm tắt ngắn gọn, đầy đủ ý chính của phần văn bản sau.\n\n"
        )
        out = await llm.acomplete(
            messages=[LLMMessage(role="user", content=f"{instr}[{label}]\n{text}")],
            temperature=0.1, max_tokens=512,
        )
        return _text(out)

    res = await _db().execute(select(Document).where(Document.id.in_(doc_ids)))
    docs = res.scalars().all()
    if not docs:
        return tool_result("Không tìm thấy văn bản đính kèm trong hệ thống.")

    per_doc: list[str] = []
    for doc in docs:
        name = doc.original_filename or str(doc.id)
        if not doc.markdown_s3_key:
            per_doc.append(f"### {name}\n(chưa có nội dung đã phân tích — file có thể đang xử lý)")
            continue
        try:
            md = await storage.download_markdown(doc.markdown_s3_key)
        except Exception as e:
            per_doc.append(f"### {name}\n(lỗi tải nội dung: {e})")
            continue

        # Short enough → single pass
        if len(md) <= SINGLE_PASS_LIMIT:
            per_doc.append(f"### {name}\n{await _summarize_block(md, name)}")
            continue

        # ── MAP: split into windows, summarise each in parallel ──
        windows = [c.content for c in chunk_text(md, source=name, chunk_size=WINDOW, chunk_overlap=400)]
        total = len(windows)
        capped = total > MAX_WINDOWS
        if capped:
            windows = windows[:MAX_WINDOWS]
        partials_raw = await asyncio.gather(
            *[_summarize_block(w, f"{name} phần {i+1}/{len(windows)}") for i, w in enumerate(windows)],
            return_exceptions=True,
        )
        partials = [p for p in partials_raw if isinstance(p, str) and p.strip()]
        if not partials:
            per_doc.append(f"### {name}\n(không tóm tắt được nội dung)")
            continue

        # ── REDUCE: combine partial summaries ──
        joined = "\n\n".join(f"- {p}" for p in partials)
        reduce_instr = (
            f"Dưới đây là tóm tắt TỪNG PHẦN của văn bản '{name}'. Hãy TỔNG HỢP thành một bản "
            f"tóm tắt mạch lạc, có cấu trúc"
            + (f", TẬP TRUNG vào: {focus}" if focus else "")
            + ". Giữ nguyên số liệu/điều khoản quan trọng, KHÔNG bịa thêm.\n\n"
        )
        final = _text(await llm.acomplete(
            messages=[LLMMessage(role="user", content=reduce_instr + joined)],
            temperature=0.1, max_tokens=1500,
        ))
        note = (
            f"\n\n_(Văn bản rất dài: đã xử lý {MAX_WINDOWS}/{total} phần đầu — "
            f"có thể thiếu phần cuối.)_" if capped else ""
        )
        per_doc.append(f"### {name}\n{final}{note}")

    summary = "\n\n".join(per_doc) if per_doc else "Không tóm tắt được văn bản."
    return tool_result(summary, data={"map_reduce": True, "focus": focus or None})


# ──────────────────────────────────────────────────────────────────────────
# Adapters — agentic tools (memory + clarification)
# ──────────────────────────────────────────────────────────────────────────

async def _adapt_recall_memory(args: dict, ctx: ToolContext) -> dict:
    from app.services.graphiti_client import search_user_memory

    if not ctx.user_id:
        return tool_result("Không có ngữ cảnh người dùng để truy hồi.")
    query = (args.get("query") or "").strip()
    context = await search_user_memory(ctx.user_id, query)
    if context:
        ctx.state["user_memory_context"] = context
    return tool_result(context or "Không tìm thấy thông tin cá nhân liên quan.",
                       data={"has_memory": bool(context)})


async def _adapt_save_memory(args: dict, ctx: ToolContext) -> dict:
    from app.services.graphiti_client import save_user_fact

    fact = (args.get("fact") or "").strip()
    if not ctx.user_id:
        return tool_result("Không thể lưu: thiếu ngữ cảnh người dùng.")
    if not fact:
        return tool_result("Không thể lưu: thiếu nội dung 'fact'.")
    ok = await save_user_fact(ctx.user_id, fact)
    return tool_result(
        "Đã ghi nhớ." if ok else "Lưu ghi nhớ thất bại (sẽ thử lại sau).",
        data={"saved": ok},
    )


async def _adapt_ask_user(args: dict, ctx: ToolContext) -> dict:
    from app.services.agents.clarification import ask_user_clarification

    question = (args.get("question") or "").strip()
    options = args.get("options") or []
    if not question:
        return tool_result("Lỗi: thiếu 'question' cho ask_user.")
    await ask_user_clarification(ctx.state, question=question, options=options,
                                 context={"type": "react_clarification"})
    # Signal to the executor that the loop should stop and wait for the user.
    return tool_result(question, data={"clarification": True, "stop": True})


# ──────────────────────────────────────────────────────────────────────────
# Registry + schemas
# ──────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable[[dict, ToolContext], Awaitable[dict]]] = {
    "search_documents": _adapt_search_documents,
    "resolve_document_reference": _adapt_resolve_document_reference,
    "search_document_section": _adapt_search_document_section,
    "query_knowledge_graph": _adapt_query_knowledge_graph,
    "list_documents": _adapt_list_documents,
    "search_abbreviation": _adapt_search_abbreviation,
    "search_documents_number": _adapt_search_documents_number,
    "get_document_content": _adapt_get_document_content,
    "read_uploaded_document": _adapt_read_uploaded_document,
    "summarize_long_document": _adapt_summarize_long_document,
    "check_document_format": _adapt_check_document_format,
    "recall_memory": _adapt_recall_memory,
    "save_memory": _adapt_save_memory,
    "ask_user": _adapt_ask_user,
}


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


RAG_TOOL_SCHEMAS: list[dict] = [
    _fn(
        "search_documents",
        "Tìm nội dung trong kho văn bản pháp luật theo CHỦ ĐỀ/KHÁI NIỆM chung, hoặc tìm "
        "trong văn bản đã được resolve trước đó. Dùng khi câu hỏi KHÔNG nêu tên văn bản cụ "
        "thể (vd 'quy định về bảo vệ dữ liệu cá nhân'). mode='vector' cho trích/tóm tắt, "
        "'kg' cho quan hệ thực thể, 'hybrid' (mặc định) khi không chắc.\n"
        "scope: KHI người dùng có ĐÍNH KÈM file — 'uploaded' chỉ tìm trong file đính kèm; "
        "'knowledge_base' tìm trong toàn kho (BỎ QUA file đính kèm). Để ĐỐI CHIẾU file với "
        "kho hoặc HỎI ĐÁP xuyên cả hai: gọi 2 lần — một lần scope='uploaded' và một lần "
        "scope='knowledge_base'. Bỏ trống nếu không có file đính kèm.",
        {
            "query": {"type": "string", "description": "Câu truy vấn nội dung."},
            "mode": {"type": "string", "enum": ["vector", "kg", "hybrid"], "default": "hybrid"},
            "scope": {"type": "string", "enum": ["uploaded", "knowledge_base"],
                      "description": "Phạm vi tìm khi có file đính kèm (xem mô tả)."},
        },
        ["query"],
    ),
    _fn(
        "resolve_document_reference",
        "Phân giải MỘT văn bản được nêu TÊN/SỐ HIỆU (Luật X, Nghị định Y, Thông tư Z, "
        "'văn bản số 13/2023/NĐ-CP') ra văn bản cụ thể. Gọi tool này TRƯỚC khi tìm nội "
        "dung/điều khoản trong một văn bản có tên. Nếu cần so sánh nhiều văn bản, gọi NHIỀU "
        "lần (mỗi văn bản một lần) — có thể trong cùng một lượt để chạy song song. KHÔNG "
        "dùng cho khái niệm chung ('an ninh mạng', 'bảo vệ dữ liệu').",
        {"reference": {"type": "string", "description": "Tên hoặc số hiệu văn bản."}},
        ["reference"],
    ),
    _fn(
        "search_document_section",
        "Lấy nội dung CHÍNH XÁC của một Điều/Khoản/Chương/Mục trong văn bản đang xét. Gọi "
        "sau resolve_document_reference (khi văn bản có tên) hoặc khi đã có văn bản trong "
        "ngữ cảnh. vd section_reference='Điều 5', 'Khoản 2 Điều 8', 'Chương II'.",
        {"section_reference": {"type": "string", "description": "Tham chiếu phần/điều/khoản/chương."}},
        ["section_reference"],
    ),
    _fn(
        "query_knowledge_graph",
        "Tra cứu QUAN HỆ giữa các thực thể / cơ cấu tổ chức / 'ai chịu trách nhiệm' qua "
        "knowledge graph. vd 'Bộ Công an có những đơn vị nào'.",
        {"entity": {"type": "string", "description": "Thực thể/quan hệ cần tra."}},
        ["entity"],
    ),
    _fn(
        "list_documents",
        "Liệt kê các văn bản hiện có trong workspace. Dùng khi user muốn xem có những tài "
        "liệu nào.",
        {},
    ),
    _fn(
        "search_abbreviation",
        "Tra NGHĨA của một từ viết tắt (vd 'BMNN là gì', 'TTGT viết tắt của gì'). KHÁC với "
        "việc tìm văn bản có tên viết tắt.",
        {"abbreviation": {"type": "string", "description": "Từ viết tắt cần tra nghĩa."}},
        ["abbreviation"],
    ),
    _fn(
        "search_documents_number",
        "Tìm văn bản theo SỐ HIỆU khi user chỉ muốn định vị văn bản (không hỏi nội dung). "
        "vd 'tìm văn bản 53/2022/NĐ-CP'.",
        {"query": {"type": "string", "description": "Số hiệu văn bản."}},
        ["query"],
    ),
    _fn(
        "get_document_content",
        "Lấy TOÀN BỘ nội dung văn bản đã được resolve, phục vụ tóm tắt/đọc đầy đủ. Phải gọi "
        "resolve_document_reference trước.",
        {},
    ),
    _fn(
        "read_uploaded_document",
        "Đọc TOÀN BỘ nội dung văn bản NGƯỜI DÙNG ĐÍNH KÈM trong lượt này (không cần resolve). "
        "Dùng khi cần: TÓM TẮT file đính kèm, KIỂM TRA CHÍNH TẢ/ngữ pháp (đọc rồi tự rà soát), "
        "hoặc đọc nội dung file TRƯỚC khi đối chiếu với kho văn bản. Sau khi đọc, để đối chiếu "
        "đúng/sai hãy gọi thêm search_documents(scope='knowledge_base').",
        {},
    ),
    _fn(
        "summarize_long_document",
        "TÓM TẮT văn bản đính kèm bằng map-reduce — DÙNG cho file DÀI (nhiều trang) để KHÔNG "
        "bị cắt mất nội dung như read_uploaded_document (vốn cắt ở 48k ký tự). Đọc toàn văn, "
        "chia phần, tóm tắt từng phần rồi gộp. Đặt 'focus' nếu chỉ cần tóm tắt một khía cạnh "
        "(vd 'các điều khoản xử phạt'); bỏ trống để tóm tắt toàn diện.",
        {"focus": {"type": "string",
                   "description": "Khía cạnh cần tập trung khi tóm tắt (tùy chọn)."}},
    ),
    _fn(
        "check_document_format",
        "Kiểm tra THỂ THỨC/ĐỊNH DẠNG file Word (.docx) người dùng đính kèm (căn lề, cỡ chữ, "
        "khoảng cách dòng...) so với chuẩn hành chính VN (NĐ 30/2020). Dùng khi user nói "
        "'kiểm tra thể thức', 'kiểm tra định dạng', 'căn lề/cỡ chữ đúng chuẩn chưa'. Trả về "
        "báo cáo lỗi có sẵn — KHÔNG cần search.",
        {},
    ),
    _fn(
        "recall_memory",
        "Truy hồi NGỮ CẢNH CÁ NHÂN của người dùng (thiết bị, đơn vị, vai trò...) khi câu hỏi "
        "nhắc tới 'tôi', 'đơn vị tôi', 'cơ quan tôi' hoặc cần so sánh với hoàn cảnh của họ.",
        {"query": {"type": "string", "description": "Khía cạnh cá nhân cần truy hồi."}},
        ["query"],
    ),
    _fn(
        "save_memory",
        "Ghi nhớ MỘT thông tin cá nhân BỀN của người dùng để dùng cho các lần sau (vd user "
        "nói 'nhớ giúp tôi rằng đơn vị tôi là Công an tỉnh Hà Tĩnh'). Chỉ lưu fact bền, "
        "KHÔNG lưu câu hỏi hay nội dung nhất thời.",
        {"fact": {"type": "string", "description": "Thông tin cần ghi nhớ, viết gọn rõ."}},
        ["fact"],
    ),
    _fn(
        "ask_user",
        "HỎI LẠI người dùng khi truy vấn mơ hồ/thiếu thông tin (vd nhiều văn bản trùng tên, "
        "không rõ văn bản nào). Sau khi gọi, hệ thống dừng và chờ người dùng trả lời.",
        {
            "question": {"type": "string", "description": "Câu hỏi làm rõ (tiếng Việt)."},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": "Các lựa chọn gợi ý (tùy chọn)."},
        },
        ["question"],
    ),
]


async def dispatch_tool(name: str, args: dict, ctx: ToolContext) -> dict:
    """Look up and run a tool adapter, normalising unknown/failed calls.

    Never raises: an unknown tool or an adapter exception is returned as a
    ``tool_result`` so the executor can feed it back to the LLM for recovery.
    """
    adapter = TOOL_REGISTRY.get(name)
    if adapter is None:
        logger.warning(f"[react_tools] unknown tool requested: {name!r}")
        return tool_result(f"Lỗi: công cụ '{name}' không tồn tại.")

    import time as _time

    _t0 = _time.monotonic()
    try:
        res = await adapter(args or {}, ctx)
        _record_react_tool(name, args, res, int((_time.monotonic() - _t0) * 1000), None)
        return res
    except Exception as e:  # noqa: BLE001 — surface to LLM, don't crash the loop
        logger.error(f"[react_tools] tool {name!r} failed: {e}", exc_info=True)
        _record_react_tool(name, args, None, int((_time.monotonic() - _t0) * 1000), str(e))
        return tool_result(f"Lỗi khi chạy '{name}': {e}")


def _record_react_tool(name: str, args: dict | None, res: dict | None,
                       latency_ms: int, error: str | None) -> None:
    """Best-effort capture of a ReAct tool call into the dataset collector."""
    try:
        from app.services.agent.trace_collector import get_collector

        coll = get_collector()
        if coll is None:
            return
        res = res or {}
        coll.add_tool_call(
            name=name,
            args=args or {},
            result_summary=str(res.get("summary") or "")[:4000],
            sources_count=len(res.get("sources") or []),
            images_count=len(res.get("images") or []),
            data=res.get("data") or None,
            latency_ms=latency_ms,
            error=error,
        )
    except Exception:  # pragma: no cover - never break the loop
        pass
