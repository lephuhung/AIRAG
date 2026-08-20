"""
People Agent
============

Single-file agent handling MongoDB-based people search operations:
- mongo_search_cccd  — search by CCCD (Căn cước công dân) number
- mongo_search_name  — search by name (partial, case-insensitive)
- mongo_search_bhxh  — search by BHXH (Bảo hiểm xã hội) number
- mongo_search_phone — search by phone number
- mongo_search_advanced -- extract multi-criteria (Name, DOB, Address...)

Uses existing tool functions from app.services.agent.tools.
"""

from __future__ import annotations

import logging
import re as _re
from typing import TYPE_CHECKING, Callable

from app.services.agent.langfuse_tracing import _get_langfuse_client

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)

# =============================================================================
# Result Mappers
# =============================================================================

def _map_mongo_result(result: dict) -> dict:
    """Map MongoDB people search result to SupervisorState."""
    display = result.get("display", "")
    persons = result.get("persons", [])
    return {
        "mongo_results": persons,
        "kg_summaries": [display],
        "final_answer": display,  # Needed by answer_generator mongo branch
    }


# =============================================================================
# Tool Functions
# =============================================================================

async def _tool_mongo_cccd(state: SupervisorState):
    """Search MongoDB people by CCCD."""
    from app.services.agent.tools import search_people_by_cccd

    async for res in search_people_by_cccd(state.get("rewritten_query", "")):
        yield res


async def _tool_mongo_name(state: SupervisorState):
    """Search MongoDB people by name."""
    from app.services.agent.tools import search_people_by_name

    async for res in search_people_by_name(state.get("rewritten_query", ""), limit=10):
        yield res


async def _tool_mongo_bhxh(state: SupervisorState):
    """Search MongoDB people by BHXH."""
    from app.services.agent.tools import search_people_by_bhxh

    async for res in search_people_by_bhxh(state.get("rewritten_query", "")):
        yield res


async def _tool_mongo_phone(state: SupervisorState):
    """Search MongoDB people by phone."""
    from app.services.agent.tools import search_people_by_phone

    async for res in search_people_by_phone(state.get("rewritten_query", ""), limit=10):
        yield res


async def _tool_mongo_advanced(state: SupervisorState):
    """Search MongoDB people using multiple criteria extracted via PII or LLM."""
    from app.services.agent.tools import search_people_advanced
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    import json
    import re

    user_query = state.get("rewritten_query", state.get("original_query", ""))

    pii_criteria = state.get("_pii_criteria")
    if pii_criteria:
        criteria = {
            "name": pii_criteria.get("name", ""),
            "dob": pii_criteria.get("dob", ""),
            "address": pii_criteria.get("address", ""),
            "phone": pii_criteria.get("phone", ""),
            "cccd": pii_criteria.get("cccd", ""),
            "bhxh": pii_criteria.get("bhxh", ""),
        }
        logger.info(f"[_tool_mongo_advanced] Using pre-extracted PII criteria: {criteria}")
        async for res in search_people_advanced(criteria, limit=10):
            yield res
        return

    # ── Fallback: LLM extraction (legacy path when PII service is off / failed) ──
    sys_prompt = (
        "Bạn là một chuyên gia trích xuất dữ liệu. Hãy phân tích yêu cầu của người dùng "
        "và trích xuất các thông tin tìm kiếm sau thành mã JSON thật chuẩn xác (chỉ trả về JSON, không giải thích).\n"
        "Các trường có thể có: 'name' (Họ tên), 'dob' (Năm sinh hoặc ngày sinh), 'address' (Quê quán, địa chỉ), 'phone' (Số điện thoại).\n"
        "Nếu không có thông tin tương ứng cho một trường, hãy để chuỗi rỗng: \"\".\n\n"
        "Ví dụ:\n"
        'Người dùng: "Tìm người có tên Nguyễn Văn A, sinh năm 1995, quê ở Hà Nội"\n'
        'Output JSON: {"name": "Nguyễn Văn A", "dob": "1995", "address": "Hà Nội", "phone": ""}'
    )

    llm = get_llm_provider()
    extraction_res = await llm.acomplete(
        messages=[
            LLMMessage(role="system", content=sys_prompt),
            LLMMessage(role="user", content=user_query)
        ],
        temperature=0.0
    )

    content = extraction_res if isinstance(extraction_res, str) else getattr(extraction_res, "content", "{}")

    # Parsing JSON robustly
    criteria = {}
    try:
        # Strip markdown like ```json
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1].strip()
        criteria = json.loads(content)
    except Exception as e:
        logger.error(f"[_tool_mongo_advanced] Failed to parse JSON from LLM: {content}. Error: {e}")
        # fallback parsing using regex
        criteria = {"name": user_query, "dob": "", "address": "", "phone": ""}

    logger.info(f"[_tool_mongo_advanced] Extracted criteria: {criteria}")
    async for res in search_people_advanced(criteria, limit=10):
        yield res


# =============================================================================
# Tool Registry
# =============================================================================

PEOPLE_TOOL_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "mongo_search_cccd": (_tool_mongo_cccd, _map_mongo_result),
    "mongo_search_name": (_tool_mongo_name, _map_mongo_result),
    "mongo_search_bhxh": (_tool_mongo_bhxh, _map_mongo_result),
    "mongo_search_phone": (_tool_mongo_phone, _map_mongo_result),
    "mongo_search_advanced": (_tool_mongo_advanced, _map_mongo_result),
}


# =============================================================================
# PII-based Smart Routing
# =============================================================================

async def _extract_pii_and_route(state: SupervisorState) -> tuple[str, str, dict]:
    """Use PII extraction service to determine the best intent and query.

    Returns (intent, query, criteria_dict).
    When PII service is unavailable or returns nothing, falls back to the
    existing intent + rewritten_query from the supervisor.
    """
    from app.services.pii import extract_pii

    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query") or original_query
    fallback_intent = state.get("intent", "mongo_search_name")

    # Only invoke PII extraction when the query looks like it might contain
    # person identifiers or names.
    pii_text = rewritten_query or original_query
    if not pii_text or len(pii_text.strip()) < 3:
        return fallback_intent, rewritten_query, {}

    entities = await extract_pii(pii_text)
    if not entities:
        logger.info("[people_agent_node] PII extraction skipped/unavailable, using fallback intent=%s", fallback_intent)
        return fallback_intent, rewritten_query, {}

    phone_numbers = entities.get("phone_number", [])
    id_numbers = entities.get("id_number", [])
    human_names = entities.get("human_name", [])

    # Clean id_numbers: strip non-digits for length heuristics
    clean_ids = [_re.sub(r"\D", "", id_val) for id_val in id_numbers]

    # ── Single-criterion routing ──────────────────────────────────────────
    if phone_numbers and not id_numbers and not human_names:
        intent = "mongo_search_phone"
        # Search ALL extracted numbers (not just the first). search_by_phone()
        # accepts a comma list and extracts every 10-digit group ($in match),
        # so "a, b, c" finds people for every number the PII model returned.
        query = ", ".join(phone_numbers)
        logger.info("[people_agent_node] PII route → phone: %s", query)
        return intent, query, {}

    if clean_ids and not phone_numbers and not human_names:
        id_val = clean_ids[0]
        if len(id_val) == 12:
            intent = "mongo_search_cccd"
            query = id_val
            logger.info("[people_agent_node] PII route → cccd: %s", query)
            return intent, query, {}
        if len(id_val) == 10:
            intent = "mongo_search_bhxh"
            query = id_val
            logger.info("[people_agent_node] PII route → bhxh: %s", query)
            return intent, query, {}
        # Unknown length id — treat as generic id search via advanced
        intent = "mongo_search_advanced"
        criteria = {"name": "", "dob": "", "address": "", "phone": "", "cccd": id_val}
        logger.info("[people_agent_node] PII route → advanced (unknown id %d digits): %s", len(id_val), id_val)
        return intent, pii_text, criteria

    if human_names and not phone_numbers and not clean_ids:
        intent = "mongo_search_name"
        query = human_names[0]
        logger.info("[people_agent_node] PII route → name: %s", query)
        return intent, query, {}

    # ── Multi-criteria routing (name + phone / id / address) ─────────────
    if phone_numbers or clean_ids or human_names:
        intent = "mongo_search_advanced"
        cccd = ""
        bhxh = ""
        for id_val in clean_ids:
            if len(id_val) == 12:
                cccd = id_val
            elif len(id_val) == 10:
                bhxh = id_val
            else:
                # If only one id and length is weird, put it in cccd field
                # (the advanced tool searches across fields with OR logic)
                cccd = id_val if not cccd else cccd

        criteria = {
            "name": human_names[0] if human_names else "",
            "dob": "",
            "address": "",
            "phone": phone_numbers[0] if phone_numbers else "",
            "cccd": cccd,
            "bhxh": bhxh,
        }
        logger.info(
            "[people_agent_node] PII route → advanced (name=%r phone=%r cccd=%r bhxh=%r)",
            criteria["name"], criteria["phone"], criteria["cccd"], criteria["bhxh"],
        )
        return intent, pii_text, criteria

    # Nothing useful extracted → fallback
    return fallback_intent, rewritten_query, {}


# =============================================================================
# People Agent Node
# =============================================================================

async def people_agent_node(state: SupervisorState) -> dict:
    """
    Execute MongoDB people search based on intent.

    Flow:
    1. (NEW) Call PII extraction service to refine intent + query
    2. Look up tool in registry
    3. Call tool function
    4. Map result to SupervisorState
    5. Emit sources/images events for SSE streaming
    6. Return partial state update
    """
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    langfuse = _get_langfuse_client()

    # Phase 0: PII extraction for smarter routing
    intent, query, pii_criteria = await _extract_pii_and_route(state)
    # Inject the refined query back into state so downstream nodes see it
    state["rewritten_query"] = query
    if pii_criteria:
        state["_pii_criteria"] = pii_criteria

    logger.info(f"[LANGGRAPH_NODE] Entering people_agent_node, intent={intent!r}, query={query!r}")

    if not state.get("user_can_use_people", False):
        logger.info(
            f"[people_agent_node] Skipped for user_id={state.get('user_id')!r} (no permission)"
        )
        await push_event(state, "status", {"step": "searching", "detail": "Không tìm thấy dữ liệu."})
        return {
            "mongo_results": [],
            "kg_summaries": ["Không tìm thấy dữ liệu."],
            "final_answer": "Không tìm thấy dữ liệu.",
            "next_agent": AgentType.FINISH,
        }

    await push_event(state, "status", {"step": "searching", "detail": "Đang tìm kiếm..."})

    if intent not in PEOPLE_TOOL_REGISTRY:
        logger.warning(f"[_agent] No tool for intent {intent!r}")
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="people_agent",
                    input={"intent": intent, "query": query},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "no_tool"})
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] people_agent span failed: {e}")
        return {"next_agent": AgentType.FINISH}

    tool_fn, mapper = PEOPLE_TOOL_REGISTRY[intent]

    try:
        all_persons = []
        all_summaries = []

        async for partial_result in tool_fn(state):
            # MongoDB không kết nối được → thông báo "hệ thống đang bận" và dừng,
            # không để result_evaluator retry vào hệ thống đang lỗi.
            if partial_result.get("error") == "unavailable":
                busy = partial_result.get("display") or "Hệ thống tra cứu đang bận, vui lòng thử lại sau."
                logger.warning("[people_agent_node] MongoDB unavailable — returning busy message")
                await push_event(state, "status", {"step": "error", "detail": busy})
                return {
                    "mongo_results": [],
                    "kg_summaries": [busy],
                    "final_answer": busy,
                    "next_agent": AgentType.FINISH,
                }

            updates = mapper(partial_result)

            sources = updates.get("sources", [])
            if sources:
                await push_event(state, "sources", sources)

            new_persons = updates.get("mongo_results", [])
            if new_persons:
                all_persons.extend(new_persons)
                await push_event(state, "people_data", all_persons)

            new_summaries = updates.get("kg_summaries", [])
            if new_summaries:
                all_summaries.extend(new_summaries)

        logger.info(
            f"[LANGGRAPH_DECISION] people_agent_node completed: mongo_results={len(all_persons)}"
        )

        final_display = "\n".join(all_summaries) if all_summaries else "Không tìm thấy dữ liệu."

        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="people_agent",
                    input={"intent": intent, "query": query},
                    level="DEFAULT",
                )
                obs.update(
                    output={
                        "outcome": "found" if all_persons else "no_results",
                        "persons_count": len(all_persons),
                    }
                )
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] people_agent span failed: {e}")

        return {
            "mongo_results": all_persons,
            "kg_summaries": all_summaries,
            "final_answer": final_display,
        }

    except Exception as e:
        logger.error(f"[_agent] tool {intent} failed: {e}", exc_info=True)
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="people_agent",
                    input={"intent": intent, "query": query},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "error", "error": str(e)})
                obs.end()
            except Exception:
                pass
        return {
            "kg_summaries": [f"Lỗi tìm kiếm: {str(e)}"],
        }


# =============================================================================
# People Document Search Node
# =============================================================================

# Common name keys across the various people collections (BHXH/LG/EVN/…). Used to
# anchor document relevance on the matched person's actual name.
_PERSON_NAME_KEYS = (
    "hoTen", "ho_ten", "ho_va_ten", "hovaten", "HoTen", "HoVaTen",
    "name", "full_name", "fullName", "Name",
)


def _digit_anchors(text: str) -> list[str]:
    """Identifier strings (CCCD/BHXH/phone) = runs of >=6 digits, separators stripped."""
    return _re.findall(r"\d{6,}", _re.sub(r"[\s.\-]", "", text or ""))


def _person_name_anchors(persons: list) -> list[str]:
    """Lowercased person names from the MongoDB records, for relevance matching."""
    names: list[str] = []
    for p in persons or []:
        if not isinstance(p, dict):
            continue
        for k in _PERSON_NAME_KEYS:
            v = p.get(k)
            if isinstance(v, str) and len(v.strip()) >= 4:
                names.append(v.strip().lower())
                break
    return names


def _filter_relevant_sources(anchor_text: str, persons: list, sources: list) -> list:
    """Keep only document chunks that actually mention the searched identifier or person.

    People companion search is an identifier/name lookup, yet hybrid search always
    returns top-k chunks. A chunk is relevant only if its text contains the searched
    identifier (CCCD/BHXH/phone digits) or the matched person's name; otherwise the
    "related documents" block is just filler. When no anchor can be derived (e.g. a
    vague name query with no MongoDB hit) we fail OPEN and keep the sources, so we
    only ever drop documents we are confident are unrelated.
    """
    digit_anchors = _digit_anchors(anchor_text)
    name_anchors = _person_name_anchors(persons)
    if not digit_anchors and not name_anchors:
        return sources
    kept = []
    for s in sources:
        content = getattr(s, "content", None)
        if content is None and isinstance(s, dict):
            content = s.get("content", "")
        content = content or ""
        norm_digits = _re.sub(r"[\s.\-]", "", content)
        low = content.lower()
        if any(a in norm_digits for a in digit_anchors) or any(n in low for n in name_anchors):
            kept.append(s)
    return kept


async def people_doc_search_node(state: SupervisorState) -> dict:
    """Companion document (RAG) search for the people path.

    Runs AFTER people_agent_node so that EVERY person lookup (CCCD / phone /
    name / BHXH / advanced) also surfaces documents that mention the person or
    identifier — the MongoDB record alone is not enough, the same value (a CCCD
    number, a phone number, a name) often appears in indexed documents too.

    Reuses the RAG agent's hybrid search tool (vector + KG + BM25) with the
    user's query as-is; BM25 reliably catches literal identifiers like a CCCD
    number. Results land in `sources` (consumed by mongo_formatter_node for the
    "related documents" block) and `people_doc_kg` (kept separate from the
    MongoDB display in kg_summaries).

    Best-effort: any failure here must NOT break the person record answer, so
    it returns empty results instead of raising.
    """
    from app.services.agent.streaming import push_event

    query = state.get("rewritten_query") or state.get("original_query", "")
    logger.info(f"[LANGGRAPH_NODE] Entering people_doc_search_node, query={query[:100]!r}")

    await push_event(state, "status", {"step": "searching", "detail": "Đang tra cứu tài liệu liên quan..."})

    try:
        from app.services.agents.rag_agent import _tool_search, _map_search_result

        result = await _tool_search(state)
        updates = _map_search_result(result)

        raw_sources = updates.get("sources", []) or []
        raw_images = updates.get("images", []) or []

        # Relevance gate: hybrid search always returns top-k chunks, but for an
        # identifier/name lookup most are noise. Drop chunks that don't actually
        # mention the CCCD/BHXH/phone or the matched person — otherwise Block 2
        # becomes filler text. When nothing relevant remains, sources=[] and the
        # formatter omits the "related documents" block entirely (only people data).
        anchor_text = f"{state.get('original_query', '')} {state.get('rewritten_query', '')}"
        sources = _filter_relevant_sources(anchor_text, state.get("mongo_results", []), raw_sources)
        dropped = len(raw_sources) - len(sources)

        # Only keep images that belong to a document we actually kept.
        if sources:
            kept_doc_ids = {getattr(s, "document_id", None) for s in sources}
            images = [im for im in raw_images if getattr(im, "document_id", None) in kept_doc_ids]
        else:
            images = []

        if sources:
            await push_event(state, "sources", sources)
        if images:
            await push_event(state, "images", images)

        # Degraded search (some workspace failed, e.g. CUDA OOM) must not be
        # rendered as a confident "no related documents" downstream.
        from app.api.chat_agent import last_search_failures

        degraded = bool(last_search_failures.get())

        logger.info(
            f"[LANGGRAPH_DECISION] people_doc_search_node completed: "
            f"sources={len(sources)} (dropped {dropped} irrelevant), "
            f"images={len(images)}, degraded={degraded}"
        )

        return {
            "sources": sources,
            "images": images,
            # Keep doc KG OUT of kg_summaries to avoid mixing with the mongo block.
            # When no relevant document remains, drop the KG too so Block 2 is fully
            # omitted (it is unused once sources is empty anyway).
            "people_doc_kg": (updates.get("kg_summaries", []) if sources else []),
            "people_doc_search_degraded": degraded,
        }

    except Exception as e:
        logger.error(f"[people_doc_search_node] document search failed: {e}", exc_info=True)
        # Best-effort: never break the person record answer
        return {"people_doc_kg": []}