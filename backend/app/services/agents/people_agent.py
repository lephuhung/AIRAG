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
from typing import TYPE_CHECKING, Callable

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

async def _tool_mongo_cccd(state: SupervisorState) -> dict:
    """Search MongoDB people by CCCD."""
    from app.services.agent.tools import search_people_by_cccd

    return await search_people_by_cccd(state.get("rewritten_query", ""))


async def _tool_mongo_name(state: SupervisorState) -> dict:
    """Search MongoDB people by name."""
    from app.services.agent.tools import search_people_by_name

    return await search_people_by_name(state.get("rewritten_query", ""), limit=10)


async def _tool_mongo_bhxh(state: SupervisorState) -> dict:
    """Search MongoDB people by BHXH."""
    from app.services.agent.tools import search_people_by_bhxh

    return await search_people_by_bhxh(state.get("rewritten_query", ""))


async def _tool_mongo_phone(state: SupervisorState) -> dict:
    """Search MongoDB people by phone."""
    from app.services.agent.tools import search_people_by_phone

    return await search_people_by_phone(state.get("rewritten_query", ""), limit=10)


async def _tool_mongo_advanced(state: SupervisorState) -> dict:
    """Search MongoDB people using multiple criteria extracted via LLM."""
    from app.services.agent.tools import search_people_advanced
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    import json
    import re

    user_query = state.get("rewritten_query", state.get("original_query", ""))
    
    # Simple extraction prompt
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
    return await search_people_advanced(criteria, limit=10)


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
# People Agent Node
# =============================================================================

async def people_agent_node(state: SupervisorState) -> dict:
    """
    Execute MongoDB people search based on intent.

    Flow:
    1. Look up tool in registry
    2. Call tool function
    3. Map result to SupervisorState
    4. Emit sources/images events for SSE streaming
    5. Return partial state update
    """
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    intent = state.get("intent", "mongo_search_name")
    logger.info(f"[people_agent] intent={intent!r}")

    # Emit status
    await push_event(state, "status", {"step": "searching", "detail": "Đang tìm kiếm..."})

    if intent not in PEOPLE_TOOL_REGISTRY:
        logger.warning(f"[people_agent] No tool for intent {intent!r}")
        return {"next_agent": AgentType.FINISH}

    tool_fn, mapper = PEOPLE_TOOL_REGISTRY[intent]

    try:
        result = await tool_fn(state)
        updates = mapper(result)

        # Emit sources if present
        sources = updates.get("sources", [])
        if sources:
            await push_event(state, "sources", sources)

        # Add iteration count
        updates["iterations"] = state.get("iterations", 0) + 1

        logger.info(
            f"[people_agent] completed: mongo_results={len(updates.get('mongo_results', []))}"
        )

        return updates

    except Exception as e:
        logger.error(f"[people_agent] tool {intent} failed: {e}", exc_info=True)
        return {
            "kg_summaries": [f"Lỗi tìm kiếm: {str(e)}"],
            "iterations": state.get("iterations", 0) + 1,
        }