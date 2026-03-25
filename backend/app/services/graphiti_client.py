"""
Graphiti Memory Client
======================

Temporal knowledge-graph memory for the LangGraph agent pipeline.
Replaces the flat pgvector UserMemory table with a rich graph that
tracks *how* facts about users change over time.

Architecture
------------
- Storage : Neo4j (already in docker-compose stack on bolt://localhost:7687)
- LLM     : Qwen3-4B (MEMORY_AGENT_BASE_URL, OpenAI-compatible) for entity/fact extraction
- Embedder: NexusRAGEmbedder — wraps the existing BAAI/bge-m3 EmbeddingService singleton
            so we don't load a second embedding model

Data model
----------
Every conversation turn is added as an Episode (EpisodeType.text).
Graphiti autonomously extracts Entities and temporal Edges (facts) from the
episode body.  Facts that contradict earlier ones are *invalidated* (not deleted),
so we can always query what was true at any point in time.

Each user's data is partitioned by group_id = "nexusrag_user_{user_id}".

Public API
----------
    await initialize_graphiti()                        # call once at app startup
    context = await search_user_memory(uid, query)     # → formatted string for system prompt
    await add_conversation_episode(uid, user_msg, ai_msg, session_id)  # background task

Internal helpers
----------------
    get_graphiti_client() → Graphiti   (singleton, lazily created)
    _format_memory_context(edges)      # edges → human-readable facts string
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_graphiti_client: Any | None = None  # graphiti_core.Graphiti


# ---------------------------------------------------------------------------
# Custom Embedder — wraps NexusRAG's BAAI/bge-m3 EmbeddingService
# ---------------------------------------------------------------------------

from graphiti_core.embedder.client import EmbedderClient


class NexusRAGEmbedder(EmbedderClient):
    """
    Implements the graphiti_core EmbedderClient ABC using the existing
    EmbeddingService singleton (BAAI/bge-m3, 1024-dim).

    Must subclass EmbedderClient (not just duck-type it) because Graphiti
    validates the embedder via Pydantic isinstance() check internally.

    Graphiti calls  create(input)  where input may be a single string or a
    list of strings.  We always return a *single* embedding vector (list[float])
    for the first / only string passed.  Batch calls are handled by create_batch.
    """

    def __init__(self):
        self._svc = None  # lazy — EmbeddingService loads the model on first use

    def _get_service(self):
        if self._svc is None:
            from app.services.embedder import EmbeddingService

            self._svc = EmbeddingService()
        return self._svc

    async def create(self, input_data) -> list[float]:
        """Return a single embedding vector for one text input."""
        svc = self._get_service()
        if isinstance(input_data, list):
            text = input_data[0] if input_data else ""
        else:
            text = str(input_data)

        if not text.strip():
            # Return zero vector for empty input rather than raising
            return [0.0] * settings.GRAPHITI_EMBEDDING_DIM

        # embed_text is synchronous — sentence-transformers releases the GIL
        return svc.embed_text(text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""
        svc = self._get_service()
        if not input_data_list:
            return []
        texts = [t if t.strip() else " " for t in input_data_list]
        return svc.embed_texts(texts)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def get_graphiti_client():
    """
    Return the module-level Graphiti singleton, creating it lazily on first call.

    The Graphiti constructor is *synchronous* — it only stores the configuration.
    The actual async setup (build_indices_and_constraints) happens in initialize_graphiti().

    Root-cause of "api_key must be set" error
    -----------------------------------------
    Graphiti internally creates an OpenAIRerankerClient (cross-encoder) even when a
    custom llm_client is passed.  That component reads the OpenAI key directly from
    the environment variable OPENAI_API_KEY, bypassing LLMConfig.  We set it here
    from our own config so every internal Graphiti component that uses the OpenAI
    SDK finds the key without needing a real OpenAI account.
    """
    global _graphiti_client
    if _graphiti_client is not None:
        return _graphiti_client

    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.llm_client.config import LLMConfig
    except ImportError as exc:
        raise RuntimeError(
            "graphiti-core is not installed. "
            "Add 'graphiti-core>=0.3.0' to requirements.txt and reinstall."
        ) from exc

    # Set OPENAI_API_KEY env var so Graphiti's internal OpenAIRerankerClient
    # (and any other OpenAI-SDK consumer inside graphiti-core) can initialise.
    # We use our own configured key (defaults to "sk-nexusrag" for local endpoints).
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.GRAPHITI_LLM_API_KEY

    llm_config = LLMConfig(
        api_key=settings.GRAPHITI_LLM_API_KEY,
        model=settings.GRAPHITI_LLM_MODEL,
        base_url=settings.GRAPHITI_LLM_BASE_URL,
    )
    # max_tokens MUST be < vLLM --max-model-len (15312).
    # OpenAIGenericClient default is 16384 which causes 400 Bad Request on every call.
    llm_client = OpenAIGenericClient(config=llm_config, max_tokens=8192)
    embedder = NexusRAGEmbedder()

    _graphiti_client = Graphiti(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
    )
    logger.info(
        f"[graphiti] Client created — Neo4j: {settings.NEO4J_URI}, "
        f"LLM: {settings.GRAPHITI_LLM_MODEL} @ {settings.GRAPHITI_LLM_BASE_URL}"
    )
    return _graphiti_client


# ---------------------------------------------------------------------------
# Startup initializer
# ---------------------------------------------------------------------------


async def initialize_graphiti() -> None:
    """
    Create Neo4j indices and constraints required by Graphiti.
    Must be called once during app startup (idempotent — safe to call repeatedly).
    Raises RuntimeError if graphiti-core is not installed (handled gracefully by
    the caller in main.py).
    """
    client = get_graphiti_client()
    await client.build_indices_and_constraints()
    logger.info("[graphiti] Indices and constraints verified/created in Neo4j")


# ---------------------------------------------------------------------------
# Memory search
# ---------------------------------------------------------------------------


async def search_user_memory(
    user_id: int,
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search the Graphiti knowledge graph for facts relevant to *query*,
    scoped to the given user (via group_id).

    Returns a formatted multi-line string suitable for injection into the
    LLM system prompt, or an empty string if nothing is found.

    client.search() returns list[EntityEdge] — each edge has a .fact str.
    """
    if not query.strip():
        return ""

    client = get_graphiti_client()
    group_id = f"nexusrag_user_{user_id}"
    logger.info(f"[graphiti] search for user_id={user_id}, query={query[:80]!r}")

    try:
        edges: list = await client.search(
            query=query,
            group_ids=[group_id],
            num_results=top_k,
        )
    except Exception as exc:
        logger.warning(f"[graphiti] search failed for user {user_id}: {exc}")
        return ""

    return _format_memory_context(edges)


# ---------------------------------------------------------------------------
# Episode saving
# ---------------------------------------------------------------------------

_FACT_EXTRACTOR_PROMPT = """\
You are a personal-fact extractor for a memory system.

Your task: given a user message —
1. Extract ONLY factual statements about the user themselves \
(their name, job, location, devices, preferences, personal info, etc.).
2. Discard questions, requests, and anything that is NOT a personal fact.
3. IMPORTANT: Convert ALL first-person pronouns (tôi, tao, mình, I, me, my) \
to "người dùng" (third-person). This is required so the memory system can \
correctly link facts to the user entity.

Respond ONLY with valid JSON — no explanation, no markdown.

Output format:
{"has_facts": true,  "facts": "<third-person factual statements, Vietnamese or original language>"}
{"has_facts": false, "facts": ""}

Examples:
User: "tôi đang công tác ở đâu?"
→ {"has_facts": false, "facts": ""}

User: "Tôi công tác tại Công an tỉnh Hà Tĩnh"
→ {"has_facts": true, "facts": "Người dùng công tác tại Công an tỉnh Hà Tĩnh"}

User: "Tôi đang sử dụng MacBook Pro 14 inch 2021 và iPhone 16, tôi có thể soạn thảo tài liệu bí mật không?"
→ {"has_facts": true, "facts": "Người dùng đang sử dụng MacBook Pro 14 inch 2021 và iPhone 16"}

User: "Tôi không biết nên làm gì hôm nay"
→ {"has_facts": false, "facts": ""}

User: "tên tôi là Nguyễn Văn A, tôi sinh năm 1990, tôi có thể làm gì với hệ thống này?"
→ {"has_facts": true, "facts": "Người dùng có tên là Nguyễn Văn A, người dùng sinh năm 1990"}

User: "tên tôi là Hưng"
→ {"has_facts": true, "facts": "Người dùng có tên là Hưng"}

User: "My name is John and I work at Google"
→ {"has_facts": true, "facts": "The user's name is John and the user works at Google"}
"""


async def _llm_extract_facts(text: str) -> str:
    """
    Use the memory-agent LLM (Qwen3-4B) to extract personal factual statements
    from a potentially mixed user message.

    Returns the extracted facts string, or "" if the message contains no facts
    about the user (pure question, generic request, etc.).

    Falls back to returning the original text if the LLM call fails, so that
    we never silently drop a potentially useful episode.
    """
    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage as _LLMMsg

        classifier = get_memory_agent()
        response_text = ""
        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=text)],
            system_prompt=_FACT_EXTRACTOR_PROMPT,
            temperature=0.0,
            max_tokens=256,
        ):
            if chunk.text:
                response_text += chunk.text

        # Parse JSON response
        import json as _json

        # Strip potential <think>...</think> tags that Qwen3 may emit
        clean = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        ).strip()
        # Extract JSON object
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            logger.warning(
                f"[graphiti] fact-extractor returned no JSON: {response_text[:80]!r}"
            )
            return text  # fallback: store as-is

        data = _json.loads(m.group())
        if not data.get("has_facts", False):
            return ""
        return data.get("facts", "").strip() or ""

    except Exception as e:
        logger.warning(
            f"[graphiti] fact-extractor LLM failed ({e}), storing original text"
        )
        return text  # fallback: store as-is


async def add_conversation_episode(
    user_id: int,
    user_message: str,
    assistant_message: str,
    session_id: str | None = None,
) -> None:
    """
    Persist the USER turn as a Graphiti Episode for personal memory extraction.

    Design decisions
    ----------------
    1. Only the user's own message is stored (no assistant answer).
       The RAG answer is grounded in documents and pollutes the personal graph
       with organization/event facts from those documents.

    2. Questions are skipped entirely.
       "Tôi đang công tác ở đâu?" carries no new fact — it would only create
       a spurious Episodic node and never produce a RELATES_TO edge.

    3. Episode name is fixed to "user_{user_id}_memory" (not session-scoped).
       Graphiti uses the episode name as a label/source tag; using a per-session
       ID caused proliferation of name-variant nodes instead of accumulating
       facts on a single user entity.

    4. The user identifier prefix anchors extraction to a stable entity:
       "Người dùng (ID: 3): tôi công tác ở Công an tỉnh Hà Tĩnh"
       → Entity: Người_dùng_3 → Fact: công tác tại Công an tỉnh Hà Tĩnh
    """
    if not user_message.strip():
        return

    stripped = user_message.strip()

    # Too short to contain a meaningful personal fact
    if len(stripped) < 10:
        return

    # Use LLM to extract personal facts and discard questions/requests.
    # e.g. "Tôi dùng MacBook Pro, tôi có thể làm X không?"
    #      → "Tôi dùng MacBook Pro"
    # Pure questions / generic requests → "" → skip entirely.
    facts_only = await _llm_extract_facts(stripped)
    if not facts_only:
        logger.info(
            f"[graphiti] No personal facts — skipping episode for user {user_id}: {stripped[:80]!r}"
        )
        return

    logger.info(f"[graphiti] Extracted facts for user {user_id}: {facts_only[:100]!r}")

    # Replace the generic "người dùng" / "the user" placeholder produced by the LLM
    # with a stable, unique internal entity name that anchors all facts to a single
    # Entity node in the graph without exposing the numeric user ID to the LLM output.
    #
    # The internal entity name uses a short hash of the user_id so that:
    #   - Different users never share the same entity node (no cross-user fact leakage)
    #   - The ID is invisible in search results shown to the LLM / user
    #   - The entity name is still stable across sessions (same hash every time)
    #
    # Note: Vietnamese Unicode (ư, ờ) prevents simple (?i) regex matching, so
    # we use explicit string replacement for the two casing variants.
    # "Người dùng ID=<N>" is the stable internal entity anchor used for Graphiti extraction.
    # The numeric ID ensures uniqueness across users (group_id partitions the graph but
    # Graphiti can still merge same-named entities).  The ID is NEVER shown to users —
    # it is stripped in _format_memory_context before the facts reach the LLM or UI.
    # Tested: plain "Người dùng" causes wrong cross-device edges; this form works correctly.
    user_entity = f"Người dùng ID={user_id}"

    episode_text = facts_only
    for src in ("Người dùng", "người dùng", "The user", "the user"):
        episode_text = episode_text.replace(src, user_entity)

    # The episode body IS the fact text — no outer prefix needed because the
    # entity name is already embedded in the text itself.
    episode_body = episode_text

    client = get_graphiti_client()
    group_id = f"nexusrag_user_{user_id}"

    # Fixed name per user — not per session — so all turns accumulate on the
    # same user entity in the KG instead of creating isolated per-session nodes.
    episode_name = f"user_{user_id}_memory"

    try:
        from graphiti_core.nodes import EpisodeType

        await client.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source=EpisodeType.text,
            source_description="NexusRAG user message — personal memory",
            group_id=group_id,
            reference_time=datetime.now(tz=timezone.utc),
        )
        logger.info(
            f"[graphiti] Episode saved for user {user_id} ({len(episode_body)} chars)"
        )
    except Exception as exc:
        # Non-fatal — log and continue. Memory loss is preferable to blocking chat.
        logger.warning(f"[graphiti] add_episode failed for user {user_id}: {exc}")


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def _format_memory_context(edges: list) -> str:
    """
    Convert a list of Graphiti EntityEdge objects into a human-readable
    string for the LLM system prompt.

    The internal NXUser_<hash> entity tag is stripped from each fact before
    formatting so the LLM / user never sees it.

    Output format:
        [Memory]
        - <fact 1>
        - <fact 2>
        ...

    Returns an empty string if edges is empty or all facts are blank.
    """
    if not edges:
        return ""

    facts: list[str] = []
    for edge in edges:
        # EntityEdge has a .fact attribute (str) with the extracted fact text
        fact = getattr(edge, "fact", None) or getattr(edge, "name", None)
        if fact and str(fact).strip():
            # Strip the "Người dùng ID=<N>" internal anchor and replace with "Bạn"
            # so facts read naturally: "Bạn công tác tại..." instead of "Người dùng ID=3 công tác tại..."
            # Uses a regex to match any numeric ID variant.
            cleaned = re.sub(r"Người dùng ID=\d+\s*", "Bạn ", str(fact)).strip()
            cleaned = re.sub(r"người dùng ID=\d+\s*", "Bạn ", cleaned).strip()
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
            if cleaned:
                facts.append(cleaned)

    if not facts:
        return ""

    lines = ["[Memory]"] + [f"- {f}" for f in facts]
    return "\n".join(lines)
