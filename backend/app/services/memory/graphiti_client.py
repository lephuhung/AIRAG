"""
Graphiti Memory Client
======================

Temporal knowledge-graph memory for the LangGraph agent pipeline.
Replaces the flat pgvector UserMemory table with a rich graph that
tracks *how* facts about users change over time.

Architecture
------------
- Storage : Neo4j (already in docker-compose stack on bolt://localhost:7687)
- LLM     : memory agent (gemma-4-E4B served as `qwen-memory`, MEMORY_AGENT_BASE_URL, OpenAI-compatible) for entity/fact extraction
- Embedder: NexusRAGEmbedder — wraps the existing EmbeddingService singleton
            (model = HRAG_EMBEDDING_MODEL, currently mainguyen9/vietlegal-harrier-0.6b)
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

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Memory formatting budget — max characters injected into system prompt
# ~1000 chars ≈ 250 tokens (conservative, leaves room for main prompt)
_MEMORY_CONTEXT_MAX_CHARS = 1000

# ---------------------------------------------------------------------------
# Identity-query patterns (trigger full-graph fetch instead of vector search)
# ---------------------------------------------------------------------------
# These patterns match user questions about themselves (name, job, age, …).
# When matched, we bypass the embedding search and read ALL facts from Neo4j
# directly, which is far more reliable than cosine similarity for pronoun-heavy,
# short queries like "tôi tên là gì" or "tôi làm việc ở đâu".
_IDENTITY_PATTERNS = re.compile(
    r"(?i)"
    r"(tên\s*(của\s*)?tôi|tôi\s*tên|tên\s*tôi|my\s*name)"
    r"|(tôi\s*(là|l[\xE0a]\s*ai|sinh|tuổi)|tôi\s*\w{1,4}\s*ai)"
    r"|(tôi\s*(đang\s+|đã\s+|sẽ\s+|vừa\s+|mới\s+)?(công\s*tác|làm\s*việc|đang\s*ở|sống|học|dùng|sử\s*dụng|sư\s*dụng|xài))"
    r"|(thông\s*tin.*tôi|tôi.*thông\s*tin)"
    r"|(who\s*am\s*i|what.*my\s*(name|job|role|age))"
    r"|(nhớ.*tôi|biết.*tôi|tôi.*là\s*ai)"
)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_graphiti_client: Any | None = None  # graphiti_core.Graphiti
# Runtime-config snapshot version the current client was built with (mutable
# dict so the rebuild guard in get_graphiti_client() needs no extra global).
_graphiti_config_version: dict = {"v": -1}


# ---------------------------------------------------------------------------
# Custom Embedder — wraps NexusRAG's EmbeddingService (HRAG_EMBEDDING_MODEL)
# ---------------------------------------------------------------------------

from graphiti_core.embedder.client import EmbedderClient


class NexusRAGEmbedder(EmbedderClient):
    """
    Implements the graphiti_core EmbedderClient ABC using the existing
    EmbeddingService singleton (model = HRAG_EMBEDDING_MODEL, 1024-dim;
    must match GRAPHITI_EMBEDDING_DIM).

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
            from app.services.embedding.embedder import EmbeddingService

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

    # Rebuild guard: when the admin changes the Graphiti LLM config via the
    # WebUI (runtime-config version bump), drop the cached client so the next
    # call rebuilds it with the new model/endpoint. Cheap — the constructor
    # only stores config; Neo4j indices are created separately in
    # initialize_graphiti() and are idempotent.
    if _graphiti_client is not None:
        from app.services.runtime_config import snapshot_version as _rt_version

        if _graphiti_config_version.get("v") != _rt_version():
            logger.info(
                f"[graphiti] Runtime LLM config changed "
                f"({_graphiti_config_version.get('v')} → {_rt_version()}) — rebuilding client"
            )
            _graphiti_client = None
        else:
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

    from app.services.runtime_config import get_effective_sync, snapshot_version

    # Effective Graphiti LLM config: DB override (WebUI) over GRAPHITI_LLM_* env.
    # Set UNCONDITIONALLY on every (re)build so a key rotated via the WebUI
    # propagates to Graphiti's internal OpenAI-SDK consumers too.
    _cfg = get_effective_sync("graphiti")
    os.environ["OPENAI_API_KEY"] = _cfg.api_key

    llm_config = LLMConfig(
        api_key=_cfg.api_key,
        model=_cfg.model,
        base_url=_cfg.base_url,
    )
    globals()["_graphiti_config_version"]["v"] = snapshot_version()
    # max_tokens is the OUTPUT budget and MUST leave room for the INPUT within the
    # served model's context window. qwen-memory (vllm-memory:8088) has
    # max_model_len=8192. Setting the constructor default is NOT enough: Graphiti's
    # internal extraction steps pass max_tokens=DEFAULT_MAX_TOKENS (16384)
    # EXPLICITLY on some calls, overriding the constructor and 400'ing
    # ("max_tokens=16384 cannot be greater than max_model_len=8192"). So we clamp
    # every call in a subclass. 2048 output leaves ~6k tokens for the prompt.
    _SAFE_MAX_OUTPUT = 2048

    class _CappedOpenAIGenericClient(OpenAIGenericClient):
        # All Graphiti extraction goes through the public generate_response();
        # clamp there (overriding the private _generate_response is fragile — it
        # receives max_tokens positionally and double-binds).
        async def generate_response(self, *args, max_tokens=None, **kwargs):
            if max_tokens is None or max_tokens > _SAFE_MAX_OUTPUT:
                max_tokens = _SAFE_MAX_OUTPUT
            return await super().generate_response(*args, max_tokens=max_tokens, **kwargs)

    llm_client = _CappedOpenAIGenericClient(config=llm_config, max_tokens=_SAFE_MAX_OUTPUT)
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
        f"LLM: {_cfg.model} @ {_cfg.base_url} (source={_cfg.source})"
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
# Memory search — helpers
# ---------------------------------------------------------------------------


async def _fetch_all_user_facts(user_id: uuid.UUID, limit: int = 20) -> list:
    """
    Fetch ALL RELATES_TO edges for a user directly from Neo4j via Cypher.

    Returns a list of SimpleNamespace objects with a `.fact` attribute so
    that _format_memory_context can consume them identically to Graphiti edges.

    Used as fallback when vector similarity search returns no results, and as
    the primary retrieval method for identity-type queries ("tôi tên là gì").
    """
    from types import SimpleNamespace

    group_id = f"nexusrag_user_{user_id}"
    cypher = (
        "MATCH (e:Episodic) "
        "WHERE e.group_id = $group_id AND e.content IS NOT NULL "
        "RETURN e.content AS fact, e.created_at AS created_at "
        "UNION "
        "MATCH ()-[r:RELATES_TO]->() "
        "WHERE r.group_id = $group_id AND r.fact IS NOT NULL "
        "RETURN r.fact AS fact, r.created_at AS created_at "
        "ORDER BY created_at DESC "
        f"LIMIT {int(limit)}"
    )
    try:
        client = get_graphiti_client()
        records, _, _ = await client.driver.execute_query(
            cypher, group_id=group_id
        )
        results = [SimpleNamespace(fact=r["fact"]) for r in records if r["fact"]]
        logger.info(
            f"[graphiti] Cypher fetch returned {len(results)} facts for user {user_id}"
        )
        return results
    except Exception as exc:
        logger.warning(
            f"[graphiti] Cypher fallback failed for user {user_id}: {exc}"
        )
        return []


# ---------------------------------------------------------------------------
# LLM Identity Classifier
# ---------------------------------------------------------------------------

async def _llm_classify_identity(query: str) -> bool:
    """
    Use the memory-agent LLM (e.g. Gemma-9B) to determine if a query is asking
    about the user's own identity, equipment, organization, or personal facts.
    Returns True if it is an identity query, False otherwise.
    Falls back to regex if the LLM call fails.
    """
    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage as _LLMMsg
        import json as _json

        classifier = get_memory_agent()
        prompt = (
            "You are a query classifier. Your task is to determine if the user's query "
            "is asking about their own personal information, identity, equipment, organization, "
            "or facts related to themselves (e.g., 'tôi tên gì', 'đơn vị tôi có trách nhiệm gì', "
            "'tôi xài máy gì', 'địa chỉ của tôi').\n"
            "If the query is asking about the user's own information, return true.\n"
            "If it's a general question or asking about someone else, return false.\n"
            "Respond ONLY with valid JSON in this exact format: {\"is_identity\": true} or {\"is_identity\": false}.\n"
            "Do not include any other text or markdown formatting.\n\n"
            f"Query: {query}"
        )
        
        response_text = ""
        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=128,
        ):
            if chunk.text:
                response_text += chunk.text
                
        # Parse JSON response
        clean = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            logger.warning(f"[graphiti] identity-classifier returned no JSON: {response_text[:80]!r}, falling back to regex")
            return bool(_IDENTITY_PATTERNS.search(query))
            
        data = _json.loads(m.group())
        return data.get("is_identity", False)
        
    except Exception as e:
        logger.warning(f"[graphiti] identity-classifier LLM failed ({e}), falling back to regex")
        return bool(_IDENTITY_PATTERNS.search(query))

# ---------------------------------------------------------------------------
# Memory search — main entry point
# ---------------------------------------------------------------------------


async def search_user_memory(
    user_id: uuid.UUID,
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search the Graphiti knowledge graph for facts relevant to *query*,
    scoped to the given user (via group_id).

    Retrieval strategy — 3 layers:

    Layer 1 — Identity shortcut
        Queries about the user themselves ("tôi tên là gì", "tôi làm việc
        ở đâu", "who am I") bypass vector similarity entirely and fetch ALL
        user facts from Neo4j via Cypher.  Cosine similarity is unreliable
        for pronoun-heavy short queries with no lexical overlap to stored facts.

    Layer 2 — Vector search (Graphiti)
        Standard embedding-based search.  All returned edges are kept;
        keyword overlap score is used only for *sorting*, never for filtering.
        This ensures Graphiti results are never silently dropped.

    Layer 3 — Cypher fallback
        If Layer 2 returns no edges, fall back to fetching ALL user facts
        directly from Neo4j — guaranteeing we always surface available memory.

    Returns a formatted multi-line string suitable for injection into the
    LLM system prompt, or an empty string if nothing is found.
    """
    if not query.strip():
        return ""

    logger.info(f"[graphiti] search for user_id={user_id}, query={query[:80]!r}")

    # ------------------------------------------------------------------
    # Layer 1: Identity shortcut — fetch ALL facts for self-referential queries
    # ------------------------------------------------------------------
    is_identity = await _llm_classify_identity(query)
    if is_identity:
        logger.info(
            f"[graphiti] identity query detected — fetching all facts for user {user_id}"
        )
        edges = await _fetch_all_user_facts(user_id, limit=top_k * 4)
        if edges:
            return _format_memory_context(edges, query=query, include_all=True)
        logger.info(f"[graphiti] no facts found in Neo4j for user {user_id}")
        # Falls through to vector search (graph may simply be empty for this user)

    # ------------------------------------------------------------------
    # Layer 2: Standard Graphiti vector search
    # ------------------------------------------------------------------
    group_id = f"nexusrag_user_{user_id}"
    client = get_graphiti_client()
    try:
        edges: list = await client.search(
            query=query,
            group_ids=[group_id],
            num_results=top_k,
        )
    except Exception as exc:
        logger.warning(f"[graphiti] vector search failed for user {user_id}: {exc}")
        edges = []

    if edges:
        logger.info(
            f"[graphiti] vector search returned {len(edges)} edges for user {user_id}"
        )
        return _format_memory_context(edges, query=query, include_all=False)

    # ------------------------------------------------------------------
    # Layer 3: Cypher fallback — always surface memory even on vector miss
    # ------------------------------------------------------------------
    logger.info(
        f"[graphiti] vector search empty — falling back to Cypher for user {user_id}"
    )
    fallback_edges = await _fetch_all_user_facts(user_id, limit=top_k * 2)
    return _format_memory_context(fallback_edges, query=query, include_all=True)


# ---------------------------------------------------------------------------
# Episode saving
# ---------------------------------------------------------------------------

# First-person subject pronouns that, inside a personal-memory fact, refer to
# the user.  Used as a *fallback* anchor: when the LLM fact-extractor is
# unavailable and we store the raw user message, replacing these keeps the fact
# linked to the user entity node instead of creating an orphan.  The lookarounds
# are word-boundary guards (\w is Unicode-aware → safe for Vietnamese), so we
# never corrupt a substring inside a larger word.
_FIRST_PERSON_RE = re.compile(
    r"(?<!\w)(tôi|Tôi|mình|Mình|tao|Tao|tớ|Tớ|I|me|My|my)(?!\w)"
)


def _anchor_to_user(text: str, user_id: uuid.UUID) -> str:
    """Rewrite *text* so every reference to the speaker points at one stable
    entity node — ``"Người dùng ID=<uuid>"``.

    Two passes:
      1. Replace the third-person placeholders the extractor LLM is instructed
         to emit (``"Người dùng"`` / ``"the user"``).
      2. Fallback safety net — replace first-person subject pronouns. On the
         normal path the LLM has already converted these, so this is a no-op; on
         the fallback path (LLM down → raw text stored) it is what keeps the fact
         attached to the user instead of producing an unlinked node.

    If no anchor ends up present (e.g. a fact with no pronoun at all), the entity
    is prepended so Graphiti can still attribute the fact to the user.
    """
    user_entity = f"Người dùng ID={user_id}"
    for src in ("Người dùng", "người dùng", "The user", "the user"):
        text = text.replace(src, user_entity)
    text = _FIRST_PERSON_RE.sub(user_entity, text)
    if user_entity not in text:
        text = f"{user_entity}: {text}"
    return text


_FACT_EXTRACTOR_PROMPT = """\
You are a personal-fact extractor for a memory system.

Your task: given a user message —
1. Extract factual statements about the user OR the user's unit/organization —
   their name, job/role, location, devices, preferences, personal info, AND
   facts about their đơn vị/cơ quan (its name, size, số lượng cán bộ/chiến sỹ/
   nhân sự, structure, function). A fact stated as CONTEXT inside a question
   still counts — extract the declarative part and drop the question part.
2. Discard the question/request itself and anything that is NOT such a fact.
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

User: "Đơn vị tôi có 40 cán bộ chiến sỹ, có thể tạo lập mạng LAN để gửi nhận tài liệu mật không?"
→ {"has_facts": true, "facts": "Đơn vị của người dùng có 40 cán bộ chiến sỹ"}

User: "Cơ quan tôi trực thuộc Bộ Công an, vậy có phải xin phép khi mua thiết bị không?"
→ {"has_facts": true, "facts": "Cơ quan của người dùng trực thuộc Bộ Công an"}

User: "My name is John and I work at Google"
→ {"has_facts": true, "facts": "The user's name is John and the user works at Google"}
"""


async def _llm_extract_facts(text: str, *, max_attempts: int = 2) -> str:
    """
    Use the memory-agent LLM (gemma-4-E4B, served as `qwen-memory`) to extract personal factual statements
    from a potentially mixed user message.

    Returns:
      * ``""`` when the LLM confidently reports the message holds no personal
        fact (pure question / generic request) — caller skips the episode.
      * the extracted third-person fact string on success.
      * the ORIGINAL text as a fallback when the LLM call fails or returns
        unparseable output after ``max_attempts`` tries — so we never silently
        drop a potentially useful episode. The caller anchors this raw text via
        :func:`_anchor_to_user` before storing.

    The retry loop here is the *in-handler* layer; the memory worker adds a
    second, durable RabbitMQ-level retry on top (see ``handle_memory``).
    """
    import json as _json

    last_err: Exception | str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            from app.services.llm import get_memory_agent
            from app.services.llm.types import LLMMessage as _LLMMsg

            classifier = get_memory_agent()
            response_text = ""
            async for chunk in classifier.astream(
                [_LLMMsg(role="user", content=text)],
                system_prompt=_FACT_EXTRACTOR_PROMPT,
                temperature=0.0,
                max_tokens=512,  # headroom so reasoning <think> blocks don't truncate the JSON
            ):
                if chunk.text:
                    response_text += chunk.text

            # Strip potential <think>...</think> tags the model may emit
            clean = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            ).strip()
            # Extract JSON object
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if not m:
                last_err = f"no JSON in response: {response_text[:80]!r}"
                logger.warning(
                    f"[graphiti] fact-extractor attempt {attempt}/{max_attempts} "
                    f"returned no JSON: {response_text[:80]!r}"
                )
                continue  # retry

            data = _json.loads(m.group())
            if not data.get("has_facts", False):
                return ""  # definitive: no personal fact → skip
            return data.get("facts", "").strip() or ""

        except Exception as e:
            last_err = e
            logger.warning(
                f"[graphiti] fact-extractor attempt {attempt}/{max_attempts} "
                f"failed: {e}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(0.5 * attempt)

    # All attempts exhausted — fall back to storing the raw text (anchored by
    # the caller) rather than losing the fact entirely.
    logger.warning(
        f"[graphiti] fact-extractor failed after {max_attempts} attempts "
        f"({last_err}) — storing original text as fallback"
    )
    return text


async def extract_personal_facts(text: str) -> list[str]:
    """Extract durable personal facts from a user message, for the "saved to
    memory" notice shown at the end of an answer.

    READ-ONLY: this does NOT write to the graph. Persistence is owned by the
    memory worker (:func:`add_conversation_episode`, enqueued every turn), which
    runs the SAME ``_llm_extract_facts`` gate — so what this returns matches what
    actually gets stored. We surface the notice from here (not from the model's
    ``save_memory`` tool call) because the model calls that tool inconsistently,
    whereas the worker save is deterministic.

    Returns ``[]`` when the message carries no personal fact (questions,
    greetings, too-short input) — i.e. exactly the cases the worker also skips.
    """
    stripped = (text or "").strip()
    # Mirror add_conversation_episode's short-circuit so the notice never claims
    # a save for input the worker would drop.
    if len(stripped) < 5:
        return []
    facts = (await _llm_extract_facts(stripped) or "").strip()
    if not facts:
        return []
    # facts may be a single sentence or several separated by newlines/semicolons.
    items = [
        f.strip(" -•\t")
        for f in re.split(r"[\n;]+", facts)
        if f.strip(" -•\t")
    ]
    return items or [facts]


async def add_conversation_episode(
    user_id: uuid.UUID,
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

    # Too short to contain any personal fact. Kept low (5) so short but real
    # facts like "Tôi là An" (9 chars) are NOT dropped; greetings ("hi", "ok")
    # fall through to the LLM gate which classifies them as has_facts=false.
    if len(stripped) < 5:
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

    # Anchor every reference to the speaker onto one stable entity node
    # ("Người dùng ID=<uuid>"). This (a) prevents cross-user fact leakage, (b) is
    # stripped before facts reach the LLM/UI (see _format_memory_context), and
    # (c) — crucially for the fallback path where facts_only is raw text — also
    # rewrites first-person pronouns so an un-extracted message still links to
    # the user instead of producing an orphan node.
    episode_body = _anchor_to_user(facts_only, user_id)

    client = get_graphiti_client()
    group_id = f"nexusrag_user_{user_id}"

    # Fixed name per user — not per session — so all turns accumulate on the
    # same user entity in the KG instead of creating isolated per-session nodes.
    episode_name = f"user_{user_id}_memory"

    from graphiti_core.nodes import EpisodeType

    # In-handler retry for transient Neo4j / Graphiti blips. On final failure we
    # RAISE so the memory worker's durable RabbitMQ retry (5s/15s/60s) and DLQ
    # take over — instead of the old behaviour of silently dropping the fact.
    max_attempts = 2
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
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
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[graphiti] add_episode attempt {attempt}/{max_attempts} "
                f"failed for user {user_id}: {exc}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(0.5 * attempt)

    raise RuntimeError(
        f"[graphiti] add_episode failed for user {user_id} after "
        f"{max_attempts} attempts: {last_exc}"
    )


async def save_user_fact(user_id: uuid.UUID, fact: str) -> bool:
    """Persist a single, caller-decided fact to the user's personal memory graph.

    Unlike :func:`add_conversation_episode` — which LLM-extracts facts from a raw
    conversation turn — this stores ``fact`` AS-IS. The caller (e.g. an agent's
    ``save_memory`` tool) has already decided this is worth remembering, so we do
    NOT re-filter it. Anchors to the same stable user entity used elsewhere so
    facts accumulate on a single node instead of proliferating.

    Returns True on success, False if the fact is empty or the write fails
    (non-fatal — memory loss is preferable to blocking chat).
    """
    fact = (fact or "").strip()
    if not fact:
        return False

    episode_body = _anchor_to_user(fact, user_id)

    client = get_graphiti_client()
    try:
        from graphiti_core.nodes import EpisodeType

        await client.add_episode(
            name=f"user_{user_id}_memory",
            episode_body=episode_body,
            source=EpisodeType.text,
            source_description="NexusRAG user fact — explicit save_memory",
            group_id=f"nexusrag_user_{user_id}",
            reference_time=datetime.now(tz=timezone.utc),
        )
        logger.info(f"[graphiti] save_user_fact stored for user {user_id}: {fact[:80]!r}")
        return True
    except Exception as exc:
        logger.warning(f"[graphiti] save_user_fact failed for user {user_id}: {exc}")
        return False


# Strong refs to in-flight background saves so the event loop doesn't GC them
# mid-run (asyncio only holds weak refs to tasks).
_bg_save_tasks: set = set()


def save_user_fact_background(user_id: uuid.UUID, fact: str) -> None:
    """Persist a user fact WITHOUT blocking the caller.

    Graphiti ``add_episode`` runs several LLM extraction calls (~30-50s on the
    local memory model) — awaiting it inside the chat turn stalled the answer.
    This schedules the write on the running event loop and returns immediately;
    the task outlives the request (best-effort — a memory write lost on shutdown
    is preferable to a slow answer). Errors are logged, never raised.
    """
    fact = (fact or "").strip()
    if not fact or not user_id:
        return
    import asyncio

    async def _runner():
        try:
            await save_user_fact(user_id, fact)
        except Exception as exc:  # save_user_fact already swallows, belt-and-suspenders
            logger.warning(f"[graphiti] background save_user_fact crashed: {exc}")

    try:
        task = asyncio.create_task(_runner())
    except RuntimeError:
        # No running loop (shouldn't happen in the async request path) — skip.
        logger.warning("[graphiti] no running loop for background save; skipped")
        return
    _bg_save_tasks.add(task)
    task.add_done_callback(_bg_save_tasks.discard)


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def _format_memory_context(
    edges: list,
    query: str = "",
    budget: int = _MEMORY_CONTEXT_MAX_CHARS,
    include_all: bool = False,
) -> str:
    """
    Convert a list of Graphiti EntityEdge objects (or SimpleNamespace with .fact)
    into a human-readable string for the LLM system prompt, truncated to budget.

    Args:
        edges:       List of edge objects with a .fact attribute.
        query:       The original search query — used for relevance scoring/sorting.
        budget:      Maximum character budget for the output string.
        include_all: When True, ALL edges are included regardless of keyword
                     overlap score.  Use this for identity queries and Cypher
                     fallback results where lexical matching is unreliable.
                     When False (default), facts with zero keyword overlap are
                     skipped only once at least one higher-scoring fact has been
                     included — preventing a flood of completely unrelated facts
                     while still surfacing all facts when none match the query.

    The internal entity anchor ("Người dùng ID=<UUID>") is stripped from each
    fact before formatting so the LLM / user never sees it.

    Output format:
        Thông tin cá nhân đã biết về người dùng:
        - <fact 1>
        - <fact 2>
        ...

    Returns an empty string if edges is empty or all facts are blank.
    """
    if not edges:
        return ""

    # Tokenize query into keywords for relevance scoring
    query_keywords = set(re.findall(r"\b\w{2,}\b", query.lower())) if query else set()

    facts: list[tuple[int, str]] = []  # (relevance_score, fact_str)
    for edge in edges:
        fact = getattr(edge, "fact", None) or getattr(edge, "name", None)
        if not fact or not str(fact).strip():
            continue

        cleaned = str(fact).strip()
        # Strip internal anchor (Người dùng ID=<UUID>) → "Bạn"
        cleaned = re.sub(r"Người dùng ID=[^\s]+\s*", "Bạn ", cleaned).strip()
        cleaned = re.sub(r"người dùng ID=[^\s]+\s*", "Bạn ", cleaned).strip()
        if not cleaned:
            continue
        # Capitalize first letter
        cleaned = cleaned[0].upper() + cleaned[1:]

        # Relevance score: count keyword overlaps between fact and query
        fact_keywords = set(re.findall(r"\b\w{2,}\b", cleaned.lower()))
        score = len(query_keywords & fact_keywords) if query_keywords else 0

        facts.append((score, cleaned))

    if not facts:
        return ""

    # Sort by descending relevance score
    facts.sort(key=lambda x: x[0], reverse=True)

    # Build output within budget
    # Header is deliberately NOT a "[...]" bracket: a "[Memory]" header reads to
    # the LLM like a citation token and leaked into answers as raw "[Memory]" /
    # "Source: Memory" text. A plain Vietnamese label carries no citation shape.
    _header = "Thông tin cá nhân đã biết về người dùng:"
    lines: list[str] = [_header]
    total_len = len(_header) + 1  # +1 for trailing newline
    included = 0

    for score, fact in facts:
        # When include_all=False, skip zero-relevance facts only after we
        # already have at least one higher-scoring fact in the output.
        # When include_all=True (identity/Cypher-fallback path), never skip.
        if not include_all and score == 0 and included > 0:
            continue
        line = f"- {fact}"
        line_len = len(line) + 1  # +1 for newline
        if total_len + line_len > budget:
            break
        lines.append(line)
        total_len += line_len
        included += 1

    # Return empty string if we only have the header with no facts
    if included == 0:
        return ""

    return "\n".join(lines)
