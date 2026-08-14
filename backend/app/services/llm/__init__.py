"""
LLM Provider Package
=====================
Factory functions to create LLM and embedding providers based on config.

Usage::

    from app.services.llm import get_llm_provider, get_embedding_provider

    llm = get_llm_provider()          # uses LLM_PROVIDER from .env
    emb = get_embedding_provider()    # uses KG_EMBEDDING_PROVIDER from .env
"""
from __future__ import annotations

from functools import lru_cache

from app.services.llm.base import EmbeddingProvider, LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Create (and cache) the LLM provider configured via ``LLM_PROVIDER``.

    Wrapped with Langfuse generation tracing (label ``main_llm``) so every
    answer-generator / direct-answer / evaluator LLM call is captured.
    """
    from app.core.config import settings
    from app.services.agent.langfuse_tracing import trace_llm

    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        from app.services.llm.gemini import GeminiLLMProvider

        if not settings.GOOGLE_AI_API_KEY:
            raise ValueError("GOOGLE_AI_API_KEY is required when LLM_PROVIDER=gemini")
        inner = GeminiLLMProvider(
            api_key=settings.GOOGLE_AI_API_KEY,
            model=settings.LLM_MODEL_FAST,
            thinking_level=settings.LLM_THINKING_LEVEL,
        )
    elif provider == "ollama":
        from app.services.llm.ollama import OllamaLLMProvider

        inner = OllamaLLMProvider(
            host=settings.OLLAMA_HOST,
            model=settings.OLLAMA_MODEL,
            enable_thinking=settings.OLLAMA_ENABLE_THINKING,
        )
    elif provider == "openai_compatible":
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider

        inner = OpenAICompatibleLLMProvider(
            base_url=settings.OPENAI_COMPATIBLE_BASE_URL,
            model=settings.OPENAI_COMPATIBLE_MODEL,
            api_key=settings.OPENAI_COMPATIBLE_API_KEY,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: gemini, ollama, openai_compatible")

    return trace_llm(inner, label="main_llm")


@lru_cache
def get_memory_agent() -> LLMProvider:
    """Create (and cache) a dedicated LLM provider for internal agent tasks (like memory).

    Wrapped with Langfuse generation tracing (label ``memory_agent``) so intent
    classification, query analysis, enrichment and abbreviation calls are captured.
    """
    from app.core.config import settings
    from app.services.agent.langfuse_tracing import trace_llm

    if settings.MEMORY_AGENT_LOCAL:
        from app.services.llm.vllm_local import LocalVLLMProvider
        inner = LocalVLLMProvider(
            model=settings.MEMORY_AGENT_MODEL,
            gpu_memory_utilization=settings.MEMORY_AGENT_GPU_UTILIZATION,
            cuda_device=settings.MEMORY_AGENT_CUDA_DEVICE,
        )
    else:
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider
        # Memory agent using remote vLLM (OpenAI compatible)
        inner = OpenAICompatibleLLMProvider(
            base_url=settings.MEMORY_AGENT_BASE_URL,
            model=settings.MEMORY_AGENT_MODEL,
            api_key=settings.MEMORY_AGENT_API_KEY,
        )

    return trace_llm(inner, label="memory_agent")


def _resolve_thinking_endpoint() -> tuple[str, str, str]:
    """Resolve 9router gateway URL + stable model alias + auth token.

    Preference per field:
      NEXUSRAG_LG_THINKING_* → ANTHROPIC_* (9router env style) → built-in default

    The ``model`` value is a FIXED alias (``nexusrag-thinking``). Remap it to a
    real upstream on 9router — the app does not restart for A/B tests.
    """
    from app.core.config import settings

    base_url = (
        (settings.NEXUSRAG_LG_THINKING_BASE_URL or "").strip()
        or (settings.ANTHROPIC_BASE_URL or "").strip()
        or "http://host.docker.internal:20128/v1"
    )
    model = (
        (settings.NEXUSRAG_LG_THINKING_MODEL or "").strip()
        or (settings.ANTHROPIC_DEFAULT_FABLE_MODEL or "").strip()
        or "nexusrag-thinking"
    )
    api_key = (
        (settings.NEXUSRAG_LG_THINKING_API_KEY or "").strip()
        or (settings.ANTHROPIC_AUTH_TOKEN or "").strip()
        or "sk-nexusrag"
    )
    return base_url, model, api_key


@lru_cache
def _thinking_provider_cached(base_url: str, model: str, api_key: str) -> LLMProvider:
    from app.services.agent.langfuse_tracing import trace_llm
    from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider

    # alias_model=True: request + Langfuse always use the stable alias even if
    # the proxy echoes a different upstream model id in the response body.
    inner = OpenAICompatibleLLMProvider(
        base_url=base_url,
        model=model,
        api_key=api_key,
        alias_model=True,
    )
    return trace_llm(inner, label="thinking_llm")


def get_thinking_provider() -> LLMProvider:
    """LLM for LangGraph routing (+ optional ReAct judge) via a proxy gateway.

    Always calls ``NEXUSRAG_LG_THINKING_BASE_URL`` with the fixed alias
    ``NEXUSRAG_LG_THINKING_MODEL`` (default ``nexusrag-thinking``). Remap that
    alias on the proxy to change the real model — no backend restart.
    """
    base_url, model, api_key = _resolve_thinking_endpoint()
    return _thinking_provider_cached(base_url, model, api_key)


def clear_thinking_provider_cache() -> None:
    """Drop cached thinking providers (tests only; normal A/B swaps happen on the proxy)."""
    _thinking_provider_cached.cache_clear()


@lru_cache
def get_kg_llm_provider() -> LLMProvider:
    """
    Create (and cache) a dedicated LLM provider for LegalKG extraction tasks.
    Uses LEGAL_KG_LLM_PROVIDER + LEGAL_KG_LLM_BASE_URL + LEGAL_KG_LLM_MODEL + LEGAL_KG_LLM_API_KEY.
    """
    from app.core.config import settings

    provider = settings.LEGAL_KG_LLM_PROVIDER.lower()

    if provider == "gemini":
        from app.services.llm.gemini import GeminiLLMProvider

        if not settings.GOOGLE_AI_API_KEY:
            raise ValueError("GOOGLE_AI_API_KEY is required when LEGAL_KG_LLM_PROVIDER=gemini")
        return GeminiLLMProvider(
            api_key=settings.GOOGLE_AI_API_KEY,
            model=settings.LEGAL_KG_LLM_MODEL,
            thinking_level=settings.LLM_THINKING_LEVEL,
        )

    if provider == "ollama":
        from app.services.llm.ollama import OllamaLLMProvider

        return OllamaLLMProvider(
            host=settings.LEGAL_KG_LLM_BASE_URL,
            model=settings.LEGAL_KG_LLM_MODEL,
        )

    if provider == "openai_compatible":
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider

        return OpenAICompatibleLLMProvider(
            base_url=settings.LEGAL_KG_LLM_BASE_URL,
            model=settings.LEGAL_KG_LLM_MODEL,
            api_key=settings.LEGAL_KG_LLM_API_KEY,
        )

    raise ValueError(f"Unknown LEGAL_KG_LLM_PROVIDER: {provider!r}. Supported: gemini, ollama, openai_compatible")


def get_embedding_provider() -> EmbeddingProvider:
    """Create (and cache) the embedding provider for KG (LightRAG)."""
    from app.core.config import settings

    provider = settings.KG_EMBEDDING_PROVIDER.lower()

    if provider == "gemini":
        from app.services.llm.gemini import GeminiEmbeddingProvider

        if not settings.GOOGLE_AI_API_KEY:
            raise ValueError("GOOGLE_AI_API_KEY is required when KG_EMBEDDING_PROVIDER=gemini")
        return GeminiEmbeddingProvider(
            api_key=settings.GOOGLE_AI_API_KEY,
            model=settings.KG_EMBEDDING_MODEL,
        )

    if provider == "ollama":
        from app.services.llm.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            host=settings.OLLAMA_HOST,
            model=settings.KG_EMBEDDING_MODEL,
        )

    if provider == "local":
        from app.services.llm.ollama import LocalEmbeddingProvider

        return LocalEmbeddingProvider(model_name=settings.KG_EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown KG_EMBEDDING_PROVIDER: {provider!r}. Supported: gemini, ollama, local"
    )


__all__ = [
    "get_llm_provider",
    "get_memory_agent",
    "get_thinking_provider",
    "clear_thinking_provider_cache",
    "get_kg_llm_provider",
    "get_embedding_provider",
    "LLMProvider",
    "EmbeddingProvider",
]
