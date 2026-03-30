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
    """Create (and cache) the LLM provider configured via ``LLM_PROVIDER``."""
    from app.core.config import settings

    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        from app.services.llm.gemini import GeminiLLMProvider

        if not settings.GOOGLE_AI_API_KEY:
            raise ValueError("GOOGLE_AI_API_KEY is required when LLM_PROVIDER=gemini")
        return GeminiLLMProvider(
            api_key=settings.GOOGLE_AI_API_KEY,
            model=settings.LLM_MODEL_FAST,
            thinking_level=settings.LLM_THINKING_LEVEL,
        )

    if provider == "ollama":
        from app.services.llm.ollama import OllamaLLMProvider

        return OllamaLLMProvider(
            host=settings.OLLAMA_HOST,
            model=settings.OLLAMA_MODEL,
        )

    if provider == "openai_compatible":
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider

        return OpenAICompatibleLLMProvider(
            base_url=settings.OPENAI_COMPATIBLE_BASE_URL,
            model=settings.OPENAI_COMPATIBLE_MODEL,
            api_key=settings.OPENAI_COMPATIBLE_API_KEY,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: gemini, ollama, openai_compatible")


@lru_cache
def get_memory_agent() -> LLMProvider:
    """Create (and cache) a dedicated LLM provider for internal agent tasks (like memory)."""
    from app.core.config import settings

    if settings.MEMORY_AGENT_LOCAL:
        from app.services.llm.vllm_local import LocalVLLMProvider
        return LocalVLLMProvider(
            model=settings.MEMORY_AGENT_MODEL,
            gpu_memory_utilization=settings.MEMORY_AGENT_GPU_UTILIZATION,
            cuda_device=settings.MEMORY_AGENT_CUDA_DEVICE,
        )

    from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider
    # Memory agent using remote vLLM (OpenAI compatible)
    return OpenAICompatibleLLMProvider(
        base_url=settings.MEMORY_AGENT_BASE_URL,
        model=settings.MEMORY_AGENT_MODEL,
        api_key=settings.MEMORY_AGENT_API_KEY,
    )


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
    "get_kg_llm_provider",
    "get_embedding_provider",
    "LLMProvider",
    "EmbeddingProvider",
]
