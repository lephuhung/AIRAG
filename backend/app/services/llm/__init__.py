"""
LLM Provider Package
=====================
Factory functions to create LLM providers based on the effective runtime config
(DB override via app/services/runtime_config.py, falling back to `.env`).

All factories keep their SYNCHRONOUS signatures — callers are sync (LangGraph
nodes, supervisor) or run providers in threadpools (caption_worker). Provider
instances are cached per role and rebuilt automatically when
runtime_config.snapshot_version() changes (admin saves/clears an override →
version bump → next factory call rebuilds). No caller needs to await anything.

Usage::

    from app.services.llm import get_llm_provider, get_embedding_provider

    llm = get_llm_provider()          # role "main" — .env default or DB override
    vis = get_llm_provider(role="vision")  # inherits "main" unless overridden
"""
from __future__ import annotations

from app.services.llm.base import EmbeddingProvider, LLMProvider

# role -> (snapshot_version, provider). Rebuilt lazily whenever the runtime
# config version changes; never blocks a sync caller on DB access.
_PROVIDERS: dict[str, tuple[int, LLMProvider]] = {}


def build_provider(cfg) -> LLMProvider:
    """Pure factory: EffectiveLLMConfig -> LLMProvider (untraced).

    Supports gemini | ollama | openai_compatible. Shared by the public factories
    and the admin test-connection endpoint (which probes unsaved configs).
    """
    from app.core.config import settings

    provider = (cfg.provider or "").lower()

    if provider == "gemini":
        from app.services.llm.gemini import GeminiLLMProvider

        if not cfg.api_key:
            raise ValueError("GOOGLE_AI_API_KEY is required when provider=gemini")
        return GeminiLLMProvider(
            api_key=cfg.api_key,
            model=cfg.model,
            thinking_level=cfg.extra.get("thinking_level", settings.LLM_THINKING_LEVEL),
        )

    if provider == "ollama":
        from app.services.llm.ollama import OllamaLLMProvider

        return OllamaLLMProvider(
            host=cfg.base_url,
            model=cfg.model,
            enable_thinking=bool(cfg.extra.get("enable_thinking", False)),
        )

    if provider == "openai_compatible":
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider

        # is_vllm gates the vLLM-only chat_template_kwargs extra_body — strict
        # third-party APIs (OpenAI, DeepSeek…) reject it as an unknown param.
        return OpenAICompatibleLLMProvider(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key or "none",
            is_vllm=bool(cfg.extra.get("is_vllm", False)),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported: gemini, ollama, openai_compatible"
    )


def _resolve_role_config(role: str):
    """EffectiveLLMConfig for a cache key, applying vision→main inheritance."""
    from app.services.runtime_config import get_effective_sync

    cfg = get_effective_sync(role)
    if role == "vision" and cfg.source != "db":
        # No dedicated vision override → inherit main's configuration exactly.
        cfg = get_effective_sync("main")
    return cfg


def _cached_provider(cache_key: str, cfg_role: str, build) -> LLMProvider:
    """Version-keyed provider cache. Sync, non-blocking, lazy rebuild."""
    from app.services.runtime_config import snapshot_version

    ver = snapshot_version()
    hit = _PROVIDERS.get(cache_key)
    if hit and hit[0] == ver:
        return hit[1]
    provider = build()
    _PROVIDERS[cache_key] = (ver, provider)
    return provider


def get_llm_provider(role: str = "main") -> LLMProvider:
    """Main chat/vision LLM provider.

    Wrapped with Langfuse generation tracing (label ``main_llm``) so every
    answer-generator / direct-answer / evaluator / captioner LLM call is
    captured. ``role="vision"`` returns a dedicated vision override when one is
    configured in the DB, otherwise exactly the "main" configuration.
    """
    from app.services.agent.langfuse_tracing import trace_llm
    from app.services.runtime_config import _build_from_settings, get_effective_sync

    if role not in ("main", "vision"):
        raise ValueError(f"get_llm_provider() supports roles 'main'/'vision', got {role!r}")

    def _build():
        try:
            cfg = _resolve_role_config(role)
        except Exception as exc:
            # Fail-open: broken override resolution must not take chat down.
            import logging

            logging.getLogger(__name__).warning(
                f"[llm] resolving {role} failed ({exc}) — using .env defaults"
            )
            cfg = _build_from_settings("main" if role == "vision" else role)
        inner = build_provider(cfg)
        if role == "vision":
            return trace_llm(inner, label="vision_llm")
        return trace_llm(inner, label="main_llm")

    return _cached_provider(f"role:{role}", role, _build)


def get_memory_agent() -> LLMProvider:
    """Dedicated LLM provider for internal agent tasks (memory, classification,
    condensing, contextual embeddings).

    Wrapped with Langfuse generation tracing (label ``memory_agent``).
    MEMORY_AGENT_LOCAL=true keeps the legacy in-process vLLM branch reading
    Settings directly — that mode is not remotely reconfigurable.
    """
    from app.core.config import settings

    def _build():
        from app.services.agent.langfuse_tracing import trace_llm
        from app.services.runtime_config import (
            _build_from_settings,
            get_effective_sync,
        )

        if settings.MEMORY_AGENT_LOCAL:
            from app.services.llm.vllm_local import LocalVLLMProvider

            # Legacy local-GPU path: static config, no runtime override.
            # Still routed through _cached_provider below so the engine is
            # built once per config version instead of on every call.
            return trace_llm(
                LocalVLLMProvider(
                    model=settings.MEMORY_AGENT_MODEL,
                    gpu_memory_utilization=settings.MEMORY_AGENT_GPU_UTILIZATION,
                    cuda_device=settings.MEMORY_AGENT_CUDA_DEVICE,
                ),
                label="memory_agent",
            )

        try:
            cfg = get_effective_sync("memory_agent")
            if cfg.source != "db":
                cfg = _build_from_settings("memory_agent")
            inner = build_provider(cfg)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                f"[llm] resolving memory_agent failed ({exc}) — using .env defaults"
            )
            inner = OpenAICompatFallback().memory_agent_fallback()
        return trace_llm(inner, label="memory_agent")

    return _cached_provider("role:memory_agent", "memory_agent", _build)


class OpenAICompatFallback:
    """Last-resort construction straight from Settings (kept separate so the
    happy path stays readable)."""

    @staticmethod
    def memory_agent_fallback() -> LLMProvider:
        from app.core.config import settings
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider

        return OpenAICompatibleLLMProvider(
            base_url=settings.MEMORY_AGENT_BASE_URL,
            model=settings.MEMORY_AGENT_MODEL,
            api_key=settings.MEMORY_AGENT_API_KEY,
        )


def _resolve_thinking_endpoint() -> tuple[str, str, str]:
    """Resolve 9router gateway URL + stable model alias + auth token.

    Preference per field:
      NEXUSRAG_LG_THINKING_* → ANTHROPIC_* (9router env style) → built-in default

    The ``model`` value is a FIXED alias (``nexusrag-thinking``). Remap it to a
    real upstream on 9router — the app does not restart for A/B tests.
    Kept for backward compatibility with existing imports/tests.
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


def get_thinking_provider() -> LLMProvider:
    """LLM for LangGraph routing (+ optional ReAct judge).

    Reads the effective "thinking" config (DB override over NEXUSRAG_LG_THINKING_*
    / ANTHROPIC_* aliases). Always uses alias_model=True: request + Langfuse use
    the stable alias even if the proxy echoes a different upstream model id.
    """
    def _build():
        from app.services.agent.langfuse_tracing import trace_llm
        from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider
        from app.services.runtime_config import _build_from_settings, get_effective_sync

        try:
            cfg = get_effective_sync("thinking")
            if cfg.source != "db":
                cfg = _build_from_settings("thinking")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                f"[llm] resolving thinking failed ({exc}) — using .env defaults"
            )
            base_url, model, api_key = _resolve_thinking_endpoint()
            from app.services.runtime_config import EffectiveLLMConfig

            cfg = EffectiveLLMConfig(
                provider="openai_compatible", base_url=base_url, model=model, api_key=api_key
            )
        inner = OpenAICompatibleLLMProvider(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key or "sk-nexusrag",
            alias_model=True,
        )
        return trace_llm(inner, label="thinking_llm")

    return _cached_provider("role:thinking", "thinking", _build)


def clear_thinking_provider_cache() -> None:
    """Drop cached providers. Backward-compat shim: normal A/B swaps happen via
    runtime-config version bumps; this now clears everything (tests only)."""
    _PROVIDERS.clear()


def get_kg_llm_provider() -> LLMProvider:
    """Dedicated LLM provider for LegalKG extraction tasks.

    Reads the effective "kg_extract" config (DB override over LEGAL_KG_LLM_*).
    Not Langfuse-traced (matches pre-feature behaviour — KG calls have their own
    MinIO extraction logging gated by HRAG_KG_LOG_EXTRACTION).
    """
    def _build():
        from app.services.runtime_config import _build_from_settings, get_effective_sync

        try:
            cfg = get_effective_sync("kg_extract")
            if cfg.source != "db":
                cfg = _build_from_settings("kg_extract")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                f"[llm] resolving kg_extract failed ({exc}) — using .env defaults"
            )
            cfg = _build_from_settings("kg_extract")
        return build_provider(cfg)

    return _cached_provider("role:kg_extract", "kg_extract", _build)


def get_embedding_provider() -> EmbeddingProvider:
    """Create (and cache) the embedding provider for KG (LightRAG).

    Unchanged by the runtime-config feature: embedding/reranker models are
    GPU-loaded at startup and swapping them invalidates stored vectors — they
    stay `.env`-managed (see plan §1 scope-out).
    """
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
    "build_provider",
    "get_llm_provider",
    "get_memory_agent",
    "get_thinking_provider",
    "_resolve_thinking_endpoint",
    "clear_thinking_provider_cache",
    "get_kg_llm_provider",
    "get_embedding_provider",
    "LLMProvider",
    "EmbeddingProvider",
]
