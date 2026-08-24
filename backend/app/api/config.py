"""
Config status endpoint — expose active LLM/embedding provider info to frontend.
Reads the runtime LLM config (DB override → .env default) so the UI reflects
WebUI-configured models, not just .env.
"""
from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.deps import get_current_active_user
from app.models.user import User
from app.services import runtime_config

router = APIRouter(prefix="/config", tags=["config"])


def _role_summary(role: str) -> dict:
    cfg = runtime_config.get_effective_sync(role)
    return {
        "provider": cfg.provider,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "source": cfg.source,
    }


@router.get("/status")
async def get_config_status(
    user: User = Depends(get_current_active_user),
):
    """Return active provider and model names for UI display."""
    llm = _role_summary("main")
    kg = runtime_config.get_effective_sync("kg_extract")

    return {
        "llm_provider": llm["provider"],
        "llm_model": llm["model"],
        # Runtime-configured roles (DB override or .env default)
        "roles": {role: _role_summary(role) for role in runtime_config.ROLES},
        "kg_embedding_provider": settings.KG_EMBEDDING_PROVIDER.lower(),
        "kg_embedding_model": settings.KG_EMBEDDING_MODEL,
        "kg_embedding_dimension": settings.KG_EMBEDDING_DIMENSION,
        "hrag_embedding_model": settings.HRAG_EMBEDDING_MODEL,
        "hrag_reranker_model": settings.HRAG_RERANKER_MODEL,
    }
