"""
HRAG API router — aggregates workspace, document, and RAG endpoints.
"""
from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.workspaces import router as workspaces_router
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router
from app.api.config import router as config_router
from app.api.minio_events import router as minio_events_router
from app.api.document_types import router as document_types_router
from app.api.workers import router as workers_router
from app.api.tenants import router as tenants_router
from app.api.admin import router as admin_router
from app.api.chat_session import router as chat_session_router
from app.api.chat_agent_lg import router as chat_agent_lg_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(workspaces_router)
api_router.include_router(documents_router)
api_router.include_router(rag_router)
api_router.include_router(config_router)
api_router.include_router(minio_events_router)
api_router.include_router(document_types_router)
api_router.include_router(workers_router)
api_router.include_router(tenants_router)
api_router.include_router(admin_router)
api_router.include_router(chat_session_router)
api_router.include_router(chat_agent_lg_router)
