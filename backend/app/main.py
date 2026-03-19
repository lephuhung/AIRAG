"""
NexusRAG — standalone Knowledge Base + RAG application.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine, Base

# Ensure all models are imported so Base.metadata knows about them
import app.models.knowledge_base  # noqa: F401
import app.models.document        # noqa: F401
import app.models.document_type   # noqa: F401
import app.models.chat_message    # noqa: F401
import app.models.user            # noqa: F401
import app.models.tenant          # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NexusRAG API...")
    import os
    auto_create = os.environ.get("AUTO_CREATE_TABLES", "true").lower() == "true"
    if auto_create:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Auto-migrate: add new columns if missing
            await conn.execute(
                text("ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS system_prompt TEXT")
            )
            # Ensure chat_messages table + indexes exist (idempotent)
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    workspace_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    message_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    sources JSON,
                    related_entities JSON,
                    image_refs JSON,
                    thinking TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_workspace_id ON chat_messages(workspace_id)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_message_id ON chat_messages(message_id)"
            ))
            await conn.execute(text(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS ratings JSON"
            ))
            await conn.execute(text(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS agent_steps JSON"
            ))
            # Worker pipeline sub-task flags
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS embed_done BOOLEAN DEFAULT FALSE"
            ))
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS captions_done BOOLEAN DEFAULT FALSE"
            ))
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS kg_done BOOLEAN DEFAULT FALSE"
            ))
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS raw_chunks_json TEXT"
            ))
            # MinIO migration: swap markdown_content column for markdown_s3_key
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS markdown_s3_key VARCHAR(500)"
            ))
            await conn.execute(text(
                "ALTER TABLE documents DROP COLUMN IF EXISTS markdown_content"
            ))
            # MinIO uploads: store the raw file S3 key
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS upload_s3_key VARCHAR(500)"
            ))
            # Digital signature metadata (native PDF only, JSON array)
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS digital_signatures JSON"
            ))
            # Document type classification
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type_id INTEGER "
                "REFERENCES document_types(id) ON DELETE SET NULL"
            ))
            # ── Auth & multi-tenant columns ────────────────────────────────────
            # knowledge_bases: visibility, owner_id, tenant_id
            await conn.execute(text(
                "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'personal'"
            ))
            await conn.execute(text(
                "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id)"
            ))
            await conn.execute(text(
                "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS tenant_id INTEGER REFERENCES tenants(id)"
            ))
            # documents: uploaded_by
            await conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS uploaded_by INTEGER REFERENCES users(id)"
            ))
            # chat_messages: user_id
            await conn.execute(text(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id)"
            ))
            # Ensure ALL lowercase enum values exist in PostgreSQL.
            # On a fresh DB, create_all() + values_callable creates them lowercase.
            # On an existing DB from before values_callable, create_all() made them
            # UPPERCASE — so we add lowercase variants and migrate data below.
            for _new_val in (
                'pending', 'parsing', 'ocring', 'chunking',
                'embedding', 'building_kg', 'indexed', 'failed',
            ):
                await conn.execute(text(
                    f"ALTER TYPE documentstatus ADD VALUE IF NOT EXISTS '{_new_val}'"
                ))
            # Migrate UPPERCASE enum values → lowercase (safe if already lowercase)
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'documents'
                    ) THEN
                        -- Migrate legacy statuses first
                        UPDATE documents SET status = 'indexed'
                            WHERE status::text IN ('processing', 'indexing', 'INDEXED');
                        UPDATE documents SET status = 'chunking'
                            WHERE status::text IN ('parsed', 'CHUNKING');
                        UPDATE documents SET status = 'embedding'
                            WHERE status::text IN ('indexed_partial', 'EMBEDDING');
                        -- Migrate remaining UPPERCASE → lowercase
                        UPDATE documents SET status = 'pending'
                            WHERE status::text = 'PENDING';
                        UPDATE documents SET status = 'parsing'
                            WHERE status::text = 'PARSING';
                        UPDATE documents SET status = 'ocring'
                            WHERE status::text = 'OCRING';
                        UPDATE documents SET status = 'building_kg'
                            WHERE status::text = 'BUILDING_KG';
                        UPDATE documents SET status = 'failed'
                            WHERE status::text = 'FAILED';
                    END IF;
                EXCEPTION WHEN others THEN
                    -- ignore: enum may not have legacy values
                    NULL;
                END $$;
            """))
        logger.info("Database tables created/verified")

        # Ensure MinIO buckets exist
        from app.services.storage_service import get_storage_service
        try:
            storage = get_storage_service()
            await storage.ensure_bucket()
            await storage.ensure_uploads_bucket()
            logger.info("MinIO buckets verified/created")
        except Exception as _minio_err:
            logger.warning(f"MinIO bucket setup failed (non-fatal): {_minio_err}")

        # Seed document types from classifier (idempotent — only inserts missing)
        try:
            from app.core.database import async_session_maker
            from app.models.document_type import DocumentType
            from app.services.document_type_classifier import get_all_document_types
            from sqlalchemy import select as _select

            async with async_session_maker() as _seed_db:
                for dt in get_all_document_types():
                    exists = await _seed_db.execute(
                        _select(DocumentType).where(DocumentType.slug == dt["slug"])
                    )
                    if exists.scalar_one_or_none() is None:
                        _seed_db.add(DocumentType(
                            slug=dt["slug"],
                            name=dt["name"],
                            description=dt["description"],
                        ))
                await _seed_db.commit()
            logger.info("Document types seeded/verified")
        except Exception as _seed_err:
            logger.warning(f"Document type seed failed (non-fatal): {_seed_err}")

        # ── Seed SuperAdmin user (idempotent) ─────────────────────────────────
        try:
            from app.core.database import async_session_maker
            from app.models.user import User as UserModel
            from app.core.security import hash_password
            from sqlalchemy import select as _select

            async with async_session_maker() as _admin_db:
                exists = await _admin_db.execute(
                    _select(UserModel).where(UserModel.email == settings.FIRST_SUPERADMIN_EMAIL)
                )
                if exists.scalar_one_or_none() is None:
                    admin = UserModel(
                        email=settings.FIRST_SUPERADMIN_EMAIL,
                        password_hash=hash_password(settings.FIRST_SUPERADMIN_PASSWORD),
                        full_name="Super Admin",
                        is_active=True,
                        is_superadmin=True,
                    )
                    _admin_db.add(admin)
                    await _admin_db.commit()
                    logger.info(f"SuperAdmin user created: {settings.FIRST_SUPERADMIN_EMAIL}")
                else:
                    logger.info("SuperAdmin user already exists")
        except Exception as _admin_err:
            logger.warning(f"SuperAdmin seed failed (non-fatal): {_admin_err}")

        # ── Migrate legacy workspaces: set visibility='public' for ownerless ──
        try:
            from app.core.database import async_session_maker

            async with async_session_maker() as _migrate_db:
                result = await _migrate_db.execute(text(
                    "UPDATE knowledge_bases SET visibility = 'public' "
                    "WHERE owner_id IS NULL AND visibility = 'personal'"
                ))
                if result.rowcount > 0:
                    await _migrate_db.commit()
                    logger.info(f"Migrated {result.rowcount} legacy workspaces to visibility='public'")
                else:
                    await _migrate_db.commit()
        except Exception as _mig_err:
            logger.warning(f"Legacy workspace migration failed (non-fatal): {_mig_err}")
    else:
        logger.info("AUTO_CREATE_TABLES=false — skipping auto-migration")
    yield
    logger.info("Shutting down...")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="NexusRAG — Knowledge Base with semantic search, knowledge graph, and LLM chat",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}


# API routes
from app.api.router import api_router  # noqa: E402

app.include_router(api_router, prefix="/api/v1")

# Static files — document images extracted by NexusRAG (Docling)
_docling_data = Path(__file__).resolve().parent.parent / "data" / "docling"
_docling_data.mkdir(parents=True, exist_ok=True)
app.mount("/static/doc-images", StaticFiles(directory=str(_docling_data)), name="static_doc_images")

# Import models so SQLAlchemy registers them
from app.models import knowledge_base, document, chat_message, user, tenant  # noqa: E402, F401
