"""
HRAG — standalone Knowledge Base + RAG application.
"""

import asyncio
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

# Import the whole models package so EVERY table is registered on
# Base.metadata BEFORE create_all() runs in the lifespan below. app/models/
# __init__.py imports every model module (knowledge_base, document, user,
# integration, audit_log, exchange_summary, …); importing the package here is
# the single source of truth and avoids relying on transitive imports from the
# API router to register tables like audit_logs / telegram_* / api_keys.
import app.models  # noqa: F401

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Update HRAG prefix in settings if needed (but currently we just use the val)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HRAG API...")
    import os

    auto_create = os.environ.get("AUTO_CREATE_TABLES", "true").lower() == "true"

    # ── Multi-worker readiness guard ──────────────────────────────────────
    # Each uvicorn/gunicorn worker is a SEPARATE process: it runs this whole
    # lifespan, opens its own DB pool, and (lazily) loads its OWN embedder +
    # reranker into GPU VRAM. Raising WEB_CONCURRENCY>1 therefore requires
    # REDIS_ENABLED (else Stop / GPU cap / retrieval cache fall back to
    # per-process state and stop being correct across workers), GPU headroom
    # for N× retrieval models, and running the inline migrations once up front.
    web_concurrency = int(os.environ.get("WEB_CONCURRENCY", "1") or "1")
    if web_concurrency > 1:
        if not settings.REDIS_ENABLED:
            logger.warning(
                "WEB_CONCURRENCY=%d but REDIS_ENABLED=false — the Stop button, the "
                "GPU-search concurrency cap and the retrieval cache are per-process "
                "and will NOT be correct across workers. Set REDIS_ENABLED=true.",
                web_concurrency,
            )
        if auto_create:
            logger.warning(
                "WEB_CONCURRENCY=%d with AUTO_CREATE_TABLES=true — all %d workers will "
                "run the inline schema migrations concurrently at boot. Prefer running "
                "migrations once with a single worker, then deploy with "
                "AUTO_CREATE_TABLES=false.",
                web_concurrency, web_concurrency,
            )
        logger.warning(
            "WEB_CONCURRENCY=%d — each worker loads its OWN embedder + reranker into "
            "GPU VRAM (~N× the retrieval-model footprint). Ensure GPU headroom or move "
            "retrieval models to a dedicated service before scaling.",
            web_concurrency,
        )

    if auto_create:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Auto-migrate: add new columns if missing
            await conn.execute(
                text(
                    "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS system_prompt TEXT"
                )
            )
            # Telegram link: numeric telegram user id (added after initial release)
            await conn.execute(
                text(
                    "ALTER TABLE IF EXISTS telegram_links ADD COLUMN IF NOT EXISTS telegram_user_id VARCHAR(64)"
                )
            )
            # Per-user preferences (TTS voice/speed, future settings)
            await conn.execute(
                text(
                    "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS settings JSONB NOT NULL DEFAULT '{}'::jsonb"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_telegram_links_telegram_user_id ON telegram_links(telegram_user_id)"
                )
            )
            # Create chat_sessions table if not exists
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL DEFAULT 'New Chat',
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions(user_id)"
                )
            )
            # Origin channel of the session: 'web' (default) or 'telegram'.
            await conn.execute(
                text(
                    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS source VARCHAR(16) NOT NULL DEFAULT 'web'"
                )
            )

            # Ensure chat_messages table + indexes exist (idempotent)
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    message_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    sources JSON,
                    related_entities JSON,
                    image_refs JSON,
                    thinking TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            )
            await conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS session_id VARCHAR(36) REFERENCES chat_sessions(id) ON DELETE CASCADE"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id ON chat_messages(session_id)"
                )
            )
            await conn.execute(
                text("ALTER TABLE chat_messages DROP COLUMN IF EXISTS workspace_id")
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_messages_message_id ON chat_messages(message_id)"
                )
            )
            await conn.execute(
                text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS ratings JSON")
            )
            await conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS agent_steps JSON"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS potential_abbreviations JSON"
                )
            )
            # Add document_ids column for file attachments
            await conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS document_ids JSON"
                )
            )
            # Add people_data column for MongoDB people search results
            await conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS people_data JSON"
                )
            )
            # Exchange summaries for per-Q&A conversation context
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS chat_exchange_summaries (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    exchange_index INTEGER NOT NULL,
                    user_message_id VARCHAR(50) NOT NULL,
                    assistant_message_id VARCHAR(50),
                    topic_label VARCHAR(255) NOT NULL,
                    key_entities JSON,
                    summary TEXT NOT NULL,
                    cited_sources JSON,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_exchange_summaries_session_id ON chat_exchange_summaries(session_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_exchange_summaries_exchange_index ON chat_exchange_summaries(exchange_index)"
                )
            )
            # Add cited_sources column if not exists (table may already exist from previous run)
            await conn.execute(
                text(
                    "ALTER TABLE chat_exchange_summaries ADD COLUMN IF NOT EXISTS cited_sources JSON"
                )
            )

            # Worker pipeline sub-task flags
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS embed_done BOOLEAN DEFAULT FALSE"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS captions_done BOOLEAN DEFAULT FALSE"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS kg_done BOOLEAN DEFAULT FALSE"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS raw_chunks_json TEXT"
                )
            )
            # MinIO migration: swap markdown_content column for markdown_s3_key
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS markdown_s3_key VARCHAR(500)"
                )
            )
            await conn.execute(
                text("ALTER TABLE documents DROP COLUMN IF EXISTS markdown_content")
            )
            # MinIO uploads: store the raw file S3 key
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS upload_s3_key VARCHAR(500)"
                )
            )
            # Digital signature metadata (native PDF only, JSON array)
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS digital_signatures JSON"
                )
            )
            # Document type classification
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type_id INTEGER "
                    "REFERENCES document_types(id) ON DELETE SET NULL"
                )
            )
            # Official document reference number (e.g. "13/2023/NĐ-CP")
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_number VARCHAR(100)"
                )
            )
            # Document title/subject extracted from header (e.g. "Luật Bảo vệ Bí mật nhà nước")
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_title VARCHAR(500)"
                )
            )
            # Manual signer name override
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS signer_name VARCHAR(255)"
                )
            )
            # Root Document node entity_id in Neo4j KG (used for metadata updates)
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS kg_root_entity_id VARCHAR(500)"
                )
            )
            # Rich Header Metadata extracted by LLM from Page 1 OCR
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS location VARCHAR(255)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS issuing_agency VARCHAR(255)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS parent_agency VARCHAR(255)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS published_date VARCHAR(100)"
                )
            )
            # ── Auth & multi-tenant columns ────────────────────────────────────
            # knowledge_bases: visibility, owner_id, tenant_id
            await conn.execute(
                text(
                    "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'personal'"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS tenant_id INTEGER REFERENCES tenants(id)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS is_default BOOLEAN DEFAULT FALSE"
                )
            )
            # documents: uploaded_by
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS uploaded_by INTEGER REFERENCES users(id)"
                )
            )
            # documents: is_chat_upload
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_chat_upload BOOLEAN DEFAULT FALSE"
                )
            )
            # documents: content_hash (SHA256 hex) for duplicate-upload detection
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents(content_hash)"
                )
            )
            # documents: trạng thái hiệu lực pháp lý (khác status = trạng thái xử lý).
            # validity_events lưu các tuyên bố thay_the/bai_bo/het_hieu_luc trích từ
            # điều khoản thi hành — dùng để cross-match khi văn bản bị ảnh hưởng
            # được upload SAU văn bản tuyên bố.
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS validity_status VARCHAR(30) DEFAULT 'unknown'"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded_by_number VARCHAR(100)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded_by_document_id UUID REFERENCES documents(id) ON DELETE SET NULL"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS effective_date VARCHAR(100)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS validity_events JSON"
                )
            )
            # chat_messages: user_id
            await conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id)"
                )
            )

            # document_type_system_prompts: kg_system_prompt
            await conn.execute(
                text(
                    "ALTER TABLE document_type_system_prompts ADD COLUMN IF NOT EXISTS kg_system_prompt TEXT"
                )
            )

            # --- Abbreviations Table (Explicit for clarity or metadata sync) ---
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS abbreviations (
                    id SERIAL PRIMARY KEY,
                    short_form VARCHAR(50) NOT NULL,
                    full_form VARCHAR(255) NOT NULL,
                    description TEXT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_abbreviations_short_form ON abbreviations(short_form)"
                )
            )
            # Ensure ALL lowercase enum values exist in PostgreSQL.
            # On a fresh DB, create_all() + values_callable creates them lowercase.
            # On an existing DB from before values_callable, create_all() made them
            # UPPERCASE — so we add lowercase variants and migrate data below.
            for _new_val in (
                "pending",
                "parsing",
                "ocring",
                "chunking",
                "embedding",
                "building_kg",
                "indexed",
                "failed",
            ):
                await conn.execute(
                    text(
                        f"ALTER TYPE documentstatus ADD VALUE IF NOT EXISTS '{_new_val}'"
                    )
                )
            # Migrate UPPERCASE enum values → lowercase (safe if already lowercase)
            await conn.execute(
                text("""
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
            """)
            )
            # --- chat_files table (docx/audio files attached to chat sessions) ---
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS chat_files (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    file_name VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    file_size INTEGER NOT NULL,
                    minio_original_key VARCHAR(500),
                    minio_markdown_key VARCHAR(500),
                    markdown_content TEXT,
                    report TEXT,
                    issues_count INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_files_session_id ON chat_files(session_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_files_user_id ON chat_files(user_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_files_workspace_id ON chat_files(workspace_id)"
                )
            )
            # --- format_metadata table (extracted docx formatting info) ---
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS format_metadata (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_file_id UUID NOT NULL REFERENCES chat_files(id) ON DELETE CASCADE,
                    format_data JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_format_metadata_chat_file_id ON format_metadata(chat_file_id)"
                )
            )
            # Trigger to auto-delete MinIO objects when chat_files row is deleted
            # Inserts into chat_files_cleanup for background worker to process
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS chat_files_cleanup (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    minio_original_key VARCHAR(500),
                    minio_markdown_key VARCHAR(500),
                    created_at TIMESTAMP DEFAULT NOW(),
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            )
            await conn.execute(
                text("""
                CREATE OR REPLACE FUNCTION chat_files_delete_cleanup()
                RETURNS TRIGGER AS $$
                BEGIN
                    INSERT INTO chat_files_cleanup (minio_original_key, minio_markdown_key)
                    VALUES (OLD.minio_original_key, OLD.minio_markdown_key);
                    RETURN OLD;
                END;
                $$ LANGUAGE plpgsql;
            """)
            )
            await conn.execute(
                text("""
                DROP TRIGGER IF EXISTS chat_files_cleanup_trigger ON chat_files;
            """)
            )
            await conn.execute(
                text("""
                CREATE TRIGGER chat_files_cleanup_trigger
                AFTER DELETE ON chat_files
                FOR EACH ROW
                EXECUTE FUNCTION chat_files_delete_cleanup();
            """)
            )
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

        # Seed document types from classifier defaults (idempotent — only inserts missing)
        try:
            from app.core.database import async_session_maker
            from app.services.document_type_classifier import seed_document_types

            async with async_session_maker() as _seed_db:
                await seed_document_types(_seed_db)
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
                    _select(UserModel).where(
                        UserModel.email == settings.FIRST_SUPERADMIN_EMAIL
                    )
                )
                existing_admin = exists.scalar_one_or_none()
                if existing_admin is None:
                    admin = UserModel(
                        email=settings.FIRST_SUPERADMIN_EMAIL,
                        password_hash=hash_password(settings.FIRST_SUPERADMIN_PASSWORD),
                        full_name="Super Admin",
                        is_active=True,
                        is_superadmin=True,
                    )
                    _admin_db.add(admin)
                    await _admin_db.commit()
                    logger.info(
                        f"SuperAdmin user created: {settings.FIRST_SUPERADMIN_EMAIL}"
                    )
                else:
                    # Ensure superadmin is always active and has superadmin flag
                    if not existing_admin.is_active or not existing_admin.is_superadmin:
                        existing_admin.is_active = True
                        existing_admin.is_superadmin = True
                        await _admin_db.commit()
                        logger.info(
                            f"SuperAdmin user re-activated: {settings.FIRST_SUPERADMIN_EMAIL}"
                        )
                    else:
                        logger.info("SuperAdmin user already exists and is active")
        except Exception as _admin_err:
            logger.warning(f"SuperAdmin seed failed (non-fatal): {_admin_err}")

        # ── Migrate legacy workspaces: set visibility='public' for ownerless ──
        try:
            from app.core.database import async_session_maker

            async with async_session_maker() as _migrate_db:
                result = await _migrate_db.execute(
                    text(
                        "UPDATE knowledge_bases SET visibility = 'public' "
                        "WHERE owner_id IS NULL AND visibility = 'personal'"
                    )
                )
                if result.rowcount > 0:
                    await _migrate_db.commit()
                    logger.info(
                        f"Migrated {result.rowcount} legacy workspaces to visibility='public'"
                    )
                else:
                    await _migrate_db.commit()
        except Exception as _mig_err:
            logger.warning(f"Legacy workspace migration failed (non-fatal): {_mig_err}")

        # ── Add avatar_url column to users (idempotent) ───────────────────────
        try:
            async with engine.begin() as _col_conn:
                await _col_conn.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(1024)"
                    )
                )
            logger.info("users.avatar_url column ensured")
        except Exception as _col_err:
            logger.warning(f"avatar_url migration failed (non-fatal): {_col_err}")

        # ── Add TOTP two-factor columns to users (idempotent) ─────────────────
        try:
            async with engine.begin() as _2fa_conn:
                await _2fa_conn.execute(
                    text("ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_secret VARCHAR(64)")
                )
                await _2fa_conn.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                        "totp_enabled BOOLEAN NOT NULL DEFAULT false"
                    )
                )
            logger.info("users.totp_secret / totp_enabled columns ensured")
        except Exception as _2fa_err:
            logger.warning(f"2FA columns migration failed (non-fatal): {_2fa_err}")
    else:
        logger.info("AUTO_CREATE_TABLES=false — skipping auto-migration")

    # ── Eager Model Loading ───────────────────────────────────────────────
    if settings.HRAG_EAGER_MODEL_LOADING:
        logger.info("Eager loading HRAG models...")
        from app.services.models.loader import preload_models

        # Pre-load embedder and reranker (and optionally local OCR)
        # to ensure the first request is fast.
        preload_models()
    else:
        # Lazy mode (default in Docker to avoid a VRAM spike at boot): instead of
        # loading the embedder/reranker on the FIRST user query — where the heavy
        # CrossEncoder init blocks the event loop for tens of seconds, causing the
        # request to look "stuck" until the frontend cancels it — warm them in the
        # BACKGROUND, off the event loop, right after startup completes. Startup
        # stays fast and the VRAM spike is deferred (not simultaneous with vLLM
        # boot, since the backend starts after infra is healthy).
        logger.info("HRAG_EAGER_MODEL_LOADING=false — warming retrieval models in background")

        async def _background_warm_models() -> None:
            try:
                from app.services.models.loader import preload_models
                # Run the blocking model load/warm in a worker thread so it never
                # stalls the event loop (request handling / health checks stay live).
                await asyncio.to_thread(preload_models)
                logger.info("[preload] Background retrieval-model warm-up complete")
            except Exception as _warm_err:
                logger.warning(f"[preload] Background warm-up failed (non-fatal): {_warm_err}")

        # Keep a reference on app.state so the task isn't garbage-collected mid-run.
        app.state._warm_models_task = asyncio.create_task(_background_warm_models())

    # ── Whisper (STT) Warm-up ─────────────────────────────────────────────
    # The first /stt/transcribe call otherwise loads the Whisper model inline
    # (~70s for large-v3), which can blow past proxy timeouts. Warm it in the
    # background so the first real dictation is fast. Only faster-whisper needs
    # this (the openai provider is remote).
    if settings.STT_ENABLED and settings.STT_PROVIDER == "faster_whisper":
        async def _background_warm_stt() -> None:
            try:
                from app.services.stt import get_stt_provider
                provider = get_stt_provider()
                if hasattr(provider, "warmup"):
                    await asyncio.to_thread(provider.warmup)
                    logger.info("[preload] Whisper (STT) warm-up complete")
            except Exception as _stt_err:
                logger.warning(f"[preload] STT warm-up failed (non-fatal): {_stt_err}")

        app.state._warm_stt_task = asyncio.create_task(_background_warm_stt())

    # ── Graphiti Memory Initialization ───────────────────────────────────
    # Build Neo4j indices and constraints required by Graphiti's knowledge
    # graph memory layer.  Idempotent — safe to call on every startup.
    # Non-fatal: if Neo4j is unavailable, memory falls back to empty context.
    try:
        from app.services.memory.graphiti_client import initialize_graphiti

        await initialize_graphiti()
    except Exception as _graphiti_err:
        logger.warning(f"Graphiti memory init failed (non-fatal): {_graphiti_err}")

    # ── MongoDB Connection Warmup ─────────────────────────────────────────
    # Pre-establish MongoDB TCP connection so the first search request is instant.
    # Non-fatal: if MongoDB is unreachable, falls back to lazy connect on first use.
    try:
        from app.services.people.mongo_client import get_mongo_client

        _mongo_client = get_mongo_client()
        # Force TCP handshake + auth now (pymongo connects lazily on first op)
        _mongo_client.admin.command("ping")
        logger.info("[mongo] Connection warmup OK")
    except Exception as _mongo_err:
        logger.warning(f"MongoDB warmup failed (non-fatal): {_mongo_err}")

    # ── Redis warmup + cross-process stream-cancel listener ───────────────
    # Only when REDIS_ENABLED — lets the backend run >1 worker/replica with a
    # working Stop button (the cancel may land on a different process than the
    # one running the agent). Non-fatal: if Redis is down we log and keep the
    # single-process behaviour for this run.
    if settings.REDIS_ENABLED:
        try:
            from app.core.redis_client import ping_redis
            from app.api.chat_session import start_cancel_listener

            if await ping_redis():
                await start_cancel_listener()
                logger.info("[redis] Connection OK — stream-cancel listener started")
            else:
                logger.warning(
                    "[redis] Ping failed — cross-process cancel disabled this run"
                )
        except Exception as _redis_err:
            logger.warning(f"Redis warmup failed (non-fatal): {_redis_err}")

    yield
    logger.info("Shutting down...")
    await engine.dispose()
    try:
        from app.services.people.mongo_client import close_mongo_client

        close_mongo_client()
    except Exception:
        pass
    try:
        from app.api.chat_session import stop_cancel_listener
        from app.core.redis_client import close_redis

        await stop_cancel_listener()
        await close_redis()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    description="HRAG — Knowledge Base with semantic search, knowledge graph, and LLM chat",
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

# Audit log — records management mutations (workspaces, tenants, users, …)
from app.middleware.audit import AuditMiddleware  # noqa: E402

app.add_middleware(AuditMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # BrokenPipeError (errno 32) and ConnectionResetError mean the client
    # disconnected before we finished writing the response. This is normal
    # behaviour (e.g. user cancels an upload). Log at WARNING, not ERROR.
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        logger.warning(
            f"Client disconnected ({type(exc).__name__}): "
            f"{request.method} {request.url.path}"
        )
        # Can't send a response — the pipe is gone. Return a minimal 499.
        return JSONResponse(
            status_code=499, content={"detail": "Client closed request"}
        )
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

# Static files — document images extracted by HRAG (Docling)
_docling_data = Path(__file__).resolve().parent.parent / "data" / "docling"
_docling_data.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/doc-images",
    StaticFiles(directory=str(_docling_data)),
    name="static_doc_images",
)

# NOTE: all models are already registered on Base.metadata via the
# `import app.models` at the top of this file (before create_all). No second
# import block is needed here.
