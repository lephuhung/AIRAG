from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    APP_NAME: str = "HRAG"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Infrastructure
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5433/hrag"
    )
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8002)
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672/")
    RABBITMQ_MANAGEMENT_URL: str = Field(default="http://localhost:15672")
    RABBITMQ_MANAGEMENT_USER: str = Field(default="guest")
    RABBITMQ_MANAGEMENT_PASS: str = Field(default="guest")

    # MinIO
    MINIO_ENDPOINT: str = Field(default="http://localhost:9000")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: str = Field(default="minioadmin")
    MINIO_BUCKET_UPLOADS: str = Field(default="hrag-uploads")
    MINIO_BUCKET_MARKDOWN: str = Field(default="hrag-markdown")
    MINIO_SECURE: bool = Field(default=False)
    MINIO_WEBHOOK_ENABLED: bool = Field(default=False)
    # Public URL reachable by the browser for presigned uploads.
    # In Docker: set to http://localhost:9000 (or your server IP).
    # Defaults to MINIO_ENDPOINT when not set.
    MINIO_PUBLIC_ENDPOINT: str = Field(default="")

    # LLM
    LLM_PROVIDER: str = Field(default="gemini")
    GOOGLE_AI_API_KEY: str = Field(default="")
    LLM_MODEL_FAST: str = Field(default="gemini-2.5-flash")
    LLM_THINKING_LEVEL: str = Field(default="medium")
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=4096)
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="gemma3:12b")
    OLLAMA_ENABLE_THINKING: bool = Field(default=False)
    MEMORY_AGENT_MODEL: str = Field(default="qwen-memory")
    MEMORY_AGENT_BASE_URL: str = Field(default="http://localhost:8088/v1")
    MEMORY_AGENT_API_KEY: str = Field(default="sk-nexusrag")
    MEMORY_AGENT_LOCAL: bool = Field(default=False)
    MEMORY_AGENT_GPU_UTILIZATION: float = Field(default=0.15)
    MEMORY_AGENT_CUDA_DEVICE: str = Field(default="auto")

    # OpenAI-compatible provider (vLLM, LM Studio, llama.cpp, etc.)
    OPENAI_COMPATIBLE_BASE_URL: str = Field(default="http://10.10.0.240:8000/v1")
    OPENAI_COMPATIBLE_MODEL: str = Field(default="default")
    OPENAI_COMPATIBLE_API_KEY: str = Field(default="sk-nexusrag")

    # TTS (Text-to-Speech) — engine-agnostic provider layer (app/services/tts/)
    TTS_ENABLED: bool = Field(default=True)
    TTS_PROVIDER: str = Field(default="omnivoice")  # omnivoice | <future>
    TTS_OMNIVOICE_BASE_URL: str = Field(default="http://omnivoice:8880/v1")
    TTS_OMNIVOICE_MODEL: str = Field(default="omnivoice")
    TTS_OMNIVOICE_API_KEY: str = Field(default="")  # empty = no auth on the TTS server
    TTS_DEFAULT_VOICE: str = Field(default="")  # empty → server default design prompt
    TTS_DEFAULT_SPEED: float = Field(default=1.0)
    TTS_MAX_CHARS: int = Field(default=4000)  # truncate over-long inputs

    # KG Embedding
    KG_EMBEDDING_PROVIDER: str = Field(default="local")
    KG_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    KG_EMBEDDING_DIMENSION: int = Field(default=1024)

    # LegalKG Extraction LLM — model for KG entity/relation extraction
    # Can use same provider as LLM_PROVIDER but specify different URL + model
    LEGAL_KG_LLM_PROVIDER: str = Field(
        default="openai_compatible"
    )  # gemini | ollama | openai_compatible
    LEGAL_KG_LLM_BASE_URL: str = Field(default="http://10.10.0.240:8000/v1")
    LEGAL_KG_LLM_MODEL: str = Field(default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    LEGAL_KG_LLM_API_KEY: str = Field(default="sk-nexusrag")

    # Pipeline features
    HRAG_ENABLED: bool = Field(default=True)
    HRAG_ENABLE_KG: bool = Field(default=True)
    HRAG_ENABLE_IMAGE_EXTRACTION: bool = Field(default=True)
    HRAG_ENABLE_IMAGE_CAPTIONING: bool = Field(default=True)
    HRAG_ENABLE_TABLE_CAPTIONING: bool = Field(default=True)
    HRAG_ENABLE_FORMULA_ENRICHMENT: bool = Field(default=False)

    # Aliases for NEXUSRAG_* prefix (same values, read from .env)
    NEXUSRAG_ENABLED: bool = Field(default=True)
    NEXUSRAG_ENABLE_KG: bool = Field(default=True)
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = Field(default=True)
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = Field(default=True)
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = Field(default=True)
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = Field(default=False)

    # Parse-only mode: skip embed/caption/kg workers, mark as INDEXED immediately after parse.
    # Use when you only need markdown storage (no RAG retrieval, no KG, no captioning).
    HRAG_PARSE_ONLY_MODE: bool = Field(default=False)
    NEXUSRAG_PARSE_ONLY_MODE: bool = Field(default=False)

    # Chunking
    HRAG_CHUNK_MAX_TOKENS: int = Field(default=512)

    # Contextual Embeddings (Anthropic-style: prepend LLM-generated context before embedding)
    # Reduces retrieval failure rate by ~35-49% at the cost of extra LLM calls during indexing.
    # Uses the memory agent (Qwen3-4B) — no extra model needed.
    HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS: bool = Field(default=False)
    HRAG_CONTEXTUAL_MAX_TOKENS: int = Field(
        default=120
    )  # max tokens for generated context sentence
    HRAG_CONTEXTUAL_CONCURRENCY: int = Field(
        default=8
    )  # parallel LLM calls per document

    # BM25 hybrid search (lexical search merged with vector via Reciprocal Rank Fusion)
    # Reduces retrieval failure rate by an additional ~14% on top of contextual embeddings.
    # No extra model needed — pure BM25 over in-memory corpus (per workspace, lazy-built).
    HRAG_ENABLE_BM25: bool = Field(default=True)
    HRAG_BM25_PREFETCH: int = Field(
        default=20
    )  # top-N BM25 candidates before RRF merge
    HRAG_RRF_K: int = Field(
        default=60
    )  # RRF constant (higher = smoother, 60 is standard)
    # BM25 parameters tuned for Vietnamese legal text (long sentences, sparse keywords)
    HRAG_BM25_K1: float = Field(default=2.0)  # higher = more weight to term frequency variation
    HRAG_BM25_B: float = Field(default=0.5)  # lower = less penalty for document length
    # Vietnamese word segmentation for BM25 tokenisation (pyvi). When enabled,
    # multi-syllable words ("quyết định") are kept as single tokens instead of
    # being split per-syllable, improving lexical match precision. Requires the
    # `pyvi` package; falls back to whitespace tokenisation if unavailable.
    # NOTE: changing this requires a fresh BM25 index (restart worker / API, or
    # invalidate cache) so the index and queries tokenise consistently.
    HRAG_BM25_WORD_SEGMENT: bool = Field(default=False)
    # Persist the in-memory BM25 index to disk so cold starts (process restart,
    # new replica) reload instead of rebuilding the whole corpus from ChromaDB.
    HRAG_BM25_PERSIST: bool = Field(default=True)
    # Recency boost: prefer newer documents based on published_date
    HRAG_RECENTNESS_BOOST: float = Field(default=0.3)  # 0=disabled, 1=full weight
    HRAG_RECENTNESS_DECAY_DAYS: int = Field(default=365)  # half-life in days

    # Knowledge Graph
    HRAG_KG_LANGUAGE: str = Field(default="Vietnamese")
    HRAG_KG_ENTITY_TYPES: list[str] = Field(
        default=["Article", "Person", "Organization", "Task"]
    )
    HRAG_KG_RELATION_TYPES: list[str] = Field(
        default=[
            "CAN_CU",
            "VIEN_DAN",
            "SUA_DOI",
            "CHU_TRI",
            "PHOI_HOP",
            "CHIU_TRACH_NHIEM",
        ]
    )
    # KG pipeline mode:
    #   "legal"    → LegalKGService (Vietnamese admin/legal docs, purpose-built)
    #   "lightrag" → original LightRAG generic pipeline (backward compat)
    HRAG_KG_MODE: str = Field(default="legal")
    HRAG_KG_CHUNK_TOKEN_SIZE: int = Field(default=1200)
    HRAG_KG_QUERY_TIMEOUT: float = Field(default=30.0)

    # Images & tables
    HRAG_DOCLING_IMAGES_SCALE: float = Field(default=2.0)
    HRAG_MAX_IMAGES_PER_DOC: int = Field(default=50)
    HRAG_MAX_TABLE_MARKDOWN_CHARS: int = Field(default=8000)
    # Docling TableFormer table-structure recognition.
    #   HRAG_DOCLING_DO_TABLE=false   → skip table structure entirely (fastest)
    #   HRAG_DOCLING_TABLE_MODE=fast  → faster, ~15-25% less accurate than "accurate"
    HRAG_DOCLING_DO_TABLE: bool = Field(default=True)
    HRAG_DOCLING_TABLE_MODE: str = Field(default="accurate")  # "fast" | "accurate"

    # Retrieval
    HRAG_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    HRAG_RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    HRAG_VECTOR_PREFETCH: int = Field(default=20)
    HRAG_RERANKER_TOP_K: int = Field(default=8)
    HRAG_MIN_RELEVANCE_SCORE: float = Field(default=0.15)
    HRAG_DEFAULT_QUERY_MODE: str = Field(default="hybrid")
    # Batch size for the SentenceTransformer encode() / cross-encoder predict() calls
    HRAG_EMBEDDING_BATCH_SIZE: int = Field(default=32)
    HRAG_RERANKER_BATCH_SIZE: int = Field(default=32)

    # GPU device placement
    HRAG_DOCLING_DEVICE: str = Field(default="auto")
    HRAG_EMBEDDING_DEVICE: str = Field(default="auto")

    # OCR
    HRAG_ENABLE_OCR: bool = Field(default=True)
    HRAG_OCR_SCANNED_THRESHOLD: float = Field(default=0.5)
    HRAG_OCR_LOCAL: bool = Field(default=False)
    # GPU index for the local vLLM OCR process.
    # Set to "1" if GPU 0 is occupied by a large LLM server.
    # Translates to CUDA_VISIBLE_DEVICES=<value> before vLLM initialises.
    # Use "0" for the first GPU, "1" for the second, "" or "auto" to leave
    # CUDA_VISIBLE_DEVICES unchanged (vLLM picks the first available GPU).
    HRAG_OCR_CUDA_DEVICE: str = Field(default="auto")
    # Fraction of the selected GPU's VRAM vLLM may use for the OCR model KV cache.
    # HunyuanOCR is a 1B model — 0.15 (~7 GB on a 47 GB card) is ample.
    HRAG_OCR_GPU_MEMORY_UTILIZATION: float = Field(default=0.15)
    # Max sequence length passed to vLLM; None = use model default.
    HRAG_OCR_MAX_MODEL_LEN: int | None = Field(default=None)
    HUNYUAN_OCR_API_URL: str = Field(default="http://localhost:8001/v1")
    HUNYUAN_OCR_MODEL: str = Field(default="hunyuan-ocr")
    # DPI used to rasterise PDF pages before sending to the OCR model.
    HRAG_OCR_DPI: int = Field(default=150)
    # Max pages OCR'd concurrently against the remote API backend.
    HRAG_OCR_CONCURRENCY: int = Field(default=16)
    # HTTP read timeout (seconds) for a single OCR API page request.
    HRAG_OCR_HTTP_TIMEOUT: float = Field(default=120.0)
    # max_tokens requested per OCR API page completion.
    HRAG_OCR_API_MAX_TOKENS: int = Field(default=8192)

    # Knowledge Graph backend
    HRAG_KG_GRAPH_BACKEND: str = Field(default="networkx")
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="hrag123")

    # Eager model loading — shift cold-start cost to startup
    HRAG_EAGER_MODEL_LOADING: bool = Field(default=True)
    HRAG_KG_PRE_INITIALIZE: bool = Field(default=True)

    # Worker tuning
    WORKER_PREFETCH_PARSE: int = Field(default=1)
    WORKER_PREFETCH_EMBED: int = Field(default=2)
    WORKER_PREFETCH_CAPTION: int = Field(default=1)
    WORKER_PREFETCH_KG: int = Field(default=1)
    WORKER_PREFETCH_MEMORY: int = Field(default=4)
    # Max concurrent LLM calls inside a single worker process.
    #   KG    — entity/relation extraction (legal_kg_service)
    #   CAPTION — image/table captioning (caption_worker)
    HRAG_KG_LLM_CONCURRENCY: int = Field(default=8)
    HRAG_CAPTION_CONCURRENCY: int = Field(default=4)
    WORKER_KG_POLL_INTERVAL: int = Field(
        default=60
    )  # seconds — how often to scan for new workspaces (adaptive: 300s when idle)
    # Timeout per message handler (seconds). Parse workers need higher timeout
    # for large scanned PDFs with OCR (120s recommended). Other workers use
    # shorter timeouts since they do not perform LLM inference on large files.
    WORKER_PARSE_TIMEOUT: int = Field(default=120)
    WORKER_EMBED_TIMEOUT: int = Field(default=60)
    WORKER_CAPTION_TIMEOUT: int = Field(default=60)
    WORKER_KG_TIMEOUT: int = Field(default=120)
    # Memory worker: LLM fact-extraction (Qwen3-4B) + Graphiti Neo4j write.
    WORKER_MEMORY_TIMEOUT: int = Field(default=90)

    # ── LangGraph Agent ──────────────────────────────────────────────────────
    # The LangGraph supervisor is the only chat agent backend (legacy path removed).

    # Max agent iterations (loop guard for LangGraph tool_executor → answer cycle)
    NEXUSRAG_LG_MAX_ITERATIONS: int = Field(default=6)

    # Classifier model: reuse the memory agent (Qwen3-4B) for intent classification.
    # Set to False to use the main LLM provider instead (slower but no extra model needed).
    NEXUSRAG_LG_USE_MEMORY_AGENT_AS_CLASSIFIER: bool = Field(default=True)

    # LangGraph checkpointer backend:
    #   "memory" — in-memory (no cross-request persistence, default)
    #   "none"   — no checkpointer
    NEXUSRAG_LG_CHECKPOINTER: str = Field(default="memory")

    # Toggle LangGraph internal debug logging (prints node execution/state to console)
    NEXUSRAG_LG_DEBUG: bool = Field(default=False)

    # ── Langfuse observability ───────────────────────────────────────────────
    # Wrap LLM providers so every call (supervisor classifier, answer generator,
    # direct answer, memory agent, query analyzer/enricher) emits a Langfuse
    # "generation" observation with model, full prompt/completion, parameters,
    # latency and (where the provider exposes it) token usage. These calls go
    # through custom providers — NOT LangChain ChatModels — so without this the
    # LangChain CallbackHandler cannot see any LLM I/O. Set False to disable.
    LANGFUSE_TRACE_LLM: bool = Field(default=True)

    # ── ReAct executor for the RAG group (tool-aware planning) ───────────────
    # When True, the supervisor routes RAG-group queries to a single tool-calling
    # ReAct loop (react_executor_node) instead of the static intent→tool nodes.
    # Requires a provider with reliable native tool-calling (vLLM OpenAI-compat).
    NEXUSRAG_LG_RAG_REACT: bool = Field(default=False)
    # Max tool-calling rounds before forcing a final synthesis (loop guard).
    NEXUSRAG_REACT_MAX_TOOL_STEPS: int = Field(default=6)
    # top_k passed to search tools inside the ReAct loop.
    NEXUSRAG_REACT_TOP_K: int = Field(default=8)

    # ── Graphiti Memory (temporal knowledge graph, backed by Neo4j) ──────────
    # Graphiti uses the existing Neo4j instance (NEO4J_URI / NEO4J_USERNAME /
    # NEO4J_PASSWORD above) for graph storage.
    # LLM used by Graphiti for entity/fact extraction from conversations.
    # Defaults to the memory agent (Qwen3-4B) — no extra model needed.
    GRAPHITI_LLM_BASE_URL: str = Field(default="http://localhost:8088/v1")
    GRAPHITI_LLM_MODEL: str = Field(default="qwen-memory")
    GRAPHITI_LLM_API_KEY: str = Field(default="sk-nexusrag")
    # Embedding dimension — must match HRAG_EMBEDDING_MODEL (BAAI/bge-m3 = 1024).
    GRAPHITI_EMBEDDING_DIM: int = Field(default=1024)

    # MongoDB — People Search
    MONGO_HOST: str = Field(default="localhost")
    MONGO_PORT: int = Field(default=27017)
    MONGO_USER: str = Field(default="admin")
    MONGO_PASSWORD: str = Field(default="changeme")
    MONGO_DATABASE: str = Field(default="people_db")
    MONGO_AUTH_SOURCE: str = Field(default="admin")

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5174", "http://localhost:3000"]
    )

    # Authentication (JWT)
    JWT_SECRET_KEY: str = Field(default="change-me-in-production-use-a-real-secret-key")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)
    FIRST_SUPERADMIN_EMAIL: str = Field(default="admin@hrag.local")
    FIRST_SUPERADMIN_PASSWORD: str = Field(default="admin123")

    # ── Integrations: Telegram bot + third-party API keys ──
    # Bot token from @BotFather. Empty disables the Telegram webhook handler.
    TELEGRAM_BOT_TOKEN: str = Field(default="")
    # Secret echoed by Telegram in the X-Telegram-Bot-Api-Secret-Token header;
    # set it when registering the webhook so we can reject spoofed requests.
    TELEGRAM_WEBHOOK_SECRET: str = Field(default="")
    # TTL (minutes) for the one-time code that links a Telegram chat to an account.
    TELEGRAM_LINK_CODE_TTL_MINUTES: int = Field(default=10)
    # Idle window (minutes) after which a Telegram chat's active session is
    # auto-expired and the next message starts a FRESH conversation — Telegram has
    # no multi-session UI, so without this old turns leak into new answers. The
    # user can still cut a session explicitly with /new. Set 0 to disable.
    TELEGRAM_SESSION_IDLE_MINUTES: int = Field(default=30)
    # Bot username (without @), e.g. "MyAiragBot" — used to build the t.me deep
    # link returned to the web UI. Optional (UI can fall back to showing the code).
    TELEGRAM_BOT_USERNAME: str = Field(default="")
    # Public origin the app is reachable at from the internet (the Cloudflare
    # tunnel domain), e.g. "https://bot.zbots.store". Used to build the Telegram
    # webhook URL instead of the internal `http://backend:8080`. Empty → fall back
    # to the request's own base URL. No trailing slash.
    PUBLIC_BASE_URL: str = Field(default="")

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
