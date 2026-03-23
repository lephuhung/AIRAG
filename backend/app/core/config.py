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
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5433/hrag")
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
    MEMORY_AGENT_BASE_URL: str = Field(default="http://localhost:8082/v1")
    MEMORY_AGENT_API_KEY: str = Field(default="none")
    MEMORY_AGENT_LOCAL: bool = Field(default=False)
    MEMORY_AGENT_GPU_UTILIZATION: float = Field(default=0.15)
    MEMORY_AGENT_CUDA_DEVICE: str = Field(default="auto")

    # OpenAI-compatible provider (vLLM, LM Studio, llama.cpp, etc.)
    OPENAI_COMPATIBLE_BASE_URL: str = Field(default="http://127.0.0.1:8000/v1")
    OPENAI_COMPATIBLE_MODEL: str = Field(default="default")
    OPENAI_COMPATIBLE_API_KEY: str = Field(default="none")

    # KG Embedding
    KG_EMBEDDING_PROVIDER: str = Field(default="local")
    KG_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    KG_EMBEDDING_DIMENSION: int = Field(default=1024)

    # Pipeline features
    HRAG_ENABLED: bool = Field(default=True)
    HRAG_ENABLE_KG: bool = Field(default=True)
    HRAG_ENABLE_IMAGE_EXTRACTION: bool = Field(default=True)
    HRAG_ENABLE_IMAGE_CAPTIONING: bool = Field(default=True)
    HRAG_ENABLE_TABLE_CAPTIONING: bool = Field(default=True)
    HRAG_ENABLE_FORMULA_ENRICHMENT: bool = Field(default=False)

    # Chunking
    HRAG_CHUNK_MAX_TOKENS: int = Field(default=512)

    # Contextual Embeddings (Anthropic-style: prepend LLM-generated context before embedding)
    # Reduces retrieval failure rate by ~35-49% at the cost of extra LLM calls during indexing.
    # Uses the memory agent (Qwen3-4B) — no extra model needed.
    HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS: bool = Field(default=False)
    HRAG_CONTEXTUAL_MAX_TOKENS: int = Field(default=120)   # max tokens for generated context sentence
    HRAG_CONTEXTUAL_CONCURRENCY: int = Field(default=8)    # parallel LLM calls per document

    # Knowledge Graph
    HRAG_KG_LANGUAGE: str = Field(default="Vietnamese")
    HRAG_KG_ENTITY_TYPES: list[str] = Field(default=[
        "Organization", "Person", "Product", "Location", "Event",
        "Financial_Metric", "Technology", "Date", "Regulation",
    ])
    HRAG_KG_CHUNK_TOKEN_SIZE: int = Field(default=1200)
    HRAG_KG_QUERY_TIMEOUT: float = Field(default=30.0)

    # Images & tables
    HRAG_DOCLING_IMAGES_SCALE: float = Field(default=2.0)
    HRAG_MAX_IMAGES_PER_DOC: int = Field(default=50)
    HRAG_MAX_TABLE_MARKDOWN_CHARS: int = Field(default=8000)

    # Retrieval
    HRAG_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    HRAG_RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    HRAG_VECTOR_PREFETCH: int = Field(default=20)
    HRAG_RERANKER_TOP_K: int = Field(default=8)
    HRAG_MIN_RELEVANCE_SCORE: float = Field(default=0.15)
    HRAG_DEFAULT_QUERY_MODE: str = Field(default="hybrid")

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
    WORKER_KG_POLL_INTERVAL: int = Field(default=30)  # seconds — how often to scan for new workspaces

    # CORS
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:5174", "http://localhost:3000"])

    # Authentication (JWT)
    JWT_SECRET_KEY: str = Field(default="change-me-in-production-use-a-real-secret-key")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)
    FIRST_SUPERADMIN_EMAIL: str = Field(default="admin@hrag.local")
    FIRST_SUPERADMIN_PASSWORD: str = Field(default="admin123")

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
