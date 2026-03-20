from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    APP_NAME: str = "NexusRAG"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Infrastructure
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag")
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
    MINIO_BUCKET_UPLOADS: str = Field(default="nexusrag-uploads")
    MINIO_BUCKET_MARKDOWN: str = Field(default="nexusrag-markdown")
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
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=8192)
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="gemma3:12b")
    OLLAMA_ENABLE_THINKING: bool = Field(default=False)

    # OpenAI-compatible provider (vLLM, LM Studio, llama.cpp, etc.)
    OPENAI_COMPATIBLE_BASE_URL: str = Field(default="http://127.0.0.1:8000/v1")
    OPENAI_COMPATIBLE_MODEL: str = Field(default="default")
    OPENAI_COMPATIBLE_API_KEY: str = Field(default="none")

    # KG Embedding
    KG_EMBEDDING_PROVIDER: str = Field(default="local")
    KG_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    KG_EMBEDDING_DIMENSION: int = Field(default=1024)

    # Pipeline features
    NEXUSRAG_ENABLED: bool = Field(default=True)
    NEXUSRAG_ENABLE_KG: bool = Field(default=True)
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = Field(default=True)
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = Field(default=True)
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = Field(default=True)
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = Field(default=False)

    # Chunking
    NEXUSRAG_CHUNK_MAX_TOKENS: int = Field(default=512)

    # Knowledge Graph
    NEXUSRAG_KG_LANGUAGE: str = Field(default="Vietnamese")
    NEXUSRAG_KG_ENTITY_TYPES: list[str] = Field(default=[
        "Organization", "Person", "Product", "Location", "Event",
        "Financial_Metric", "Technology", "Date", "Regulation",
    ])
    NEXUSRAG_KG_CHUNK_TOKEN_SIZE: int = Field(default=1200)
    NEXUSRAG_KG_QUERY_TIMEOUT: float = Field(default=30.0)

    # Images & tables
    NEXUSRAG_DOCLING_IMAGES_SCALE: float = Field(default=2.0)
    NEXUSRAG_MAX_IMAGES_PER_DOC: int = Field(default=50)
    NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS: int = Field(default=8000)

    # Retrieval
    NEXUSRAG_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    NEXUSRAG_RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    NEXUSRAG_VECTOR_PREFETCH: int = Field(default=20)
    NEXUSRAG_RERANKER_TOP_K: int = Field(default=8)
    NEXUSRAG_MIN_RELEVANCE_SCORE: float = Field(default=0.15)
    NEXUSRAG_DEFAULT_QUERY_MODE: str = Field(default="hybrid")

    # GPU device placement
    NEXUSRAG_DOCLING_DEVICE: str = Field(default="auto")
    NEXUSRAG_EMBEDDING_DEVICE: str = Field(default="auto")

    # OCR
    NEXUSRAG_ENABLE_OCR: bool = Field(default=True)
    NEXUSRAG_OCR_SCANNED_THRESHOLD: float = Field(default=0.5)
    NEXUSRAG_OCR_LOCAL: bool = Field(default=False)
    # GPU index for the local vLLM OCR process.
    # Set to "1" if GPU 0 is occupied by a large LLM server.
    # Translates to CUDA_VISIBLE_DEVICES=<value> before vLLM initialises.
    # Use "0" for the first GPU, "1" for the second, "" or "auto" to leave
    # CUDA_VISIBLE_DEVICES unchanged (vLLM picks the first available GPU).
    NEXUSRAG_OCR_CUDA_DEVICE: str = Field(default="auto")
    # Fraction of the selected GPU's VRAM vLLM may use for the OCR model KV cache.
    # HunyuanOCR is a 1B model — 0.15 (~7 GB on a 47 GB card) is ample.
    NEXUSRAG_OCR_GPU_MEMORY_UTILIZATION: float = Field(default=0.15)
    # Max sequence length passed to vLLM; None = use model default.
    NEXUSRAG_OCR_MAX_MODEL_LEN: int | None = Field(default=None)
    HUNYUAN_OCR_API_URL: str = Field(default="http://10.8.0.8:8001/v1")
    HUNYUAN_OCR_MODEL: str = Field(default="tencent/HunyuanOCR")

    # Knowledge Graph backend
    NEXUSRAG_KG_GRAPH_BACKEND: str = Field(default="networkx")
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="nexusrag123")

    # Eager model loading — shift cold-start cost to startup
    NEXUSRAG_EAGER_MODEL_LOADING: bool = Field(default=True)
    NEXUSRAG_KG_PRE_INITIALIZE: bool = Field(default=True)

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
    FIRST_SUPERADMIN_EMAIL: str = Field(default="admin@nexusrag.local")
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
