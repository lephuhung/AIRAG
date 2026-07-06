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

    # ── Redis: shared cross-process state so the backend can run >1 worker ──
    # process / replica. Gates stream-cancel pub/sub (Phase 1), the distributed
    # GPU semaphore (Phase 2) and the shared retrieval cache (Phase 3). When
    # REDIS_ENABLED=false (default) every caller falls back to its existing
    # in-process behaviour and no Redis connection is opened.
    REDIS_ENABLED: bool = Field(default=False)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Number of backend worker processes (uvicorn --workers). Default 1 keeps the
    # single-process behaviour. Raising it REQUIRES REDIS_ENABLED=true (cross-
    # process Stop / GPU cap / cache) + GPU headroom for N× retrieval models.
    WEB_CONCURRENCY: int = Field(default=1)

    # ── GPU search concurrency (shared-VRAM guard for the RAG fan-out) ──────
    # Tier 1 (always on): per-process soft cap. Tier 2 (only when REDIS_ENABLED):
    # a cluster-wide hard cap so N backend worker processes / replicas sharing
    # the GPU can't multiply the activation-memory peak N-fold. Set GLOBAL=0 to
    # disable the distributed tier. PERMIT_TTL self-heals leaked permits after a
    # crash; WAIT_TIMEOUT bounds how long a search waits for a global slot before
    # proceeding local-only (fail-open — never wedge a user query).
    HRAG_SEARCH_GPU_CONCURRENCY: int = Field(default=2)
    HRAG_SEARCH_GPU_GLOBAL_CONCURRENCY: int = Field(default=2)
    HRAG_SEARCH_GPU_PERMIT_TTL: float = Field(default=120.0)
    HRAG_SEARCH_GPU_WAIT_TIMEOUT: float = Field(default=30.0)

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

    # STT (Speech-to-Text) — engine-agnostic provider layer (app/services/stt/)
    STT_ENABLED: bool = Field(default=True)
    STT_PROVIDER: str = Field(default="faster_whisper")  # faster_whisper | openai
    STT_LANGUAGE: str = Field(default="vi")  # forced language; "" → auto-detect
    STT_MAX_UPLOAD_MB: int = Field(default=25)
    # faster-whisper (local/offline, default)
    STT_FW_MODEL: str = Field(default="large-v3")  # tiny|base|small|medium|large-v3
    STT_FW_DEVICE: str = Field(default="auto")  # cpu|cuda|auto
    STT_FW_COMPUTE_TYPE: str = Field(default="default")  # int8|float16|default
    STT_FW_MODEL_DIR: str = Field(default="")  # HF cache dir; "" → library default
    # OpenAI-compatible (vLLM /v1/audio/transcriptions) — optional
    STT_OPENAI_BASE_URL: str = Field(default="")
    STT_OPENAI_MODEL: str = Field(default="whisper-1")
    STT_OPENAI_API_KEY: str = Field(default="")

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
    # NOTE: in the deployed stack this resolves to the memory vLLM alias
    # (gemma-4-E4B served as `qwen-memory`); .env / compose override it.
    LEGAL_KG_LLM_MODEL: str = Field(default="qwen-memory")
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
    # Uses the memory agent (gemma-4-E4B, served as `qwen-memory`) — no extra model needed.
    HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS: bool = Field(default=False)
    HRAG_CONTEXTUAL_MAX_TOKENS: int = Field(
        default=120
    )  # max tokens for generated context sentence
    HRAG_CONTEXTUAL_CONCURRENCY: int = Field(
        default=8
    )  # parallel LLM calls per document

    # Log every KG extraction LLM call (prompt + completion) to MinIO as JSONL,
    # for collecting fine-tuning datasets. Stored under datasets/legal_kg_extraction/.
    HRAG_KG_LOG_EXTRACTION: bool = Field(default=True)

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
    # Nhân điểm rerank cho chunk của văn bản ĐÃ BỊ THAY THẾ (validity_status=
    # 'superseded') khi truy vấn không scope đích danh văn bản. 1.0 = tắt.
    HRAG_SUPERSEDED_DEMOTE: float = Field(default=0.5)

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
            "THAY_THE",
            "BAI_BO",
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
    # Force full-page OCR (EasyOCR) for EVERY Docling PDF. Default OFF: most PDFs
    # have a clean text layer that Docling reads correctly (incl. Vietnamese) —
    # far faster + more accurate than OCR. Broken-text-layer Vietnamese PDFs are
    # handled instead by HRAG_OCR_DETECT_BROKEN_VN (routed to Unlimited-OCR).
    # Only set True for a corpus where most text layers are corrupt.
    HRAG_DOCLING_FORCE_FULL_PAGE_OCR: bool = Field(default=False)
    # Detect PDFs whose embedded text layer is Vietnamese but has dropped tone
    # marks (corrupt font/encoding → "BỘ"→"B") and route them to the OCR
    # pipeline (Unlimited-OCR reads Vietnamese reliably) instead of Docling.
    HRAG_OCR_DETECT_BROKEN_VN: bool = Field(default=True)
    # Min ratio of complex Vietnamese tone chars (U+1EA0–1EF9) to letters for a
    # text layer to count as "clean Vietnamese". Below this (with VN base letters
    # present) → treated as broken → OCR. Clean VN docs run ~5–15%.
    HRAG_OCR_VN_TONE_MIN_RATIO: float = Field(default=0.02)

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
    HRAG_RERANKER_DEVICE: str = Field(default="auto")

    # Remote embed/rerank microservice (scale-out). When HRAG_EMBED_RERANK_URL is
    # set, EmbeddingService/RerankerService become thin HTTP clients calling the
    # hrag-embed-rerank service instead of loading the models in-process — so
    # backend workers hold NO GPU state and can scale on CPU (WEB_CONCURRENCY>1).
    # Unset (default) = in-process load, which is also how the microservice
    # itself runs. See docs/scaling.md.
    HRAG_EMBED_RERANK_URL: str = Field(default="")
    HRAG_EMBED_RERANK_TIMEOUT: float = Field(default=30.0)

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
    # Unlimited-OCR is a ~3B model — 0.20 (~10 GB on a 48 GB card) is the floor.
    HRAG_OCR_GPU_MEMORY_UTILIZATION: float = Field(default=0.20)
    # Max sequence length passed to vLLM; None = use model default.
    HRAG_OCR_MAX_MODEL_LEN: int | None = Field(default=None)
    HUNYUAN_OCR_API_URL: str = Field(default="http://localhost:8001/v1")
    # Served-model-name of the OCR engine (vllm-ocr serves Baidu Unlimited-OCR).
    # Env var kept as HUNYUAN_OCR_MODEL for backward compatibility.
    HUNYUAN_OCR_MODEL: str = Field(default="unlimited-ocr")
    # DPI used to rasterise PDF pages before sending to the OCR model.
    HRAG_OCR_DPI: int = Field(default=150)
    # Max pages OCR'd concurrently against the remote API backend.
    HRAG_OCR_CONCURRENCY: int = Field(default=16)
    # HTTP read timeout (seconds) for a single OCR API page request.
    HRAG_OCR_HTTP_TIMEOUT: float = Field(default=120.0)
    # max_tokens requested per OCR API page completion. Must stay well below
    # the OCR engine's max-model-len (12288) minus the image prompt (~1.7-3.5k
    # tokens) — Unlimited-OCR counts prompt + completion against one context.
    HRAG_OCR_API_MAX_TOKENS: int = Field(default=6144)
    # Reconstruct the original administrative-document layout (centre/right
    # alignment, 2-column Nghị-định-30 header, signature block) from the
    # Unlimited-OCR bounding boxes and store it as layout HTML in the document
    # markdown (rendered by DocumentViewer). Embeddings stay clean — the embed
    # worker strips the layout markup before vectorising. Set false to fall back
    # to flat reading-order text.
    HRAG_OCR_PRESERVE_LAYOUT: bool = Field(default=True)
    # Same administrative-layout reconstruction, but for the DIGITAL (Docling)
    # path: rebuild the layout from each element's prov.bbox + semantic label
    # instead of Docling's flat markdown export. Default OFF — Docling's markdown
    # is well-tuned for general documents; enable for born-digital admin docs.
    HRAG_DOCLING_PRESERVE_LAYOUT: bool = Field(default=False)

    # Chunk văn bản pháp luật theo ranh giới Phần/Chương/Mục/Điều trên đường
    # OCR/legacy (mỗi Điều là một đơn vị trọn vẹn, điều dài mới size-split)
    # thay vì cắt phẳng 500 ký tự trộn cuối điều này với đầu điều kia.
    # Chỉ kích hoạt khi văn bản có >=3 heading "Điều N." — công văn/tờ trình
    # vẫn đi DocumentChunker thường.
    HRAG_LEGAL_CHUNKING: bool = Field(default=True)
    HRAG_LEGAL_CHUNK_MAX_CHARS: int = Field(default=1800)

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
    # Memory worker: LLM fact-extraction (gemma-4-E4B, served as `qwen-memory`) + Graphiti Neo4j write.
    WORKER_MEMORY_TIMEOUT: int = Field(default=90)

    # ── LangGraph Agent ──────────────────────────────────────────────────────
    # The LangGraph supervisor is the only chat agent backend (legacy path removed).

    # Max agent iterations (loop guard for LangGraph tool_executor → answer cycle)
    NEXUSRAG_LG_MAX_ITERATIONS: int = Field(default=6)

    # Classifier model: reuse the memory agent (gemma-4-E4B, served as `qwen-memory`) for intent classification.
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

    # ── Agent trace dataset (distillation data capture) ──────────────────────
    # When True, every agent run (web + Telegram, all via stream_agent_events)
    # records a structured trace — routing decisions, LLM I/O (dynamic messages
    # + completions; system prompts are NOT stored, only a hash reference since
    # they live in app/prompts/), and tool calls + results — into the
    # agent_traces table. PII (people-search results) is redacted before write.
    # Used to build SFT/distillation datasets for a smaller model. Best-effort:
    # capture failures never affect the chat response. Export via
    # `python -m scripts.export_agent_traces`.
    NEXUSRAG_TRACE_DATASET: bool = Field(default=True)

    # ── ReAct executor for the RAG group (tool-aware planning) ───────────────
    # When True, the supervisor routes RAG-group queries to a single tool-calling
    # ReAct loop (react_executor_node) instead of the static intent→tool nodes.
    # Requires a provider with reliable native tool-calling (vLLM OpenAI-compat).
    NEXUSRAG_LG_RAG_REACT: bool = Field(default=False)
    # Max tool-calling rounds before forcing a final synthesis (loop guard).
    # This is now a SAFETY NET, not the primary stop control — the sufficiency
    # gate in react_executor_node makes the model answer as soon as it has
    # enough data, so most queries finish in 2-3 rounds well under this cap.
    NEXUSRAG_REACT_MAX_TOOL_STEPS: int = Field(default=4)
    # top_k passed to search tools inside the ReAct loop.
    NEXUSRAG_REACT_TOP_K: int = Field(default=8)
    # Emit reasoning tokens (<think>) on the tool-decision turns so the model
    # reasons about WHICH tool to call before calling it. Measured: ~3.5s/turn
    # overhead on Qwen3.6-35B WITHOUT changing which tool gets chosen — so it is
    # OFF by default. The real self-evaluation/reasoning value lives in the
    # sufficiency gate + LLM-judge (which reason over the retrieved data), not in
    # per-tool-call thinking. Flip to True to restore per-turn reasoning.
    NEXUSRAG_REACT_THINK_TOOL_TURNS: bool = Field(default=False)
    # LLM-as-judge gate: before finalising, an LLM scores the draft answer
    # (grounded? covers the plan? no invented citations?) and can send the loop
    # back to gather more. Set False to skip the judge (debug latency).
    NEXUSRAG_REACT_JUDGE: bool = Field(default=True)
    # How many times the judge may bounce the draft back for revision before the
    # answer is forced out (bounds the reflect→act→reflect loop).
    NEXUSRAG_REACT_MAX_REFLECTIONS: int = Field(default=2)
    # Grounding guard (anti-fabrication enforcement). After synthesis+judge, if
    # the answer cites Vietnamese legal document numbers (e.g. "13/2023/NĐ-CP")
    # that appear NOWHERE in the retrieved sources/tool results AND the judge was
    # unsatisfied (verdict=revise) or was skipped, the answer is RETRACTED
    # (token_rollback on the live-streamed path) and replaced with an honest
    # "not enough grounded basis" answer that lists which cited numbers are not
    # in the kho. This ENFORCES the judge's revise verdict — the live-streaming
    # synthesis path otherwise only softened it with a caveat while the fabricated
    # body still reached the user. Set False to restore caveat-only behaviour.
    NEXUSRAG_REACT_GROUNDING_GUARD: bool = Field(default=True)
    # Sufficiency gate (pre-synthesis anti-fabrication). BEFORE the final answer
    # is written, the memory agent judges whether the collected sources actually
    # cover what was asked. If not — retrieval returned nothing, or chunks that
    # are on-topic-ish but miss the question's core (e.g. "Điều 8 phân loại cấp
    # độ" retrieved for a question about "xử phạt") — synthesis is SKIPPED and an
    # honest "not enough grounded basis" reply is returned, so the model is never
    # asked to write prose it can't ground (the partial-grounding fabrication
    # class). Fail-open: a flaky/errored check lets synthesis proceed (the output
    # grounding guard backstops). Set False to always synthesise.
    NEXUSRAG_REACT_SUFFICIENCY_GATE: bool = Field(default=True)
    # Targeted retry for the sufficiency gate: when the gate declares the sources
    # insufficient and NAMES the missing aspect, run ONE more search aimed at
    # exactly that aspect before giving up, then re-check. Rescues the case where
    # the earlier loop's query phrasing missed a document that IS in the kho;
    # when the document genuinely isn't there, the retry finds nothing new and
    # the honest fallback still fires. Bounded to a single extra search (no loop)
    # so latency stays predictable. Requires NEXUSRAG_REACT_SUFFICIENCY_GATE.
    NEXUSRAG_REACT_SUFFICIENCY_RETRY: bool = Field(default=True)
    # Prior conversation turns injected into the ReAct system prompt as a digest
    # ("HỘI THOẠI TRƯỚC ĐÓ") so the loop can resolve references the follow-up
    # condenser missed ("văn bản này", "điều đó", elliptical follow-ups). The
    # block is marked non-authoritative — answers still come from tools (rule
    # 3b). Set 0 to disable (kill-switch if tool-call rate regresses).
    NEXUSRAG_REACT_HISTORY_TURNS: int = Field(default=6)
    # Follow-up condense gating. True (default): every first-pass turn with
    # prior history (and no explicit doc reference in the question) goes to the
    # memory-agent JUDGE — one call returning an explicit dependence verdict
    # ({"phu_thuoc": ...}) plus the rewrite, applied only when dependent. False:
    # legacy behavior — the _FOLLOWUP_CUES regex decides which turns reach the
    # LLM (kill-switch if small-model latency/JSON reliability regresses; note
    # the cue list over-triggers on "như thế nào" interrogatives).
    NEXUSRAG_CONDENSE_LLM_JUDGE: bool = Field(default=True)

    # ── Graphiti Memory (temporal knowledge graph, backed by Neo4j) ──────────
    # Graphiti uses the existing Neo4j instance (NEO4J_URI / NEO4J_USERNAME /
    # NEO4J_PASSWORD above) for graph storage.
    # LLM used by Graphiti for entity/fact extraction from conversations.
    # Defaults to the memory agent (gemma-4-E4B, served as `qwen-memory`) — no extra model needed.
    GRAPHITI_LLM_BASE_URL: str = Field(default="http://localhost:8088/v1")
    GRAPHITI_LLM_MODEL: str = Field(default="qwen-memory")
    GRAPHITI_LLM_API_KEY: str = Field(default="sk-nexusrag")
    # Embedding dimension — must match HRAG_EMBEDDING_MODEL's output dim
    # (vietlegal-harrier-0.6b = 1024). Update both together if you swap models.
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

    # Public origin the *frontend* (web UI) is reachable at from a browser, e.g.
    # "https://service.hatinh.local". Used to build invite links
    # ({FRONTEND_BASE_URL}/register?invite=...) deterministically instead of
    # guessing from the request's Origin header (which is absent for non-browser
    # callers and wrong behind a reverse proxy). Empty → fall back to
    # PUBLIC_BASE_URL, then the request Origin, then the dev :8080→:5174 swap.
    # No trailing slash.
    FRONTEND_BASE_URL: str = Field(default="")

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
