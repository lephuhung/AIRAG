"""Standalone GPU microservice hosting the embedder + reranker over HTTP.

Runs the SAME EmbeddingService / RerankerService in in-process mode (its own
HRAG_EMBED_RERANK_URL is empty). The backend + workers become thin HTTP clients
so they hold no GPU state and can scale on CPU. See docs/scaling.md.
"""
