-- =============================================================
-- Agentic Memory: user_memories table
-- =============================================================
-- Run this script AFTER enabling pgvector extension:
--   CREATE EXTENSION IF NOT EXISTS vector;
-- =============================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_memories (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    embedding       vector(1024),
    category        VARCHAR(50) DEFAULT 'fact',
    importance      SMALLINT DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    source_session_id VARCHAR(36),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user ON user_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memories_category ON user_memories(user_id, category);

-- HNSW index for fast cosine similarity search on embeddings
CREATE INDEX IF NOT EXISTS idx_user_memories_embedding ON user_memories
    USING hnsw (embedding vector_cosine_ops);
