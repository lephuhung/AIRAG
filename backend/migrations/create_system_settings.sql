-- create_system_settings.sql
-- DB-backed runtime overrides for LLM role configuration (WebUI LLM-config).
-- See docs/plan-llm-runtime-config.md §4. Safe to run multiple times.
--
-- Deploy order: run this BEFORE rolling out backend/workers images that read it —
-- the new code treats a missing table as "no overrides" (fail-open to .env).

CREATE TABLE IF NOT EXISTS system_settings (
    key         VARCHAR(128) PRIMARY KEY,
    value_enc   TEXT NOT NULL,              -- JSON; api_key stored Fernet-encrypted
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  UUID REFERENCES users(id)   -- nullable: system-initiated writes
);

-- Reserved version-counter row. Plain integer as TEXT (not encrypted) so the
-- per-message poll in workers is a single cheap SELECT.
INSERT INTO system_settings(key, value_enc)
VALUES ('_config_version', '0')
ON CONFLICT (key) DO NOTHING;
