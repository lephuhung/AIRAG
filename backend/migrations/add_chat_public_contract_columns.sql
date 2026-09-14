-- =============================================================
-- Phase 4D (Task 9): versioned public chat contract reload metadata
-- =============================================================
-- Stores the public citation projection and the structured
-- clarification resume block on assistant messages so completed /
-- clarified turns rebuild after refresh without the frontend
-- fabricating document/workspace/binding identity.
-- Nullable JSON: NULL for user messages and pre-contract turns.
-- v1 remains the rollback arm; no existing column is altered.
-- =============================================================

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS citations JSON,
    ADD COLUMN IF NOT EXISTS clarification JSON;
