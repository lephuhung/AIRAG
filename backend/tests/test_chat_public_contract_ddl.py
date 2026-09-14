"""Task 9 fix round 1 (C2): the public-contract columns must be deployable.

Guards against a recurrence of mapped-without-DDL: the ORM-mapped reload
columns (``ChatMessage.citations`` / ``ChatMessage.clarification``) must be
covered BOTH by the startup ensure block in ``backend/app/main.py`` (the
repo's established mechanism for ``chat_messages`` columns, e.g.
``people_data``) AND by the standalone SQL migration for deployments that
run with ``AUTO_CREATE_TABLES=false``.
"""
from __future__ import annotations

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = BACKEND_ROOT / "app" / "main.py"
SQL_MIGRATION = (
    BACKEND_ROOT / "migrations" / "add_chat_public_contract_columns.sql"
)

REQUIRED_COLUMNS = ("citations", "clarification")


def test_startup_ensure_block_creates_public_contract_columns():
    text = MAIN_PY.read_text(encoding="utf-8")
    for column in REQUIRED_COLUMNS:
        needle = (
            "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS " + column
        )
        assert needle in text, (
            f"main.py startup ensure block must create chat_messages.{column} "
            "(C2: mapped columns with no executable DDL break send/history)"
        )


def test_sql_migration_covers_public_contract_columns():
    assert SQL_MIGRATION.exists(), (
        "standalone SQL migration for AUTO_CREATE_TABLES=false "
        "deployments must exist"
    )
    text = SQL_MIGRATION.read_text(encoding="utf-8")
    for column in REQUIRED_COLUMNS:
        assert column in text


def test_orm_model_maps_public_contract_columns():
    from app.models.chat_message import ChatMessage

    assert "citations" in ChatMessage.__table__.columns
    assert "clarification" in ChatMessage.__table__.columns
