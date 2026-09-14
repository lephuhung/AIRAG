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


def _extract_chat_messages_alters() -> list[str]:
    import re

    text = MAIN_PY.read_text(encoding="utf-8")
    # Additive columns only: DROP COLUMN variants are not executable on
    # every engine and are out of scope for the reload-metadata guard.
    return re.findall(
        r"ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS \w+ [^\"]+",
        text,
    )


def test_ensure_alters_execute_on_a_legacy_schema():
    """Fix round 2 (N-M1): the guard is behavioural, not a grep — the
    exact ``ALTER`` statements extracted from ``main.py`` run against a
    legacy ``chat_messages`` table missing the contract columns and the
    columns must exist afterwards. SQLite accepts the statements (JSON is
    a valid affinity); Postgres semantics (IF NOT EXISTS) match.
    """
    import sqlite3

    alters = _extract_chat_messages_alters()
    assert alters, "no chat_messages ALTER statements found in main.py"
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE chat_messages ("
            "id INTEGER PRIMARY KEY, message_id VARCHAR(50), "
            "role VARCHAR(20), content TEXT)"
        )
        for statement in alters:
            # SQLite has no ADD COLUMN IF NOT EXISTS (Postgres-only
            # syntax): normalize the clause away — the test executes the
            # column/type portion, which is the reload-metadata substance.
            import re

            executable = re.sub(
                r"\s+IF NOT EXISTS", "", statement, count=1
            )
            conn.execute(executable)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(chat_messages)")
        }
        for column in REQUIRED_COLUMNS:
            assert column in columns, (
                f"executing main.py DDL did not create {column}"
            )
    finally:
        conn.close()


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
