"""Release 1D (Task 10) — async PostgreSQL checkpoint lifecycle.

The v2 supervisor checkpoints through ``AsyncPostgresSaver`` against
``CHECKPOINT_DATABASE_URL`` — a psycopg3 DSN pointing at a database **separate**
from the application database (see ``app.core.config``). This module owns the
runtime lifecycle:

    validate_psycopg_dsn(checkpoint_dsn) -> None
    create_v2_checkpointer(checkpoint_dsn) -> AsyncPostgresSaver   # async ctx mgr
    setup_v2_checkpointer(checkpoint_dsn) -> None
    check_v2_checkpointer(checkpoint_dsn) -> CheckpointCheck

Constraints (Phase 1 plan):

- The saver is **never** instantiated and the schema is **never** migrated at
  module import. ``setup_v2_checkpointer`` is the only code path that creates
  or migrates the saver's tables (``checkpoints``, ``checkpoint_blobs``,
  ``checkpoint_writes``, ``checkpoint_migrations``); it is idempotent.
- ``check_v2_checkpointer`` is read-only: it opens a plain psycopg connection
  and reports which saver tables exist without creating, altering, or migrating
  anything.
- The DSN is validated before any connection is opened, so the asyncpg
  application DSN fails fast instead of surfacing as a psycopg connection error.

CLI (module invocation):

    python -m app.services.agents.v2.persistence.checkpoint --setup [--dsn DSN]
    python -m app.services.agents.v2.persistence.checkpoint --check [--dsn DSN]

``--dsn`` defaults to ``settings.CHECKPOINT_DATABASE_URL``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings, validate_checkpoint_dsn

#: The four tables ``AsyncPostgresSaver.setup()`` creates.
CHECKPOINT_TABLES: tuple[str, ...] = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
)


def validate_psycopg_dsn(checkpoint_dsn: str) -> None:
    """Raise ``ValueError`` unless ``checkpoint_dsn`` is a usable psycopg DSN.

    Rejects the asyncpg application DSN, an empty value, absent credentials,
    and direct reuse of ``settings.DATABASE_URL`` (the rule itself lives in
    ``app.core.config.validate_checkpoint_dsn`` so the ``Settings`` validator and
    this runtime guard cannot drift).
    """
    validate_checkpoint_dsn(checkpoint_dsn, database_url=settings.DATABASE_URL)


@asynccontextmanager
async def create_v2_checkpointer(checkpoint_dsn: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield an open ``AsyncPostgresSaver`` for the validated DSN.

    No tables are created or migrated here — the caller runs ``saver.setup()``
    (via ``setup_v2_checkpointer``) before the saver is first used.
    """
    validate_psycopg_dsn(checkpoint_dsn)
    async with AsyncPostgresSaver.from_conn_string(checkpoint_dsn) as saver:
        yield saver


async def setup_v2_checkpointer(checkpoint_dsn: str) -> None:
    """Create or migrate the saver tables (idempotent).

    Deliberately calls ``saver.setup()`` per ``AsyncPostgresSaver`` API; it is
    the only migration entrypoint in this module.
    """
    async with create_v2_checkpointer(checkpoint_dsn) as saver:
        await saver.setup()


@dataclass(frozen=True)
class CheckpointCheck:
    """Outcome of ``check_v2_checkpointer`` — never raises."""

    present: frozenset[str]
    missing: frozenset[str]

    @property
    def is_ready(self) -> bool:
        return not self.missing


async def check_v2_checkpointer(checkpoint_dsn: str) -> CheckpointCheck:
    """Report which saver tables exist, without mutating the database.

    Uses a read-only ``information_schema`` query on a plain psycopg connection;
    unlike ``setup_v2_checkpointer`` it never instantiates the saver or runs a
    migration.
    """
    validate_psycopg_dsn(checkpoint_dsn)
    async with await psycopg.AsyncConnection.connect(checkpoint_dsn, autocommit=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                (list(CHECKPOINT_TABLES),),
            )
            rows = await cur.fetchall()
    present = frozenset(row[0] for row in rows) & set(CHECKPOINT_TABLES)
    return CheckpointCheck(
        present=present,
        missing=frozenset(CHECKPOINT_TABLES) - present,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.services.agents.v2.persistence.checkpoint",
        description="Set up or inspect the async PostgreSQL v2 checkpointer.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--setup",
        action="store_true",
        help="Create/upgrade the saver tables (idempotent).",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Report whether the saver tables exist (read-only).",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="psycopg postgresql:// DSN (default: settings.CHECKPOINT_DATABASE_URL).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    dsn = args.dsn or settings.CHECKPOINT_DATABASE_URL

    try:
        if args.setup:
            asyncio.run(setup_v2_checkpointer(dsn))
        check = asyncio.run(check_v2_checkpointer(dsn))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"ready={check.is_ready} present={sorted(check.present)} "
        f"missing={sorted(check.missing)}"
    )
    return 0 if check.is_ready else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
