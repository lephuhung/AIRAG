"""Shadow v2 isolated checkpointer (Phase 3, Task 6).

Owns the shadow run's saver bundle: a fresh, process-local saver that is
structurally incapable of touching production checkpoint state.

Global constraints honored here:

- The shadow saver is NEVER the production saver: this module does not
  import ``v2.persistence.checkpoint`` (the production
  ``AsyncPostgresSaver`` factory) on any path. ``InMemorySaver`` is the
  isolated saver; no Postgres schema — temporary or otherwise — is opened.
- Frozen contract types are imported, never redefined (this module defines
  no contract model at all).
- Only nodes and atomic capabilities may be created elsewhere; this module
  creates no graph, agent, or subgraph — only the saver bundle the shadow
  graph compiles against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

__all__ = [
    "ShadowCheckpointBundle",
    "create_shadow_checkpointer",
    "is_shadow_saver",
]


def create_shadow_checkpointer() -> InMemorySaver:
    """Create a fresh isolated saver for exactly one shadow run.

    A new ``InMemorySaver`` per call means per-run checkpoint namespaces
    can never observe another run's checkpoints — shadow or production.
    No DSN, no connection, no migration: there is no production saver to
    construct on this path by construction (the production factory module
    is not imported here).
    """
    saver = InMemorySaver()
    # Brand the instance so the isolation proof can distinguish it from any
    # production saver without reaching into saver internals.
    saver.__v2_shadow_saver__ = True  # type: ignore[attr-defined]
    return saver


def is_shadow_saver(saver: Any) -> bool:
    """True only for savers minted by :func:`create_shadow_checkpointer`."""
    return bool(getattr(saver, "__v2_shadow_saver__", False))


@dataclass
class ShadowCheckpointBundle:
    """The isolated saver bundle one shadow run compiles against.

    ``namespace`` scopes the run's thread ids so two concurrent shadow runs
    cannot read each other's checkpoints even though both are in-memory.
    The bundle owns no production handle of any kind.
    """

    saver: InMemorySaver
    namespace: str
    _production_rows_ref: Any = field(default=None, repr=False)

    @property
    def production_rows(self) -> Any:
        """The production-row mapping under test (read handle only)."""
        return self._production_rows_ref
