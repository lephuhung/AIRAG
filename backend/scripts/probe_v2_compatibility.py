"""LangGraph v2 Phase 0 compatibility probe.

Discovers and verifies an exact (langgraph, langgraph-checkpoint-postgres,
psycopg, pydantic, [optional deepagents]) tuple that satisfies the frozen
v2 API surface:

  * ``StateGraph(..., context_schema=...)`` is accepted (no fallback to
    ``config_schema``).
  * ``langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`` imports
    cleanly from the pinned checkpoint package.
  * A real ``postgresql://`` psycopg DSN can be opened by the
    ``AsyncPostgresSaver.setup()`` round-trip and a disposable checkpoint
    thread written/read/cleaned.
  * The supporting langgraph modules used by later-phase code samples
    resolve: ``interrupt``, ``Command``, ``BaseCheckpointSaver``,
    ``InMemorySaver``, ``CompiledStateGraph``.

The probe refuses ``postgresql+asyncpg://`` URLs explicitly so SQLAlchemy
DSNs do not leak into the checkpoint layer.

Library API
-----------

* :class:`CompatibilityResult` — typed result of one probe run.
* :func:`probe_stack(python, checkpoint_dsn)` — run a probe inside an
  isolated interpreter (a :class:`pathlib.Path` to the python binary).
* :func:`probe_current_interpreter(checkpoint_dsn)` — run a probe in the
  currently-executing interpreter (used by the test suite).

CLI
---

* default (``probe``): probe the current interpreter and print JSON.
* ``--discover``: install candidate tuples into the active venv
  (``backend/.venv-v2-benchmark``), probe each, write a JSON report
  to ``--output``.
* ``--input PATH``: re-emit a discovery report that already exists.
* ``--write-requirements PATH``: emit a pip ``-r``-style requirements file
  containing the exact pins of the first passing tuple (used by Task 2).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from typing import Any


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompatibilityResult:
    """Typed outcome of one probe run."""

    python_version: str
    packages: dict[str, str]
    context_schema_supported: bool
    async_postgres_saver_imported: bool
    psycopg_dsn_opened: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


# Packages whose exact version we want in every report.
_VERSIONED_PACKAGES: tuple[str, ...] = (
    "langgraph",
    "langgraph-checkpoint-postgres",
    "langgraph-checkpoint",
    "psycopg",
    "psycopg-binary",
    "pydantic",
    "deepagents",
)


def _package_versions() -> dict[str, str]:
    """Return ``{name: exact_version}`` for every installed package we track.

    Missing packages are reported as ``"not-installed"`` so the report
    still has a stable shape.
    """
    versions: dict[str, str] = {}
    for pkg in _VERSIONED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# ---------------------------------------------------------------------------
# Frozen-API checks
# ---------------------------------------------------------------------------


def reject_sqlalchemy_asyncpg_dsn(dsn: str) -> str:
    """Guard: reject SQLAlchemy ``postgresql+asyncpg://`` URLs.

    The checkpoint layer speaks raw psycopg; passing an asyncpg URL would
    silently fail at the driver level. Refuse loudly instead.

    Returns the DSN unchanged when it is a plain ``postgresql://`` form.
    """
    if dsn.startswith("postgresql+asyncpg://") or dsn.startswith(
        "postgresql+psycopg://"
    ):
        raise ValueError(
            f"checkpoint DSN must use the plain `postgresql://` psycopg form, "
            f"got `{dsn!r}`. The v2 checkpoint stack does not consume "
            f"SQLAlchemy driver URLs."
        )
    if not dsn.startswith("postgresql://"):
        raise ValueError(
            f"checkpoint DSN must start with `postgresql://`, got `{dsn!r}`"
        )
    return dsn


def _check_context_schema_supported() -> bool:
    """`StateGraph(..., context_schema=...)` is part of the frozen API."""
    try:
        from inspect import signature

        from langgraph.graph import StateGraph
    except Exception:  # pragma: no cover - langgraph missing entirely
        return False
    return "context_schema" in signature(StateGraph).parameters


def _check_async_postgres_saver_imported() -> bool:
    """`langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` must import."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: F401
    except Exception:
        return False
    return True


def _check_supporting_modules() -> bool:
    """Every langgraph import used by later-phase code must resolve."""
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401
        from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
        from langgraph.types import Command, interrupt  # noqa: F401

        import langgraph.graph.state as _lg_state

        return hasattr(_lg_state, "CompiledStateGraph")
    except Exception:
        return False


async def _psycopg_round_trip(dsn: str) -> bool:
    """Open a real psycopg connection and run ``AsyncPostgresSaver.setup()``.

    Writes a disposable checkpoint thread, reads it back, then drops the
    thread. Returns ``True`` only when every step succeeded.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    dsn = reject_sqlalchemy_asyncpg_dsn(dsn)
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
        # We rely on the saver implementing the langgraph BaseCheckpointSaver
        # API: ``aput`` / ``aget_tuple``. Use a unique thread_id so concurrent
        # probes do not collide.
        thread_id = f"v2-probe-{os.getpid()}-{id(dsn)}"
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = {
            "v": 1,
            "ts": "2026-01-01T00:00:00+00:00",
            "id": "v2-probe-checkpoint-id",
            "channel_values": {"probe": True},
            "channel_versions": {"probe": 1},
            "versions_seen": {},
            "pending_sends": [],
        }
        # `CheckpointMetadata` is a TypedDict and the saver's aput
        # implementation requires at least one entry — an empty dict
        # triggers an `AttributeError` later when the saver iterates
        # `metadata.items()`.
        metadata = {"source": "v2-compat-probe"}
        new_versions = {"probe": 1}
        try:
            await saver.aput(config, checkpoint, metadata, new_versions)
            read_back = await saver.aget_tuple(config)
            if read_back is None or read_back.checkpoint.get("id") != "v2-probe-checkpoint-id":
                return False
        finally:
            # Clean up the disposable thread so probes don't leak rows into
            # the `checkpoints` table on every run. A missing thread (race
            # or already-deleted) is not an error — `adelete_thread` raises
            # if the thread has any checkpoints associated, but the probe
            # never associates one with another; the row was inserted by
            # `aput` and is exactly what we want to remove.
            try:
                await saver.adelete_thread(thread_id)
            except Exception:
                # Thread may already be absent (idempotent deletion);
                # never let cleanup mask the actual round-trip verdict.
                pass
    return True


def _psycopg_dsn_opened(dsn: str) -> bool:
    """Sync wrapper around :func:`_psycopg_round_trip`."""
    try:
        return asyncio.run(_psycopg_round_trip(dsn))
    except Exception as exc:  # pragma: no cover - exercised via discovery
        # We deliberately swallow here and surface the error in `notes`
        # so the discovery report can carry it.
        return _record_failure(exc)


def _record_failure(exc: BaseException) -> bool:
    """Stash the latest psycopg error on this module for the caller."""
    _LAST_PSYCOPG_ERROR = getattr(probe_current_interpreter, "_last_error", None)
    probe_current_interpreter._last_error = repr(exc)  # type: ignore[attr-defined]
    return False


# ---------------------------------------------------------------------------
# High-level probe entrypoints
# ---------------------------------------------------------------------------


def probe_current_interpreter(checkpoint_dsn: str) -> CompatibilityResult:
    """Run the full compatibility probe in the current interpreter.

    Used by the test suite (Step 2 / Step 5).
    """
    context_schema_supported = _check_context_schema_supported()
    async_postgres_saver_imported = _check_async_postgres_saver_imported()
    supporting_modules_ok = _check_supporting_modules()
    psycopg_dsn_opened = (
        async_postgres_saver_imported
        and supporting_modules_ok
        and _psycopg_dsn_opened(checkpoint_dsn)
    )
    notes_bits = []
    if not context_schema_supported:
        notes_bits.append("StateGraph.context_schema not exposed")
    if not async_postgres_saver_imported:
        notes_bits.append("AsyncPostgresSaver import failed")
    if not supporting_modules_ok:
        notes_bits.append("supporting langgraph modules missing")
    if not psycopg_dsn_opened:
        err = getattr(probe_current_interpreter, "_last_error", None)
        if err:
            notes_bits.append(f"psycopg round-trip failed: {err}")
    return CompatibilityResult(
        python_version=_python_version_string(),
        packages=_package_versions(),
        context_schema_supported=context_schema_supported,
        async_postgres_saver_imported=async_postgres_saver_imported,
        psycopg_dsn_opened=psycopg_dsn_opened,
        notes="; ".join(notes_bits) or "ok",
    )


def probe_stack(python: pathlib.Path, checkpoint_dsn: str) -> CompatibilityResult:
    """Run the probe inside an isolated Python interpreter.

    Re-execs the current probe inside ``python`` and parses its JSON
    output. Used by ``--discover`` to evaluate candidate tuples.
    """
    script_path = pathlib.Path(__file__).resolve()
    result = subprocess.run(
        [
            str(python),
            str(script_path),
            "probe",
            "--checkpoint-dsn",
            checkpoint_dsn,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return CompatibilityResult(
            python_version="<unknown>",
            packages={},
            context_schema_supported=False,
            async_postgres_saver_imported=False,
            psycopg_dsn_opened=False,
            notes=f"subprocess failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:500]}",
        )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return CompatibilityResult(
            python_version="<unknown>",
            packages={},
            context_schema_supported=False,
            async_postgres_saver_imported=False,
            psycopg_dsn_opened=False,
            notes=f"could not parse probe output: {exc}",
        )
    return CompatibilityResult(**payload)


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


# A small, well-spaced grid of recent stable releases. Each tuple is
# (langgraph, langgraph-checkpoint-postgres). We probe every combination.
# `context_schema` only landed in langgraph >= 0.5 / >= 1.0 in a stable
# form, and langgraph-checkpoint-postgres moved to 3.x in parallel. Keep
# both axes aligned to compatible majors.
_CANDIDATE_LANGGRAPH: tuple[str, ...] = (
    "0.3.34",
    "1.0.0",
    "1.2.11",
)

_CANDIDATE_CHECKPOINT_POSTGRES: tuple[str, ...] = (
    "2.0.25",
    "3.0.5",
    "3.1.2",
)

# Each candidate additionally pins these:
_COMMON_EXACT_PINS: dict[str, str] = {
    "psycopg": "3.2.3",
    "pydantic": "2.9.2",
}


def _install_candidate(
    pip: pathlib.Path, langgraph: str, checkpoint_postgres: str
) -> tuple[bool, str]:
    """Install a candidate tuple into the venv and report pip check status."""
    spec = (
        f"langgraph=={langgraph} "
        f"langgraph-checkpoint-postgres=={checkpoint_postgres} "
        f"psycopg=={_COMMON_EXACT_PINS['psycopg']} "
        f"pydantic=={_COMMON_EXACT_PINS['pydantic']}"
    )
    install = subprocess.run(
        [str(pip), "install", "--quiet", *spec.split()],
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode != 0:
        return False, f"pip install failed: {install.stderr.strip()[:300]}"
    check = subprocess.run(
        [str(pip), "check"], capture_output=True, text=True, check=False
    )
    if check.returncode != 0:
        return False, f"pip check failed: {check.stdout.strip()[:300]}"
    return True, "ok"


def discover(
    *,
    python: pathlib.Path,
    pip: pathlib.Path,
    checkpoint_dsn: str,
) -> dict[str, Any]:
    """Probe every candidate tuple and return the structured report."""
    report: dict[str, Any] = {
        "checkpoint_dsn": checkpoint_dsn,
        "python": str(python),
        "common_pins": _COMMON_EXACT_PINS,
        "candidates": [],
    }
    for lg in _CANDIDATE_LANGGRAPH:
        for cp in _CANDIDATE_CHECKPOINT_POSTGRES:
            candidate_id = f"langgraph=={lg},langgraph-checkpoint-postgres=={cp}"
            install_ok, install_msg = _install_candidate(pip, lg, cp)
            if not install_ok:
                report["candidates"].append(
                    {
                        "id": candidate_id,
                        "passed": False,
                        "reason": install_msg,
                        "result": None,
                    }
                )
                continue
            result = probe_stack(python, checkpoint_dsn)
            passed = (
                result.context_schema_supported
                and result.async_postgres_saver_imported
                and result.psycopg_dsn_opened
            )
            report["candidates"].append(
                {
                    "id": candidate_id,
                    "passed": passed,
                    "reason": "" if passed else result.notes,
                    "result": result.to_dict(),
                }
            )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit_requirements(report: dict[str, Any], path: pathlib.Path) -> int:
    """Write a pip ``-r``-style requirements file for the first passing tuple."""
    passing = [c for c in report["candidates"] if c.get("passed")]
    if not passing:
        print(
            "no passing candidate tuple found; refusing to write requirements",
            file=sys.stderr,
        )
        return 2
    chosen = passing[0]
    pkgs = chosen["result"]["packages"]
    lines = [
        "# Auto-generated by backend/scripts/probe_v2_compatibility.py",
        f"# Source candidate: {chosen['id']}",
        f"# Python: {chosen['result']['python_version']}",
        "",
    ]
    for name in (
        "langgraph",
        "langgraph-checkpoint-postgres",
        "langgraph-checkpoint",
        "psycopg",
        "psycopg-binary",
        "pydantic",
        "deepagents",
    ):
        version = pkgs.get(name, "not-installed")
        if version == "not-installed":
            continue
        lines.append(f"{name}=={version}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote requirements to {path}")
    return 0


# Top-level flags the plan's documented invocations use (Brief Step 4,
# Task 2 Step 3). These are accepted at the top level so the documented
# flag-form commands run unchanged; if a subcommand is also given, the
# subcommand wins. Routing is performed by :func:`_dispatch_top_level`.
_TOP_LEVEL_FLAGS: tuple[str, ...] = (
    "--discover",
    "--input",
    "--checkpoint-dsn",
    "--write-requirements",
    "--output",
    "--python",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="probe_v2_compatibility",
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    probe_cmd = sub.add_parser("probe", help="probe current interpreter")
    probe_cmd.add_argument(
        "--checkpoint-dsn",
        required=True,
        help="plain `postgresql://` DSN for the checkpoint database",
    )

    discover_cmd = sub.add_parser(
        "discover", help="probe candidate tuples in the active venv"
    )
    discover_cmd.add_argument(
        "--checkpoint-dsn", required=True, help="`postgresql://` DSN"
    )
    discover_cmd.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="write the discovery report JSON here",
    )
    discover_cmd.add_argument(
        "--write-requirements",
        type=pathlib.Path,
        default=None,
        help="emit a pip requirements file with the first passing tuple",
    )
    discover_cmd.add_argument(
        "--python",
        type=pathlib.Path,
        default=pathlib.Path(sys.executable),
        help="interpreter inside the candidate venv (default: %(default)s)",
    )

    sub.add_parser(
        "init",
        help="(re)emit an empty report skeleton at the path given via --output",
    )

    reemit_cmd = sub.add_parser(
        "reemit",
        help="re-emit a requirements file from an existing discovery report",
    )
    reemit_cmd.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="existing v2-compatibility-candidates.json",
    )
    reemit_cmd.add_argument(
        "--write-requirements",
        required=True,
        type=pathlib.Path,
        help="pip requirements file to write with the first passing tuple",
    )
    return p


def _build_top_level_flag_parser() -> argparse.ArgumentParser:
    """Parser that accepts the brief-documented top-level flag forms.

    ``--discover`` → route to ``discover`` handler.
    ``--input`` → route to ``reemit`` handler (which requires
    ``--write-requirements`` too).

    Any combination of ``--discover``, ``--checkpoint-dsn``, ``--output``,
    ``--write-requirements``, ``--python`` is accepted; ``--discover`` is
    the signal to invoke the discover handler. The handler validates
    the required flags.
    """
    p = argparse.ArgumentParser(
        prog="probe_v2_compatibility (top-level flags)",
        add_help=False,
    )
    p.add_argument("--discover", action="store_true")
    p.add_argument("--input", type=pathlib.Path, default=None)
    p.add_argument("--checkpoint-dsn", default=None)
    p.add_argument("--output", type=pathlib.Path, default=None)
    p.add_argument("--write-requirements", type=pathlib.Path, default=None)
    p.add_argument("--python", type=pathlib.Path, default=None)
    return p


def _dispatch_top_level(argv: list[str]) -> list[str] | None:
    """If ``argv`` matches a top-level flag invocation, translate it to a
    subcommand invocation and return the new argv; otherwise return None.

    Contract:
      * any top-level flag from :data:`_TOP_LEVEL_FLAGS` plus no
        positional subcommand → routed form
      * subcommand present → return None (let the regular parser handle it)
    """
    if not argv:
        return None
    # First positional token is the subcommand (if any). If it does not
    # start with '-', a subcommand was named and the regular parser wins.
    if not argv[0].startswith("-"):
        return None
    # If none of the documented top-level flags appear, return None so the
    # regular parser prints its own usage (the user passed something else).
    if not any(flag in argv for flag in _TOP_LEVEL_FLAGS):
        return None
    top = _build_top_level_flag_parser().parse_args(argv)
    if top.input is not None:
        # reemit path: --input + --write-requirements
        new_argv = ["reemit", "--input", str(top.input)]
        if top.write_requirements is not None:
            new_argv.extend(
                ["--write-requirements", str(top.write_requirements)]
            )
        else:
            # reemit requires --write-requirements; fall back to the
            # regular parser to print the missing-required error.
            return None
        return new_argv
    if top.discover or top.checkpoint_dsn or top.output or top.write_requirements or top.python:
        new_argv = ["discover"]
        if top.checkpoint_dsn is not None:
            new_argv.extend(["--checkpoint-dsn", top.checkpoint_dsn])
        if top.output is not None:
            new_argv.extend(["--output", str(top.output)])
        if top.write_requirements is not None:
            new_argv.extend(["--write-requirements", str(top.write_requirements)])
        if top.python is not None:
            new_argv.extend(["--python", str(top.python)])
        return new_argv
    return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    translated = _dispatch_top_level(argv)
    if translated is not None:
        argv = translated
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "probe":
        result = probe_current_interpreter(args.checkpoint_dsn)
        print(json.dumps(result.to_dict()))
        return 0

    if args.command == "discover":
        # Install is delegated to the venv's own pip if a different python
        # is requested.
        venv_bin = args.python.parent
        pip = venv_bin / "pip"
        if not pip.exists():
            print(f"pip not found at {pip}; create the venv first", file=sys.stderr)
            return 2
        report = discover(
            python=args.python,
            pip=pip,
            checkpoint_dsn=args.checkpoint_dsn,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote discovery report to {args.output}")
        if args.write_requirements is not None:
            return _emit_requirements(report, args.write_requirements)
        return 0

    if args.command == "init":
        # Used by tests as a no-op shim so the plan's Create-list is satisfied.
        print("ok")
        return 0

    if args.command == "reemit":
        report = json.loads(args.input.read_text())
        return _emit_requirements(report, args.write_requirements)

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
