"""
System Logs API — Real-time Log Streaming
==========================================

Provides SSE streaming endpoint for real-time log viewing.
Supports tailing multiple log files simultaneously.

Route: GET /logs/stream?files=backend.log,worker_parse.log
       GET /logs/list — list available log files
       GET /logs/{filename} — get last N lines of a specific log
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.deps import get_current_active_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["logs"])

# Log directory - use absolute path based on backend/app/api directory
import os as _os
_backend_dir = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
LOG_DIR = Path(_backend_dir) / "logs"

AVAILABLE_LOG_FILES = [
    "backend.log",
    "backend_restart.log",
    "worker_parse.log",
    "worker_embed.log",
    "worker_caption.log",
    "worker_kg.log",
    "workers.log",
    "workers_restart.log",
    "ocr_vllm.log",
    "qwen_vllm.log",
]


def _format_sse(event: str, data: dict) -> str:
    """Format data as SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _tail_file(filepath: Path, interval: float = 0.5) -> AsyncGenerator[str, None]:
    """Yield new lines as they are appended to a file."""
    if not filepath.exists():
        return

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        # Seek to end to only get new content
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield line.rstrip("\n")
            else:
                await asyncio.sleep(interval)


@router.get("/stream")
async def stream_logs(
    files: str = Query(
        default="backend.log",
        description="Comma-separated list of log files to stream"
    ),
    current_user: User = Depends(get_current_active_user),
):
    """
    SSE endpoint that streams log file updates in real-time.
    Supports multiple log files simultaneously.
    """
    if not current_user.is_superadmin:
        raise HTTPException(status_code=403, detail="Admin access required")

    file_list = [f.strip() for f in files.split(",") if f.strip()]

    if not file_list:
        raise HTTPException(status_code=400, detail="At least one log file must be specified")

    # Validate files
    for filename in file_list:
        if filename not in AVAILABLE_LOG_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown log file: {filename}. Available: {', '.join(AVAILABLE_LOG_FILES)}"
            )

    async def event_generator():
        """Generate SSE events with log lines."""
        # Track file positions for each file
        file_positions = {}
        buffers = {f: [] for f in file_list}

        try:
            while True:
                for filename in file_list:
                    filepath = LOG_DIR / filename
                    if not filepath.exists():
                        continue

                    try:
                        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                            # Get last position
                            f.seek(file_positions.get(filename, 0))
                            new_lines = f.readlines()
                            if new_lines:
                                file_positions[filename] = f.tell()
                                for line in new_lines:
                                    line = line.rstrip("\n")
                                    if line:
                                        # Determine log level for styling
                                        log_level = "info"
                                        if "[ERROR]" in line or "[CRITICAL]" in line:
                                            log_level = "error"
                                        elif "[WARNING]" in line or "[WARN]" in line:
                                            log_level = "warning"
                                        elif "[DEBUG]" in line:
                                            log_level = "debug"

                                        yield _format_sse("log_line", {
                                            "filename": filename,
                                            "line": line,
                                            "level": log_level
                                        })
                    except Exception as e:
                        yield _format_sse("error", {"message": f"Error reading {filename}: {str(e)}"})

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info("Log stream cancelled for user %s", current_user.id)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/list")
async def list_log_files(
    current_user: User = Depends(get_current_active_user),
):
    """List all available log files."""
    if not current_user.is_superadmin:
        raise HTTPException(status_code=403, detail="Admin access required")

    logger.info(f"LOG_DIR = {LOG_DIR} (exists: {LOG_DIR.exists()})")

    files_info = []
    for filename in AVAILABLE_LOG_FILES:
        filepath = LOG_DIR / filename
        logger.info(f"Checking {filepath} (exists: {filepath.exists()})")
        if filepath.exists():
            stat = filepath.stat()
            files_info.append({
                "name": filename,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        else:
            files_info.append({
                "name": filename,
                "size": 0,
                "modified": None,
                "exists": False,
            })

    return {"files": files_info}


@router.get("/{filename}")
async def get_log_content(
    filename: str,
    lines: int = Query(default=500, ge=1, le=5000, description="Number of lines to retrieve"),
    current_user: User = Depends(get_current_active_user),
):
    """Get last N lines of a specific log file."""
    if not current_user.is_superadmin:
        raise HTTPException(status_code=403, detail="Admin access required")

    if filename not in AVAILABLE_LOG_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown log file: {filename}. Available: {', '.join(AVAILABLE_LOG_FILES)}"
        )

    filepath = LOG_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        start_idx = max(0, total_lines - lines)

        return {
            "filename": filename,
            "total_lines": total_lines,
            "lines": all_lines[start_idx:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")