#!/bin/bash
# HRAG Workers — start all 4 worker processes for local dev
#
# Workers:
#   parse   — downloads from MinIO, runs Docling parser, publishes to embed/caption/kg
#   embed   — generates BGE embeddings → ChromaDB
#   caption — generates image captions
#   kg      — LightRAG entity/relationship extraction
#
# Usage:
#   ./run_workers.sh                      # start all 4 workers
#   ./run_workers.sh parse embed          # start specific workers
#   WORKERS="parse embed" ./run_workers.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
LOG_DIR="$SCRIPT_DIR/logs"

# Activate venv
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "ERROR: venv not found at $SCRIPT_DIR/.venv"
    echo "Run ./setup.sh first."
    exit 1
fi

# Which workers to start (default: all)
WORKERS="${WORKERS:-${*:-parse embed caption kg}}"

mkdir -p "$LOG_DIR"

echo "============================================"
echo " HRAG Worker Launcher"
echo " Workers: $WORKERS"
echo " Logs: $LOG_DIR/worker_<type>.log"
echo " Press Ctrl+C to stop all workers"
echo "============================================"
echo ""

PIDS=()

cleanup() {
    echo ""
    echo "Stopping all workers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Wait for all workers to finish
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "All workers stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

for WORKER in $WORKERS; do
    LOG_FILE="$LOG_DIR/worker_${WORKER}.log"
    echo "  [${WORKER}] Starting → $LOG_FILE"
    # Run from backend dir so config.py finds the .env (walks up to AIRAG/.env)
    # NOTE: Do NOT export .env via bash — pydantic-settings handles JSON fields correctly
    (cd "$BACKEND_DIR" && CUDA_VISIBLE_DEVICES=1 WORKER_TYPE="$WORKER" python -m app.workers.runner) \
        >> "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Workers running. PIDs: ${PIDS[*]}"
echo ""
echo "Tailing all logs (Ctrl+C stops everything)..."
echo "---"

# Tail all logs (label each line with worker name)
tail -F "$LOG_DIR"/worker_*.log 2>/dev/null &
TAIL_PID=$!

# Wait for workers — if any exits unexpectedly, report it
wait "${PIDS[@]}"
kill "$TAIL_PID" 2>/dev/null || true
