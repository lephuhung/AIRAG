#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# HRAG — Development Mode (Backend API + All Workers)
# ═══════════════════════════════════════════════════════════════════════════════
# Starts uvicorn (API server) + 4 worker processes in one terminal.
# Workers auto-restart on crash (up to 10 times each, with 3s backoff).
#
# Prerequisites: docker-compose.services.yml must be running
#   docker compose -f docker-compose.services.yml up -d
#
# Usage:
#   ./run_dev.sh            # Start everything (API + workers)
#   ./run_dev.sh --no-workers  # API only (same as run_bk.sh)
#   ./run_dev.sh --workers-only # Workers only (API already running)
#
# Press Ctrl+C to stop all processes.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/backend"

# ── Parse flags ───────────────────────────────────────────────────────────────
START_API=true
START_WORKERS=true
for arg in "$@"; do
    case "$arg" in
        --no-workers)   START_WORKERS=false ;;
        --workers-only) START_API=false ;;
        *)              echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "ERROR: venv not found at $SCRIPT_DIR/.venv"
    echo "  Run: cd $SCRIPT_DIR && python3 -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt"
    exit 1
fi

# ── GPU assignment ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# ── Colours for log prefixes ─────────────────────────────────────────────────
C_RESET=$'\033[0m'
C_API=$'\033[1;32m'      # green
C_PARSE=$'\033[1;34m'    # blue
C_EMBED=$'\033[1;35m'    # magenta
C_CAPTION=$'\033[1;33m'  # yellow
C_KG=$'\033[1;36m'       # cyan
C_ERR=$'\033[1;31m'      # red

# ── Kill old processes ────────────────────────────────────────────────────────
echo -e "${C_ERR}[run_dev] Killing old processes (API and workers)...${C_RESET}"
lsof -t -i:8080 | xargs -r kill -9 2>/dev/null || true
pkill -f "python -m app.workers.runner" 2>/dev/null || true

# ── Track child PIDs for cleanup ──────────────────────────────────────────────
CHILD_PIDS=()

cleanup() {
    echo ""
    echo -e "${C_ERR}[run_dev] Stopping all processes...${C_RESET}"
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    # Wait briefly, then force-kill stragglers
    sleep 1
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    wait 2>/dev/null
    echo -e "${C_ERR}[run_dev] All processes stopped.${C_RESET}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ── Worker launcher with auto-restart ─────────────────────────────────────────
MAX_RESTARTS=10
RESTART_DELAY=3

run_worker() {
    local worker_type="$1"
    local color="$2"
    local label="[worker-${worker_type}]"
    local restarts=0

    while [ $restarts -lt $MAX_RESTARTS ]; do
        echo -e "${color}${label} Starting (attempt $((restarts + 1))/${MAX_RESTARTS})${C_RESET}"
        WORKER_TYPE="$worker_type" python -m app.workers.runner 2>&1 | \
            sed -u "s/^/${color}${label} ${C_RESET}/" &
        local worker_pid=$!

        # Wait for the worker to exit
        wait "$worker_pid" 2>/dev/null
        local exit_code=$?

        # If exit code is 0 or killed by signal (>128), don't restart
        if [ $exit_code -eq 0 ]; then
            echo -e "${color}${label} Exited cleanly.${C_RESET}"
            return 0
        fi

        restarts=$((restarts + 1))
        if [ $restarts -lt $MAX_RESTARTS ]; then
            echo -e "${C_ERR}${label} Crashed (exit=$exit_code). Restarting in ${RESTART_DELAY}s... (${restarts}/${MAX_RESTARTS})${C_RESET}"
            sleep $RESTART_DELAY
        else
            echo -e "${C_ERR}${label} Max restarts reached ($MAX_RESTARTS). Giving up.${C_RESET}"
        fi
    done
}

# ── Check infrastructure services ─────────────────────────────────────────────
echo -e "${C_API}[run_dev] Checking infrastructure services...${C_RESET}"

check_service() {
    local name="$1"
    local host="$2"
    local port="$3"
    if timeout 2 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "  ✓ ${name} (${host}:${port})"
        return 0
    else
        echo -e "  ${C_ERR}✗ ${name} (${host}:${port}) — not reachable${C_RESET}"
        return 1
    fi
}

INFRA_OK=true
check_service "PostgreSQL" localhost 5433 || INFRA_OK=false
check_service "RabbitMQ"   localhost 5672 || INFRA_OK=false
check_service "MinIO"      localhost 9000 || INFRA_OK=false
check_service "ChromaDB"   localhost 8002 || INFRA_OK=false

if [ "$INFRA_OK" = false ]; then
    echo ""
    echo -e "${C_ERR}[run_dev] Some infrastructure services are not running!${C_RESET}"
    echo "  Start them with: docker compose -f docker-compose.services.yml up -d"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# ── Start API server ─────────────────────────────────────────────────────────
if [ "$START_API" = true ]; then
    echo -e "${C_API}[run_dev] Starting API server on port 8080...${C_RESET}"
    uvicorn app.main:app --reload --port 8080 2>&1 | \
        sed -u "s/^/${C_API}[api] ${C_RESET}/" &
    CHILD_PIDS+=($!)
    sleep 2  # Let API start before workers
fi

# ── Start workers ─────────────────────────────────────────────────────────────
if [ "$START_WORKERS" = true ]; then
    echo -e "${C_API}[run_dev] Starting workers...${C_RESET}"

    run_worker "parse"   "$C_PARSE"   &
    CHILD_PIDS+=($!)

    run_worker "embed"   "$C_EMBED"   &
    CHILD_PIDS+=($!)

    run_worker "caption" "$C_CAPTION" &
    CHILD_PIDS+=($!)

    run_worker "kg"      "$C_KG"      &
    CHILD_PIDS+=($!)

    echo -e "${C_API}[run_dev] All workers started. Press Ctrl+C to stop.${C_RESET}"
fi

echo ""
echo -e "${C_API}═══════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_API}  HRAG Dev Mode Running${C_RESET}"
if [ "$START_API" = true ]; then
    echo -e "${C_API}  API:     http://localhost:8080${C_RESET}"
    echo -e "${C_API}  Docs:    http://localhost:8080/docs${C_RESET}"
fi
if [ "$START_WORKERS" = true ]; then
    echo -e "${C_PARSE}  Workers: parse, embed, caption, kg${C_RESET}"
fi
echo -e "${C_API}═══════════════════════════════════════════════════════${C_RESET}"
echo ""

# ── Wait for all children ─────────────────────────────────────────────────────
wait
