#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# HRAG — Development Mode (Backend API only)
# ═══════════════════════════════════════════════════════════════════════════════
# Workers run as Docker containers via docker-compose.services.yml.
# This script only starts the FastAPI backend server.
#
# Prerequisites: docker-compose.services.yml must be running
#   docker compose -f docker-compose.services.yml up -d
#
# Usage:
#   ./run_dev.sh            # Start API server
#   ./run_dev.sh --workers   # (DEPRECATED, ignored — workers run in Docker)
#
# Press Ctrl+C to stop.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/backend"

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
C_ERR=$'\033[1;31m'      # red

# ── Kill old processes ────────────────────────────────────────────────────────
echo -e "${C_ERR}[run_dev] Killing old API processes on port 8080...${C_RESET}"
lsof -t -i:8080 | xargs -r kill -9 2>/dev/null || true

# ── Track child PIDs for cleanup ────────────────────────────────────────────────
CHILD_PIDS=()

cleanup() {
    echo ""
    echo -e "${C_ERR}[run_dev] Stopping API server...${C_RESET}"
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    wait 2>/dev/null
    echo -e "${C_ERR}[run_dev] API server stopped.${C_RESET}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

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
echo -e "${C_API}[run_dev] Starting API server on port 8080...${C_RESET}"
uvicorn app.main:app --reload --port 8080 2>&1 | \
    sed -u "s/^/${C_API}[api] ${C_RESET}/" &
CHILD_PIDS+=($!)

echo ""
echo -e "${C_API}═══════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_API}  HRAG Dev Mode Running${C_RESET}"
echo -e "${C_API}  API:     http://localhost:8080${C_RESET}"
echo -e "${C_API}  Docs:    http://localhost:8080/docs${C_RESET}"
echo -e "${C_API}═══════════════════════════════════════════════════════${C_RESET}"
echo ""

# ── Wait for all children ─────────────────────────────────────────────────────
wait