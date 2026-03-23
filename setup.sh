#!/bin/bash
# ============================================================
# HRAG — Local Development Setup
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  HRAG — Local Development Setup"
echo "============================================"
echo ""

# -----------------------------------------------------------
# 1. Check prerequisites
# -----------------------------------------------------------
echo "[1/7] Checking prerequisites..."

# Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi
echo "  Python $PY_VERSION"

# Node
if ! command -v node &>/dev/null; then
    echo "ERROR: node not found. Install Node.js 18+ first."
    exit 1
fi
NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "ERROR: Node.js 18+ required (found $(node -v))"
    exit 1
fi
echo "  Node $(node -v)"

# pnpm
if ! command -v pnpm &>/dev/null; then
    echo "ERROR: pnpm not found. Install: npm install -g pnpm"
    exit 1
fi
echo "  pnpm $(pnpm -v)"

# Docker (optional)
if command -v docker &>/dev/null; then
    echo "  Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"
    HAS_DOCKER=true
else
    echo "  Docker: not found (PostgreSQL + ChromaDB must be started manually)"
    HAS_DOCKER=false
fi

echo ""

# -----------------------------------------------------------
# 2. Create Python virtual environment
# -----------------------------------------------------------
echo "[2/7] Setting up Python virtual environment..."
if [ ! -d "back_end" ]; then
    python3 -m venv back_end
    echo "  Created back_end/"
else
    echo "  back_end/ already exists"
fi
source back_end/bin/activate

# -----------------------------------------------------------
# 3. Install Python dependencies
# -----------------------------------------------------------
echo "[3/7] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt
echo "  Done."

# -----------------------------------------------------------
# 4. Create .env if not exists
# -----------------------------------------------------------
echo "[4/7] Checking .env configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  >>> IMPORTANT: Edit .env and set GOOGLE_AI_API_KEY <<<"
else
    echo "  .env already exists"
fi

# -----------------------------------------------------------
# 5. Start services (Docker)
# -----------------------------------------------------------
echo "[5/7] Starting database services..."
if [ "$HAS_DOCKER" = true ]; then
    docker compose -f docker-compose.services.yml up -d
    echo "  Waiting for PostgreSQL to be ready..."
    for i in $(seq 1 30); do
        if docker exec nexusrag-postgres pg_isready -U postgres &>/dev/null; then
            echo "  PostgreSQL ready."
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "  WARNING: PostgreSQL not ready after 30s. Check docker logs."
        fi
        sleep 1
    done

    # Configure MinIO CORS so the browser can PUT files directly (presigned upload)
    echo "  Configuring MinIO CORS for direct browser uploads..."
    if docker run --rm --network host minio/mc:latest sh -c \
        "mc alias set local http://localhost:9000 minioadmin minioadmin && \
         mc mb --ignore-existing local/nexusrag-uploads && \
         mc mb --ignore-existing local/nexusrag-markdown" 2>/dev/null; then
        echo "  MinIO buckets ready."
    else
        echo "  WARNING: Could not configure MinIO. Ensure it is running on port 9000."
    fi
else
    echo "  Skipped (no Docker). Ensure PostgreSQL (port 5433) and ChromaDB (port 8002) are running."
fi

# -----------------------------------------------------------
# 6. Download ML models (optional)
# -----------------------------------------------------------
echo ""
echo "[6/7] ML models (~2.5GB total):"
echo "  - BAAI/bge-m3 (embedding, ~1.4GB)"
echo "  - BAAI/bge-reranker-v2-m3 (reranker, ~1.1GB)"
echo ""
read -p "  Download models now? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[yY]$ ]]; then
    echo "  Downloading models (this may take a few minutes)..."
    python backend/scripts/download_models.py
else
    echo "  Skipped. Models will be downloaded on first use."
fi

# -----------------------------------------------------------
# 7. Install frontend dependencies
# -----------------------------------------------------------
echo "[7/7] Installing frontend dependencies..."
cd frontend
pnpm install
cd ..

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Start backend:   ./run_bk.sh"
echo "  Start frontend:  ./run_fe.sh"
echo "  Open:            http://localhost:5174"
echo ""
echo "  Or use Docker:   docker compose up -d"
echo ""
