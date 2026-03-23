#!/bin/bash
# Script to run vLLM for HunyuanOCR and Qwen3.5-0.8B as standalone OpenAI APIs
# Usage: ./run_vllm.sh

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIRTUAL_ENV="$SCRIPT_DIR/.venv"

# Check if venv exists
if [ ! -d "$VIRTUAL_ENV" ]; then
    echo "ERROR: Virtual environment not found at $VIRTUAL_ENV"
    echo "Please ensure you have a .venv in the project root with vllm installed."
    exit 1
fi

# Specify GPU 1
export CUDA_VISIBLE_DEVICES=1

echo "----------------------------------------------------------"
echo "Cleaning up existing vLLM processes on GPU 1..."
# Kill by ports
fuser -k 8001/tcp || true
fuser -k 8002/tcp || true
fuser -k 8082/tcp || true

# Aggressive kill for any vllm-related processes (excluding this script)
echo "Killing any remaining vllm or EngineCore python processes..."
pkill -9 -f "python3 -m vllm" || true
pkill -9 -f "vLLM::EngineCore" || true

# Wait for VRAM to be freed
echo "Waiting 5 seconds for VRAM to release..."
sleep 5

# Check if any processes are still on GPU 1 (physical index 1 often corresponds to what we see in nvidia-smi)
echo "Current GPU 1 state (physical):"
nvidia-smi -i 1 || true

echo "----------------------------------------------------------"
echo "Starting vLLM Servers on GPU 1"
echo "----------------------------------------------------------"

# 1. Start HunyuanOCR (OCR Model)
# Port: 8001, GPU Memory: 0.4 (approx 19.6 GB)
echo "[1/2] Starting HunyuanOCR on port 8001..."
nohup "$VIRTUAL_ENV/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model tencent/HunyuanOCR \
    --port 8001 \
    --gpu-memory-utilization 0.4 \
    --trust-remote-code \
    --served-model-name hunyuan-ocr > ocr_vllm.log 2>&1 &

# 2. Start Qwen3-4B-Instruct-2507-FP8 (Memory Agent Model)
# Port: 8082, GPU Memory: 0.15 (approx 7.3 GB)
echo "[2/2] Starting Qwen3-4B-Instruct-2507-FP8 on port 8082..."
nohup "$VIRTUAL_ENV/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
    --port 8082 \
    --max-model-len 15312 \
    --gpu-memory-utilization 0.2 \
    --trust-remote-code \
    --served-model-name qwen-memory > qwen_vllm.log 2>&1 &

echo "----------------------------------------------------------"
echo "Both models are starting in the background."
echo "OCR (HunyuanOCR): http://localhost:8001/v1"
echo "LLM (Qwen3.5):    http://localhost:8082/v1"
echo "----------------------------------------------------------"
echo "Check logs for progress:"
echo "  tail -f ocr_vllm.log"
echo "  tail -f qwen_vllm.log"
