#!/bin/bash
# File: clear_kg.sh
# Description: Xóa toàn bộ dữ liệu Knowledge Graph trong Neo4j và xóa các file lưu trữ cục bộ của LightRAG

echo "==============================================="
echo "   Knowledge Graph Full Cleanup Script         "
echo "==============================================="

# 1. Clear Neo4j Data
echo "[1/2] Đang xóa toàn bộ node/edge trong Neo4j..."

# Kiểm tra xem .venv có tồn tại không
if [ -f "backend/.venv/bin/activate" ]; then
    source backend/.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python3 -c "
import asyncio
import os
from pathlib import Path

# Load .env if it exists (manual parse to avoid dependency on python-dotenv in this script)
env_path = Path('.env')
env_vars = {}
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key] = value.strip('\"\'')

try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    print('Thư viện neo4j không được cài đặt. Bỏ qua bước xóa qua Python.')
    import sys
    sys.exit(0)

URI = env_vars.get('NEO4J_URI', 'bolt://localhost:7687')
USER = env_vars.get('NEO4J_USERNAME', 'neo4j')
PASSWORD = env_vars.get('NEO4J_PASSWORD', 'nexusrag123')

async def clear_db():
    try:
        async with AsyncGraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
            async with driver.session() as session:
                result = await session.run('MATCH (n) DETACH DELETE n')
                summary = await result.consume()
                nodes_deleted = summary.counters.nodes_deleted
                rels_deleted = summary.counters.relationships_deleted
                print(f' -> Thành công! Đã xóa {nodes_deleted} nodes và {rels_deleted} relationships khỏi Neo4j.')
    except Exception as e:
        print(f' -> Lỗi khi kết nối Neo4j (URI={URI}, USER={USER}): {e}')

asyncio.run(clear_db())
"

# 2. Clear LightRAG Local KV Files
echo "[2/2] Đang xóa thư mục dữ liệu cục bộ của LightRAG (Vector/KV store)..."
if [ -d "backend/data/lightrag" ]; then
    rm -rf backend/data/lightrag/*
    echo " -> Thành công! Đã xóa dọn dẹp backend/data/lightrag/"
else
    echo " -> Thư mục backend/data/lightrag không tồn tại, bỏ qua."
fi

echo "==============================================="
echo " Hoàn tất! Bạn có thể bắt đầu ingest dữ liệu mới."
echo "==============================================="
