#!/bin/bash
# Clear all data in ChromaDB (delete all collections)
# Usage: ./clear_chromadb.sh

set -e

CHROMA_HOST="${CHROMA_HOST:-localhost}"
CHROMA_PORT="${CHROMA_PORT:-8002}"

echo "Connecting to ChromaDB at ${CHROMA_HOST}:${CHROMA_PORT}..."

# Check if chromadb package is available
if ! python3 -c "import chromadb" 2>/dev/null; then
    echo "Error: chromadb package not found. Trying venv..."
    VENV_PYTHON="${VIRTUAL_ENV:-.venv}/bin/python3"
    if [ ! -f "$VENV_PYTHON" ]; then
        echo "Error: Virtual environment not found at $VENV_PYTHON"
        exit 1
    fi
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

$PYTHON << EOF
import chromadb

chroma_host = "$CHROMA_HOST"
chroma_port = $CHROMA_PORT

client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

# List all collections
collections = client.list_collections()
print(f"Found {len(collections)} collection(s):")

for col in collections:
    print(f"  - Deleting: {col.name}")
    client.delete_collection(col.name)

print("\nAll collections deleted successfully.")
EOF

echo "Done."
