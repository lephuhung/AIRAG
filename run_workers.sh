#!/bin/bash
# Restart all worker processes
# Workers: parse (4), embed (2), caption (2), kg (2)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting workers..."

WORKER_TYPES=("parse" "embed" "caption" "kg")
COUNTS=(4 2 2 2)

for i in "${!WORKER_TYPES[@]}"; do
    TYPE="${WORKER_TYPES[$i]}"
    COUNT="${COUNTS[$i]}"
    echo "Starting $COUNT x worker-$TYPE ..."
    for j in $(seq 1 $COUNT); do
        LOG="$SCRIPT_DIR/logs/worker-${TYPE}-${j}.log"
        nohup env WORKER_TYPE="$TYPE" PYTHONPATH="$SCRIPT_DIR/backend" "$SCRIPT_DIR/.venv/bin/python" -m app.workers.runner \
            >> "$LOG" 2>&1 &
        echo "  worker-$TYPE #$j started (PID: $!)"
    done
done

echo "All workers started."
sleep 1
ps aux | grep -E "python.*worker" | grep -v grep | head -20
