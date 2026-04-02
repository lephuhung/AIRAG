#!/bin/bash
# Restart all worker processes
# Workers: parse (4), embed (2), caption (2), kg (2)

set -e
cd "$(dirname "$0")"

echo "Starting workers..."

# Activate venv if using it
if [ -d ".venv" ]; then
    VENV_BIN=".venv/bin"
else
    VENV_BIN=""
fi

WORKER_TYPES=("parse" "embed" "caption" "kg")
COUNTS=(4 2 2 2)

for i in "${!WORKER_TYPES[@]}"; do
    TYPE="${WORKER_TYPES[$i]}"
    COUNT="${COUNTS[$i]}"
    echo "Starting $COUNT x worker-$TYPE ..."
    for j in $(seq 1 $COUNT); do
        nohup WORKER_TYPE="$TYPE" ${VENV_BIN}python -m app.workers.runner \
            >> "logs/worker-${TYPE}-${j}.log" 2>&1 &
        echo "  worker-$TYPE #$j started (PID: $!)"
    done
done

echo "All workers started."
sleep 1
ps aux | grep -E "python.*worker" | grep -v grep | head -20
