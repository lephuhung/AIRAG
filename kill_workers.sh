#!/bin/bash
# Kill all running worker processes

echo "Killing all worker processes..."
pkill -f "python.*worker" 2>/dev/null
pkill -f "app.workers.runner" 2>/dev/null
sleep 2

# Verify no workers running
REMAINING=$(ps aux | grep -E "python.*worker|app.workers.runner" | grep -v grep | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "All workers killed."
else
    echo "WARNING: $REMAINING worker process(es) still running."
    ps aux | grep -E "python.*worker|app.workers.runner" | grep -v grep
fi
