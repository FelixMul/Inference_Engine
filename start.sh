#!/usr/bin/env bash
#
# Start the inference engine server.
#
# Usage:
#   ./start.sh [PORT]
#
set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/dev/shm/Qwen3.5-35B-A3B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting engine on port ${PORT}..."
cd "$SCRIPT_DIR"
.venv/bin/python engine/server.py --port "$PORT" --model-path "$MODEL_PATH" &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for health check
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ~$((i * 5))s"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 5
done

echo "ERROR: Server not ready after 10 minutes"
exit 1
