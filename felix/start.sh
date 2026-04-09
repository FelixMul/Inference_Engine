#!/usr/bin/env bash
#
# Start Felix's inference engine.
#
# Usage:
#   ./felix/start.sh [PORT]
#
set -euo pipefail

PORT="${1:-8003}"
MODEL_PATH="${MODEL_PATH:-/dev/shm/Qwen3.5-35B-A3B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

echo "Starting Felix engine on port ${PORT}..."
cd "$SCRIPT_DIR"
$PYTHON server.py --port "$PORT" --model-path "$MODEL_PATH" &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for health
echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ~$((i * 3))s"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server died"
        exit 1
    fi
    sleep 3
done

echo "ERROR: Server not ready after 6 minutes"
exit 1
