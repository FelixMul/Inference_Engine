#!/usr/bin/env bash
#
# Quick evaluation: conformance + throughput at c=4,16,64
#
# Usage:
#   ./eval/quick_eval.sh [BASE_URL]
#
# Example:
#   ./eval/quick_eval.sh http://localhost:8000
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

BASE_URL="${1:-http://localhost:8000}"

echo "=== Quick Eval ==="
echo "Target: ${BASE_URL}"
echo ""

# 1. API conformance (seconds)
echo "--- Step 1/2: API Conformance ---"
$PYTHON -m eval.check_server --base-url "${BASE_URL}"
echo ""

# 2. Throughput at c=4,16,64 with 16 requests each
echo "--- Step 2/2: Throughput (c=4,16,64, 16 reqs each) ---"
$PYTHON -m eval.throughput.run_throughput \
    --base-url "${BASE_URL}" \
    --concurrency 4 16 64 \
    --num-requests 16 \
    --num-prompts 32 \
    --output results/quick_throughput.json \
    --baseline baseline/results/throughput_baseline.json
echo ""

echo "=== Quick Eval Complete ==="
