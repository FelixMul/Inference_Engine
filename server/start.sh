#!/usr/bin/env bash
set -euo pipefail

PORT=${1:-8000}

uvicorn server.main:app --host 0.0.0.0 --port "$PORT"
