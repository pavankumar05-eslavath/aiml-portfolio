#!/usr/bin/env bash
# Start the API for local development.
#
# Run from the repository root so that the relative paths in .env
# (DATABASE_URL, QDRANT_PATH, IMAGE_ROOT) resolve correctly.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec "$PYTHON" -m uvicorn app.main:app \
  --app-dir backend \
  --host "$HOST" \
  --port "$PORT" \
  "${@:-}"
