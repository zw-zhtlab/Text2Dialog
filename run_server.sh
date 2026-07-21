#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
if [ ! -x ".venv/bin/text2dialog-server" ]; then
  .venv/bin/python -m pip install -e .
fi
HOST="${TEXT2DIALOG_HOST:-127.0.0.1}"
PORT="${TEXT2DIALOG_PORT:-8000}"
.venv/bin/text2dialog-server --host "$HOST" --port "$PORT"
