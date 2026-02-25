#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r text2dialog/requirements.txt
cd text2dialog
HOST="${TEXT2DIALOG_HOST:-127.0.0.1}"
PORT="${TEXT2DIALOG_PORT:-8000}"
python3 -m uvicorn server:app --host "$HOST" --port "$PORT"
