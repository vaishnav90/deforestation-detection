#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
echo "Starting CanopyWatch at http://127.0.0.1:8000"
exec .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --reload
