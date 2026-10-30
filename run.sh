#!/usr/bin/env bash
# H3 Create - starts the web app at http://127.0.0.1:7880 (H3C_HOST / H3C_PORT to change).
# Settings come from the environment or a .env file next to this script (copy .env.example to .env).
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY="${H3C_PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" -m uvicorn server.main:app --host "${H3C_HOST:-127.0.0.1}" --port "${H3C_PORT:-7880}" --log-level warning
