#!/usr/bin/env bash
# Run the LeLab server from its own virtual environment, so it never picks up
# packages from a shared/system Python (e.g. Anaconda base). See setup below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/leLab-main"
VENV_PYTHON="$APP_DIR/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "error: virtual environment not found at $APP_DIR/.venv" >&2
  echo "Create it once with a Python >= 3.12 interpreter:" >&2
  echo "  cd \"$APP_DIR\"" >&2
  echo "  /path/to/python3.12 -m venv .venv" >&2
  echo "  .venv/bin/python -m pip install --upgrade pip" >&2
  echo "  .venv/bin/python -m pip install -e ." >&2
  exit 1
fi

cd "$APP_DIR"
exec "$VENV_PYTHON" -m uvicorn lelab.server:app --host 127.0.0.1 --port 8000
