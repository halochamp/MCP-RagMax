#!/bin/zsh
set -u
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${RAGMAX_PYTHON:-$APP_DIR/.venv/bin/python}"
PORT="${RAGMAX_UI_PORT:-8770}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Error: project Python not found: $PYTHON"
  echo "Run bash install_library/install.sh first."
  read -r "?Press Return to close..."
  exit 1
fi
cd "$APP_DIR" || exit 1
exec "$PYTHON" main.py --host 127.0.0.1 --port "$PORT"
