#!/usr/bin/env bash
# Install MCP-RagMax into a project-local Python 3.11 environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "=== MCP-RagMax installer ==="
echo "[1/4] Checking platform"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "[error] This release targets macOS on Apple Silicon (arm64)."; exit 1
fi
echo "[2/4] Checking Python 3.11"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then echo "[error] Could not find $PYTHON_BIN"; exit 1; fi
if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
then echo "[error] Python 3.11 is required: $($PYTHON_BIN --version)"; exit 1; fi

echo "[3/4] Creating/reusing project virtual environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then "$PYTHON_BIN" -m venv "$VENV_DIR"; fi
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --require-hashes -r "$PROJECT_DIR/requirements.txt"

echo "[4/4] Installation complete"
cat <<EOF

Next steps:
  1. Put .md/.txt/.pdf/.csv/.json documents under:
       $PROJECT_DIR/workspace/knowledge/
  2. source "$VENV_DIR/bin/activate"
  3. python tools/doctor.py
  4. python tools/build_index.py
  5. python main.py        # loopback UI: http://127.0.0.1:8770

MCP stdio entry point:
  python mcp_server.py

MCP-RagMax search/build logic is deterministic and does not start a chat LLM.
EOF
