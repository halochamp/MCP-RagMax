#!/usr/bin/env python3
"""Standalone MCP-RagMax HTML UI launcher.

MCP-RagMax no longer owns a chat agent or local LLM. Running this file starts a
loopback-only deterministic RAG console for search, build, progress/cancel, and
health inspection.
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from config import UI_PORT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic MCP-RagMax web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=UI_PORT)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("MCP-RagMax UI is local-only; use 127.0.0.1 or localhost")
    uvicorn.run("web_ui:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
