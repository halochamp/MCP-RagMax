"""Standalone public runtime configuration for MCP-RagMax."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _path_setting(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default.resolve()
    value = Path(raw).expanduser()
    return (value if value.is_absolute() else ROOT_DIR / value).resolve()


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


WORKSPACE_DIR = _path_setting("RAGMAX_WORKSPACE", ROOT_DIR / "workspace")
KNOWLEDGE_DIR = _path_setting("RAGMAX_KNOWLEDGE_DIR", WORKSPACE_DIR / "knowledge")
STATE_DIR = _path_setting("RAGMAX_STATE_DIR", WORKSPACE_DIR / ".rag_state")
CHROMA_DIR = STATE_DIR / "chroma"
BM25_PATH = STATE_DIR / "bm25.pkl"
REGISTRY_PATH = STATE_DIR / "file_hashes.json"
MEMORY_PATH = STATE_DIR / "memory.md"
LOG_DIR = STATE_DIR / "logs"
RAG_INDEX_PATH = STATE_DIR / "rag_index.json"
UI_HOST = "127.0.0.1"
UI_PORT = _bounded_int("RAGMAX_UI_PORT", 8770, 1024, 65535)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def source_key(path: str | Path) -> str:
    """Return a portable relative key for a file inside KNOWLEDGE_DIR."""
    candidate = _resolved(path)
    try:
        return candidate.relative_to(KNOWLEDGE_DIR).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"File is outside the configured knowledge directory: {candidate}"
        ) from exc


def source_path(source: str | Path) -> Path:
    """Resolve a stored key and reject absolute, traversal, or symlink escapes."""
    raw = Path(source).expanduser()
    candidate = _resolved(raw if raw.is_absolute() else KNOWLEDGE_DIR / raw)
    try:
        candidate.relative_to(KNOWLEDGE_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Source is outside the configured knowledge directory: {candidate}"
        ) from exc
    return candidate


def ensure_runtime_dirs() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
