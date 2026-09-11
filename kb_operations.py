"""Deterministic shared knowledge-base operations for MCP and HTML clients.

This module has no chat-model dependency. It owns the small read-only KB
primitives shared by the MCP adapter and standalone browser UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import file_registry
from config import RAG_INDEX_PATH
from pipeline_config import ORIENTATION_POLICY_ID


_SUPPORTED_READ_EXT = frozenset({".txt", ".md", ".pdf", ".csv", ".json"})


class RegisteredFileNotFound(LookupError):
    """Requested file is not present in the KB registry."""


class RegisteredFileAmbiguous(LookupError):
    """A partial filename matched more than one registered KB file."""

    def __init__(self, query: str, matches: list[str]):
        super().__init__(query)
        self.query = query
        self.matches = tuple(matches)


def registered_paths() -> list[str]:
    """Return registered KB paths in deterministic filename/path order."""
    paths = file_registry.all_registered()
    return sorted(paths, key=lambda value: (Path(value).name.casefold(), value.casefold()))


def search_registered_paths(query: str) -> list[str]:
    """Return registered paths whose filename contains ``query`` case-insensitively."""
    needle = query.strip().casefold()
    if not needle:
        return []
    return [path for path in registered_paths() if needle in Path(path).name.casefold()]


def resolve_registered_path(filename: str) -> Path:
    """Resolve an exact path/name or unique partial filename inside the registry only."""
    requested = filename.strip()
    if not requested:
        raise RegisteredFileNotFound(filename)

    paths = registered_paths()
    for raw in paths:
        path = Path(raw)
        if str(path) == requested or path.name == requested:
            return path

    matches = [raw for raw in paths if requested.casefold() in Path(raw).name.casefold()]
    if not matches:
        raise RegisteredFileNotFound(requested)
    if len(matches) > 1:
        raise RegisteredFileAmbiguous(requested, matches)
    return Path(matches[0])


def _read_pdf(path: Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_registered_file(filename: str) -> tuple[Path, str]:
    """Read one registered KB document without allowing arbitrary filesystem paths."""
    path = resolve_registered_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"registered file is unavailable: {path.name}")
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_READ_EXT:
        raise ValueError(f"unsupported registered file type: {ext or '(none)'}")
    if ext == ".pdf":
        text = _read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return path, text


def _orientation_status(payload: object, current_fp: str, registered_count: int) -> tuple[str, int]:
    """Classify canonical orientation freshness without touching disk or stores."""
    if not isinstance(payload, dict):
        return "invalid", 0
    try:
        index_files = int(payload.get("total") or 0)
    except (TypeError, ValueError):
        return "invalid", 0
    stored_fp = str(payload.get("registry_fingerprint") or "")
    stored_policy = str(payload.get("orientation_policy_id") or "")
    ready = bool(
        stored_fp
        and stored_fp == current_fp
        and index_files == int(registered_count)
        and stored_policy == ORIENTATION_POLICY_ID
    )
    return ("ready" if ready else "stale"), index_files


def health_snapshot() -> dict[str, object]:
    """Return deterministic store/registry/orientation health without mutating the KB."""
    import store

    issues = list(store.health_check())
    ghosts = list(file_registry.ghost_files())
    registered = registered_paths()
    pipeline_outdated = int(file_registry.pipeline_outdated_count())
    index_path = RAG_INDEX_PATH
    index_status = "missing"
    index_files = 0
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("rag_index root is not an object")
        index_status, index_files = _orientation_status(
            payload,
            file_registry.registry_fingerprint(),
            len(registered),
        )
    except FileNotFoundError:
        index_status = "missing"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        index_status = "invalid"

    return {
        "issues": issues,
        "ghost_files": ghosts,
        "registered_files": len(registered),
        "pipeline_outdated_files": pipeline_outdated,
        "index_status": index_status,
        "index_files": index_files,
    }
