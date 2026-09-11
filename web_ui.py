"""Loopback-only deterministic HTML UI for MCP-RagMax.

This process never imports or starts an LLM. Search is Dense + BM25 + RRF;
build/rebuild jobs use the same persistent job manager exposed through MCP.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

import build_jobs
import kb_operations
from config import KNOWLEDGE_DIR, RAG_INDEX_PATH, UI_HOST, UI_PORT
import rag_index_builder
from rag_retrieve import rag_retrieve

ROOT = Path(__file__).resolve().parent
INDEX_PATH = RAG_INDEX_PATH
_HTML = ROOT / "ui" / "rag.html"
_ALLOWED_ORIGINS = {
    f"http://127.0.0.1:{UI_PORT}",
    f"http://localhost:{UI_PORT}",
}
_SEARCH_LOCK = threading.Lock()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _same_origin(request: Request) -> None:
    if request.headers.get("origin") not in _ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="origin not allowed")


def _normalize_reveal_path(raw: str) -> str:
    """Remove the visible source label before resolving a registered file."""
    value = str(raw or "").strip()
    folded = value.casefold()
    if folded.startswith("[source:"):
        value = value[len("[source:"):].strip()
        if value.endswith("]"):
            value = value[:-1].rstrip()
    elif folded.startswith("source:"):
        value = value[len("source:"):].strip()
    return value.strip().strip("`*").rstrip("),.;:!?]").strip()


def _parse_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (raw or "").splitlines():
        if ":" in line and "=" not in line:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in fields:
            fields[key] = value.strip()
    return fields


def _run_search(query: str, mode: str) -> str:
    """One patchable deterministic retrieval boundary for the HTML UI."""
    return str(rag_retrieve.invoke({"query": query, "mode": mode}))


def _health_payload() -> dict[str, object]:
    snapshot = kb_operations.health_snapshot()
    active = build_jobs.active_job_snapshot()
    issues = list(snapshot.get("issues") or [])
    ghosts = list(snapshot.get("ghost_files") or [])
    index_status = str(snapshot.get("index_status") or "missing")
    pipeline_outdated = int(snapshot.get("pipeline_outdated_files") or 0)
    healthy = not issues and not ghosts and pipeline_outdated == 0 and index_status == "ready"
    return {
        "status": "healthy" if healthy else "degraded",
        "registered_files": int(snapshot.get("registered_files") or 0),
        "pipeline_outdated_files": pipeline_outdated,
        "issues": issues,
        "ghost_files": ghosts,
        "index_status": index_status,
        "index_files": int(snapshot.get("index_files") or 0),
        "build_job": active or None,
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "index_path": str(INDEX_PATH),
        "llm_used": False,
        "retrieval": "multilingual MiniLM + BM25 + RRF",
    }


@app.get("/")
def root() -> FileResponse:
    if not _HTML.is_file():
        raise HTTPException(status_code=500, detail="UI file missing")
    return FileResponse(_HTML)


@app.get("/api/health")
def health() -> dict[str, object]:
    try:
        return {"ok": True, **_health_payload()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"health failed: {type(exc).__name__}") from None


@app.get("/api/orientation")
def orientation() -> dict[str, object]:
    try:
        payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid root")
        return {"ok": True, "index": payload}
    except FileNotFoundError:
        return {"ok": True, "index": None}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="rag_index.json invalid") from None


@app.get("/api/files")
def files(query: str = "", limit: int = 100) -> dict[str, object]:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1..500")
    try:
        paths = (
            kb_operations.search_registered_paths(query)
            if query.strip()
            else kb_operations.registered_paths()
        )
        return {
            "ok": True,
            "total": len(paths),
            "files": [
                {"name": Path(path).name, "path": str(Path(path).resolve(strict=False))}
                for path in paths[:limit]
            ],
            "truncated": len(paths) > limit,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"file list failed: {type(exc).__name__}") from None


@app.post("/api/search")
async def search(request: Request) -> dict[str, object]:
    _same_origin(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    query = str(body.get("query") or "").strip() if isinstance(body, dict) else ""
    mode = str(body.get("mode") or "chunks").strip() if isinstance(body, dict) else "chunks"
    if not query or len(query) > 4000:
        raise HTTPException(status_code=400, detail="query must be 1..4000 chars")
    if mode not in {"chunks", "files", "source_first"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    try:
        with _SEARCH_LOCK:
            raw = _run_search(query, mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search failed: {type(exc).__name__}") from None
    return {
        "ok": True,
        "query": query,
        "mode": mode,
        "result": str(raw),
        "llm_used": False,
    }


@app.post("/api/reveal-path")
async def reveal_path(request: Request) -> dict[str, object]:
    """Reveal one registered KB file in Finder, following Agent Lite behavior."""
    _same_origin(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    raw_path = _normalize_reveal_path(body.get("path")) if isinstance(body, dict) else ""
    if not raw_path or len(raw_path) > 4_096:
        raise HTTPException(status_code=400, detail="path is required")
    try:
        candidate = kb_operations.resolve_registered_path(raw_path).resolve(strict=True)
    except (OSError, kb_operations.RegisteredFileNotFound, kb_operations.RegisteredFileAmbiguous):
        raise HTTPException(status_code=403, detail="file is not registered in the knowledge base") from None
    if not candidate.is_file():
        raise HTTPException(status_code=403, detail="file is not available")
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail="Finder reveal requires macOS")
    try:
        result = subprocess.run(
            ["open", "-R", str(candidate)],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise HTTPException(status_code=500, detail="could not open Finder") from None
    if result.returncode:
        raise HTTPException(status_code=500, detail="could not open Finder")
    return {"ok": True, "path": str(candidate)}


@app.post("/api/build")
def start_build(request: Request) -> dict[str, object]:
    _same_origin(request)
    raw = build_jobs.start_build()
    return {"ok": True, "raw": raw, "job": _parse_fields(raw)}


@app.post("/api/index/prepare")
def prepare_index(request: Request) -> dict[str, object]:
    """Return bounded deterministic context for an external/caller LLM."""
    _same_origin(request)
    active = build_jobs.active_job_snapshot()
    if active is not None:
        raise HTTPException(status_code=409, detail="build job is still running")
    try:
        context = rag_index_builder.prepare_index_context()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"index prepare failed: {type(exc).__name__}") from None
    return {"ok": True, "context": context, "llm_used_by_backend": False}


@app.post("/api/index/commit")
async def commit_index(request: Request) -> dict[str, object]:
    """Validate externally-generated topics and atomically write rag_index.json."""
    _same_origin(request)
    if build_jobs.active_job_snapshot() is not None:
        raise HTTPException(status_code=409, detail="build job is still running")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    topics = body.get("topics")
    fingerprint = str(body.get("expected_fingerprint") or "").strip()
    if not isinstance(topics, list):
        raise HTTPException(status_code=400, detail="topics must be an array of strings")
    try:
        result = rag_index_builder.commit_index(topics, fingerprint)
    except rag_index_builder.IndexCommitConflict:
        raise HTTPException(status_code=409, detail="knowledge changed; prepare index again") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"index commit failed: {type(exc).__name__}") from None
    return {"ok": True, "index": result, "llm_used_by_backend": False}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    raw = build_jobs.build_status(job_id)
    return {"ok": True, "raw": raw, "job": _parse_fields(raw)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, object]:
    _same_origin(request)
    raw = build_jobs.cancel_build(job_id)
    return {"ok": True, "raw": raw, "job": _parse_fields(raw)}
