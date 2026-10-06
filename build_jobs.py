"""Persistent background build jobs for MCP-RagMax.

This module is safe to import from Pipe C: it imports only the standard
library. The heavier embedding/Chroma ingestion stack is loaded only inside the
detached worker process created by ``start_build``. The managed job covers KB
ingestion only and never calls an LLM; orientation prepare/commit is synchronous
through ``rag_manage`` index_prepare/index_commit actions and caller-assisted outside this job manager.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from config import ROOT_DIR, STATE_DIR

_STATE_ROOT = STATE_DIR
_WORKER = ROOT_DIR / "build_job_worker.py"
_JOB_ID_RE = re.compile(r"^build-[0-9a-f]{12}$")
_TERMINAL_STATES = {"done", "cancelled", "error"}
_JOB_HISTORY_MAX = 8
_MAX_OUTPUT_CHARS = 8_000


def _jobs_dir() -> Path:
    return _STATE_ROOT / "build_jobs"


def _start_lock_path() -> Path:
    return _STATE_ROOT / "build_start.lock"


def _job_paths(job_id: str) -> tuple[Path, Path]:
    root = _jobs_dir()
    return root / f"{job_id}.json", root / f"{job_id}.cancel"


def _valid_job_id(job_id: object) -> str | None:
    if not isinstance(job_id, str):
        return None
    value = job_id.strip()
    return value if _JOB_ID_RE.fullmatch(value) else None


def _read_job(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_job(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    )
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _process_alive(pid: object) -> bool:
    try:
        value = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _cap(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    marker = f"\n...[truncated at {_MAX_OUTPUT_CHARS} chars]"
    return text[: max(0, _MAX_OUTPUT_CHARS - len(marker))] + marker


def _new_job_state(job_id: str, kind: str = "build_kb") -> dict[str, object]:
    now = time.time()
    return {
        "job_id": job_id,
        "job_kind": kind,
        "pid": 0,
        "status": "queued",
        "phase": "queued",
        "current_file": "",
        "progress_percent": 0.0,
        "completed_files": 0,
        "total_files": 0,
        "current_chunk": 0,
        "total_chunks": 0,
        "estimated_remaining_seconds": 0.0,
        "cancel_requested": False,
        "summary": "",
        "error": "",
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
    }


def _effective_state(
    state: dict[str, object], cancel_file: Path | None = None
) -> dict[str, object]:
    view = dict(state)
    status = str(view.get("status") or "unknown")
    if status not in _TERMINAL_STATES:
        cancel_requested = bool(cancel_file is not None and cancel_file.exists())
        view["cancel_requested"] = cancel_requested
        if cancel_requested:
            view["status"] = "cancelling"
        pid = view.get("pid")
        age = max(0.0, time.time() - float(view.get("updated_at") or 0.0))
        if not _process_alive(pid) and not (int(pid or 0) == 0 and age < 5.0):
            view["status"] = "error"
            view["phase"] = "error"
            view["error"] = "worker_not_running"
    return view


def _format_job(state: dict[str, object], *, prefix: str | None = None) -> str:
    status = prefix or str(state.get("status") or "unknown")
    lines = [
        f"status={status}",
        f"job_id={state.get('job_id') or ''}",
        f"job_kind={state.get('job_kind') or 'build_kb'}",
        f"phase={state.get('phase') or ''}",
        f"progress_percent={float(state.get('progress_percent') or 0):.1f}",
        f"completed_files={int(state.get('completed_files') or 0)}",
        f"total_files={int(state.get('total_files') or 0)}",
        f"current_chunk={int(state.get('current_chunk') or 0)}",
        f"total_chunks={int(state.get('total_chunks') or 0)}",
        f"estimated_remaining_seconds={float(state.get('estimated_remaining_seconds') or 0):.1f}",
        f"current_file={state.get('current_file') or ''}",
        f"cancel_requested={'true' if state.get('cancel_requested') else 'false'}",
    ]
    if status in {"started", "already_running", "queued", "running", "cancelling"}:
        job_id = state.get("job_id") or ""
        lines.append(f"next=rag_status(view=build, job_id={job_id})")
        lines.append(f"cancel=rag_manage(action=build_cancel, job_id={job_id})")
    if state.get("summary"):
        lines.extend(["result:", str(state["summary"])])
    if state.get("error"):
        lines.append(f"error={state['error']}")
    return _cap("\n".join(lines))


def _job_files_newest_first() -> list[Path]:
    root = _jobs_dir()
    if not root.is_dir():
        return []
    try:
        files = [path for path in root.glob("build-*.json") if path.is_file()]
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return []


def _active_job() -> tuple[dict[str, object], Path] | None:
    now = time.time()
    for state_path in _job_files_newest_first():
        state = _read_job(state_path)
        if not state or state.get("status") in _TERMINAL_STATES:
            continue
        job_id = _valid_job_id(state.get("job_id"))
        if not job_id:
            continue
        _status_file, cancel_file = _job_paths(job_id)
        effective = _effective_state(state, cancel_file)
        if effective.get("status") != "error":
            return effective, state_path
        if int(state.get("pid") or 0) == 0 and now - float(
            state.get("updated_at") or 0.0
        ) < 5.0:
            return state, state_path
    return None


def active_job_snapshot() -> dict[str, object] | None:
    """Return the current managed KB build job without starting or mutating it."""
    active = _active_job()
    return dict(active[0]) if active is not None else None


def _prune_job_history() -> None:
    kept = 0
    for state_path in _job_files_newest_first():
        state = _read_job(state_path)
        if not state:
            continue
        kept += 1
        if kept <= _JOB_HISTORY_MAX or state.get("status") not in _TERMINAL_STATES:
            continue
        job_id = _valid_job_id(state.get("job_id"))
        try:
            state_path.unlink(missing_ok=True)
            if job_id:
                _job_paths(job_id)[1].unlink(missing_ok=True)
        except OSError:
            pass


def _start_job(kind: str) -> str:
    if kind != "build_kb":
        raise ValueError(f"unsupported build job kind: {kind}")
    _STATE_ROOT.mkdir(parents=True, exist_ok=True)
    _jobs_dir().mkdir(parents=True, exist_ok=True)
    with _start_lock_path().open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            active = _active_job()
            if active is not None:
                return _format_job(active[0], prefix="already_running")
            _prune_job_history()
            job_id = f"build-{uuid.uuid4().hex[:12]}"
            status_file, cancel_file = _job_paths(job_id)
            try:
                cancel_file.unlink(missing_ok=True)
            except OSError:
                pass
            state = _new_job_state(job_id, kind)
            _write_job(status_file, state)
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(_WORKER),
                        "--job-id",
                        job_id,
                        "--kind",
                        kind,
                        "--status-file",
                        str(status_file),
                        "--cancel-file",
                        str(cancel_file),
                    ],
                    cwd=str(_WORKER.parent),
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError:
                state.update(
                    status="error",
                    phase="error",
                    error="worker_start_failed",
                    finished_at=time.time(),
                    updated_at=time.time(),
                )
                _write_job(status_file, state)
                return _format_job(state)
            return _format_job(state, prefix="started")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def start_build() -> str:
    """Start one detached KB ingestion build and return its persistent job id."""
    return _start_job("build_kb")


def build_status(job_id: object) -> str:
    clean_id = _valid_job_id(job_id)
    if not clean_id:
        return "status=not_found"
    status_file, cancel_file = _job_paths(clean_id)
    state = _read_job(status_file)
    if state is None:
        return "status=not_found"
    return _format_job(_effective_state(state, cancel_file))


def cancel_build(job_id: object) -> str:
    clean_id = _valid_job_id(job_id)
    if not clean_id:
        return "status=not_found"
    status_file, cancel_file = _job_paths(clean_id)
    state = _read_job(status_file)
    if state is None:
        return "status=not_found"
    effective = _effective_state(state, cancel_file)
    if effective.get("status") in _TERMINAL_STATES:
        return _format_job(effective)
    try:
        cancel_file.parent.mkdir(parents=True, exist_ok=True)
        cancel_file.write_text("cancel\n", encoding="utf-8")
    except OSError:
        return _format_job(effective, prefix="cancel_signal_failed")
    effective["cancel_requested"] = True
    effective["status"] = "cancelling"
    return _format_job(effective, prefix="cancelling")


def _eta(started: float, progress_percent: float) -> float:
    if progress_percent <= 0.0 or progress_percent >= 100.0:
        return 0.0
    elapsed = max(0.0, time.monotonic() - started)
    return max(0.0, elapsed * (100.0 - progress_percent) / progress_percent)


def run_persistent_job(
    job_id: str,
    status_file: Path,
    cancel_file: Path,
    kind: str = "build_kb",
) -> int:
    """Detached-worker entry point for deterministic KB ingestion jobs."""
    state = _read_job(status_file) or _new_job_state(job_id, kind)
    state["job_kind"] = kind
    state["pid"] = os.getpid()
    started_monotonic = time.monotonic()

    def cancelled() -> bool:
        return cancel_file.exists()

    def update(**changes: object) -> None:
        state.update(changes)
        state["cancel_requested"] = cancelled()
        state["updated_at"] = time.time()
        _write_job(status_file, state)

    import ingestor

    def on_file_start(idx: int, total: int, name: str) -> None:
        progress = ((idx - 1) / total * 100.0) if total else 0.0
        update(
            status="cancelling" if cancelled() else "running",
            phase="file",
            current_file=name,
            completed_files=max(0, idx - 1),
            total_files=total,
            current_chunk=0,
            total_chunks=0,
            progress_percent=min(99.0, progress),
            estimated_remaining_seconds=_eta(started_monotonic, progress),
        )

    def on_chunk(
        idx: int,
        total: int,
        name: str,
        chunk_done: int,
        chunk_total: int,
    ) -> None:
        fraction = (chunk_done / chunk_total) if chunk_total else 0.0
        progress = (((idx - 1) + fraction) / total * 100.0) if total else 0.0
        update(
            status="cancelling" if cancelled() else "running",
            phase="chunks",
            current_file=name,
            completed_files=max(0, idx - 1),
            total_files=total,
            current_chunk=chunk_done,
            total_chunks=chunk_total,
            progress_percent=min(99.0, progress),
            estimated_remaining_seconds=_eta(started_monotonic, progress),
        )

    def on_file(
        idx: int,
        total: int,
        name: str,
        status: str,
        n_chunks: int,
        error: str | None,
    ) -> None:
        progress = (idx / total * 100.0) if total else 100.0
        update(
            status="cancelling" if cancelled() else "running",
            phase="file_done",
            current_file=name,
            completed_files=idx,
            total_files=total,
            current_chunk=0,
            total_chunks=0,
            progress_percent=min(99.0, progress),
            estimated_remaining_seconds=_eta(started_monotonic, progress),
        )

    update(status="running", phase="scanning")
    try:
        result = ingestor.sync_knowledge_base(
            on_file=on_file,
            on_file_start=on_file_start,
            on_chunk_progress=on_chunk,
            should_cancel=cancelled,
        )
    except ingestor.BuildCancelled:
        update(
            status="cancelled",
            phase="cancelled",
            estimated_remaining_seconds=0.0,
            finished_at=time.time(),
        )
        return 0
    except Exception as exc:
        update(
            status="error",
            phase="error",
            error=f"build_failed:{type(exc).__name__}",
            estimated_remaining_seconds=0.0,
            finished_at=time.time(),
        )
        return 1

    if cancelled():
        update(
            status="cancelled",
            phase="cancelled",
            estimated_remaining_seconds=0.0,
            finished_at=time.time(),
        )
        return 0

    rows = list(result.get("rows") or [])
    counts = {key: 0 for key in ("new", "changed", "skip", "error")}
    chunks = 0
    for row in rows:
        status = str(row.get("status") or "")
        if status in counts:
            counts[status] += 1
        chunks += int(row.get("n_chunks") or 0)
    summary = "\n".join(
        [
            "status=ok",
            f"files={int(result.get('total_found') or 0)}",
            f"new={counts['new']}",
            f"changed={counts['changed']}",
            f"skipped={counts['skip']}",
            f"errors={counts['error']}",
            f"stored_chunks={chunks}",
            f"health_issues={len(result.get('health_issues') or [])}",
            f"ghost_files={int(result.get('ghost_count') or 0)}",
            f"seconds={float(result.get('elapsed') or 0.0):.2f}",
            "orientation_next=rag_manage(action=index_prepare)",
        ]
    )
    update(
        status="done",
        phase="done",
        progress_percent=100.0,
        completed_files=int(result.get("total_found") or 0),
        total_files=int(result.get("total_found") or 0),
        current_file="",
        current_chunk=0,
        total_chunks=0,
        estimated_remaining_seconds=0.0,
        summary=summary,
        finished_at=time.time(),
    )
    return 0
