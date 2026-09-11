"""Persistent deterministic RAG build-job lifecycle tests; no live model required."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

from _runner import Runner

r = Runner("build jobs")


def _isolated_state(module, root: Path):
    return mock.patch.object(module, "_STATE_ROOT", root)


def t_start_returns_persistent_job_id_without_waiting():
    import build_jobs

    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        with mock.patch.object(build_jobs.subprocess, "Popen", return_value=mock.Mock()) as popen:
            out = build_jobs.start_build()
        assert "status=started" in out
        job_id = next(line.split("=", 1)[1] for line in out.splitlines() if line.startswith("job_id="))
        assert job_id.startswith("build-")
        assert (Path(td) / "build_jobs" / f"{job_id}.json").is_file()
        args = popen.call_args.args[0]
        assert args[args.index("--kind") + 1] == "build_kb"


def t_non_build_job_kind_is_rejected():
    import build_jobs

    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        try:
            build_jobs._start_job("rag_rebuild_index")
        except ValueError as exc:
            assert "unsupported build job kind" in str(exc)
        else:
            raise AssertionError("rag index prepare/commit must not become a background build job")


def t_active_job_rejects_duplicate_start():
    import build_jobs

    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        job_id = "build-123456789abc"
        status_file, _cancel = build_jobs._job_paths(job_id)
        state = build_jobs._new_job_state(job_id)
        state.update(pid=os.getpid(), status="running", phase="chunks")
        build_jobs._write_job(status_file, state)
        with mock.patch.object(build_jobs.subprocess, "Popen") as popen:
            out = build_jobs.start_build()
        assert "status=already_running" in out
        assert f"job_id={job_id}" in out
        popen.assert_not_called()


def t_status_survives_process_boundary_via_state_file():
    import build_jobs

    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        job_id = "build-abcdef123456"
        status_file, _cancel = build_jobs._job_paths(job_id)
        state = build_jobs._new_job_state(job_id)
        state.update(
            pid=os.getpid(), status="running", phase="chunks",
            progress_percent=42.5, completed_files=2, total_files=5,
            current_file="manual.pdf", current_chunk=17, total_chunks=40,
        )
        build_jobs._write_job(status_file, state)
        out = build_jobs.build_status(job_id)
        assert "status=running" in out
        assert "progress_percent=42.5" in out
        assert "current_file=manual.pdf" in out
        assert "current_chunk=17" in out


def t_cancel_build_writes_cooperative_marker():
    import build_jobs

    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        job_id = "build-fedcba654321"
        status_file, cancel_file = build_jobs._job_paths(job_id)
        state = build_jobs._new_job_state(job_id)
        state.update(pid=os.getpid(), status="running", phase="file")
        build_jobs._write_job(status_file, state)
        out = build_jobs.cancel_build(job_id)
        assert "status=cancelling" in out
        assert "cancel_requested=true" in out
        assert cancel_file.read_text(encoding="utf-8") == "cancel\n"


def t_worker_completion_does_not_generate_rag_index():
    import build_jobs

    class BuildCancelled(RuntimeError):
        pass

    fake = types.SimpleNamespace(BuildCancelled=BuildCancelled)

    def sync_knowledge_base(**kwargs):
        kwargs["on_file_start"](1, 1, "one.md")
        kwargs["on_chunk_progress"](1, 1, "one.md", 1, 2)
        kwargs["on_chunk_progress"](1, 1, "one.md", 2, 2)
        kwargs["on_file"](1, 1, "one.md", "new", 2, None)
        return {
            "total_found": 1,
            "rows": [{"name": "one.md", "status": "new", "n_chunks": 2}],
            "elapsed": 0.1,
            "health_issues": [],
            "ghost_count": 0,
        }

    fake.sync_knowledge_base = sync_knowledge_base
    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        job_id = "build-111111111111"
        status_file, cancel_file = build_jobs._job_paths(job_id)
        build_jobs._write_job(status_file, build_jobs._new_job_state(job_id))
        original_builder = sys.modules.pop("rag_index_builder", None)
        try:
            with mock.patch.dict(sys.modules, {"ingestor": fake}):
                code = build_jobs.run_persistent_job(job_id, status_file, cancel_file)
            assert "rag_index_builder" not in sys.modules
        finally:
            if original_builder is not None:
                sys.modules["rag_index_builder"] = original_builder
        state = build_jobs._read_job(status_file)
        assert code == 0
        assert state["status"] == "done"
        assert state["progress_percent"] == 100.0
        assert "stored_chunks=2" in state["summary"]
        assert "orientation_next=rag_rebuild_index(mode=prepare)" in state["summary"]
        assert "index_topics=" not in state["summary"]


def t_worker_cancel_is_terminal_not_error():
    import build_jobs

    class BuildCancelled(RuntimeError):
        pass

    fake = types.SimpleNamespace(BuildCancelled=BuildCancelled)

    def sync_knowledge_base(**kwargs):
        assert kwargs["should_cancel"]()
        raise BuildCancelled("cancelled")

    fake.sync_knowledge_base = sync_knowledge_base
    with tempfile.TemporaryDirectory() as td, _isolated_state(build_jobs, Path(td)):
        job_id = "build-222222222222"
        status_file, cancel_file = build_jobs._job_paths(job_id)
        build_jobs._write_job(status_file, build_jobs._new_job_state(job_id))
        cancel_file.write_text("cancel\n", encoding="utf-8")
        with mock.patch.dict(sys.modules, {"ingestor": fake}):
            code = build_jobs.run_persistent_job(job_id, status_file, cancel_file)
        state = build_jobs._read_job(status_file)
        assert code == 0
        assert state["status"] == "cancelled"
        assert state["phase"] == "cancelled"
        assert state["error"] == ""


r.test("start returns persistent job id", t_start_returns_persistent_job_id_without_waiting)
r.test("non-build job kind is rejected", t_non_build_job_kind_is_rejected)
r.test("active job rejects duplicate start", t_active_job_rejects_duplicate_start)
r.test("status persists through state file", t_status_survives_process_boundary_via_state_file)
r.test("cancel writes cooperative marker", t_cancel_build_writes_cooperative_marker)
r.test("worker completion leaves orientation to caller", t_worker_completion_does_not_generate_rag_index)
r.test("worker cancellation is terminal not error", t_worker_cancel_is_terminal_not_error)

if __name__ == "__main__":
    r.exit()
