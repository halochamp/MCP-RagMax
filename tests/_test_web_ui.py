"""Loopback deterministic HTML UI tests; no live model/index required."""
from __future__ import annotations

from pathlib import Path
import tempfile
from unittest import mock

from fastapi.testclient import TestClient

from _runner import Runner

r = Runner("web ui")


def _client():
    import web_ui
    return web_ui, TestClient(web_ui.app)


def _origin(web_ui) -> dict[str, str]:
    return {"Origin": f"http://127.0.0.1:{web_ui.UI_PORT}"}


def t_root_serves_rag_console():
    web_ui, client = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert "MCP-RagMax" in response.text
    assert "No LLM" in response.text
    assert "Build KB" in response.text
    assert "Prepare Index" in response.text
    assert "Commit Topics" in response.text


def t_search_is_same_origin_and_no_llm():
    web_ui, client = _client()
    with mock.patch.object(web_ui, "_run_search", return_value="[1] deterministic result") as invoke:
        blocked = client.post("/api/search", json={"query": "risk"})
        assert blocked.status_code == 403
        ok = client.post(
            "/api/search",
            headers=_origin(web_ui),
            json={"query": "risk", "mode": "chunks"},
        )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["llm_used"] is False
    assert payload["result"] == "[1] deterministic result"
    invoke.assert_called_once_with("risk", "chunks")


def t_reveal_path_requires_same_origin_and_registered_file():
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        web_ui, client = _client()
        source = tmp_path / "manual.pdf"
        source.write_bytes(b"pdf")
        with mock.patch.object(web_ui.kb_operations, "resolve_registered_path", return_value=source), \
             mock.patch.object(web_ui.subprocess, "run") as opener:
            opener.return_value.returncode = 0
            blocked = client.post("/api/reveal-path", json={"path": str(source)})
            ok = client.post(
                "/api/reveal-path",
                headers=_origin(web_ui),
                json={"path": str(source)},
            )
        assert blocked.status_code == 403
        assert ok.status_code == 200
        assert ok.json()["path"] == str(source.resolve())
        assert opener.call_args.args[0][:2] == ["open", "-R"]


def t_reveal_path_rejects_unregistered_file():
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        web_ui, client = _client()
        source = tmp_path / "outside.pdf"
        source.write_bytes(b"pdf")
        with mock.patch.object(
            web_ui.kb_operations,
            "resolve_registered_path",
            side_effect=web_ui.kb_operations.RegisteredFileNotFound(str(source)),
        ):
            response = client.post(
                "/api/reveal-path",
                headers=_origin(web_ui),
                json={"path": str(source)},
            )
        assert response.status_code == 403


def t_reveal_path_strips_source_label_before_registry_lookup():
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        web_ui, client = _client()
        source = tmp_path / "manual.pdf"
        source.write_bytes(b"pdf")

        def resolve(raw_path):
            assert raw_path == str(source)
            return source

        with mock.patch.object(web_ui.kb_operations, "resolve_registered_path", side_effect=resolve), \
             mock.patch.object(web_ui.subprocess, "run") as opener:
            opener.return_value.returncode = 0
            response = client.post(
                "/api/reveal-path",
                headers=_origin(web_ui),
                json={"path": f"[source: {source}]"},
            )
        assert response.status_code == 200
        assert opener.call_args.args[0] == ["open", "-R", str(source.resolve())]


def t_index_prepare_commit_uses_external_topics_only():
    web_ui, client = _client()
    prepared = {
        "status": "prepared",
        "expected_fingerprint": "a" * 64,
        "total_files": 2,
        "sampled_files": 2,
        "filenames": ["alpha.md", "beta.pdf"],
        "top_tags": {"finance": 1},
        "source_types": {"md": 1, "pdf": 1},
        "topic_rules": {"guidance": "caller summarizes"},
        "llm_used_by_backend": False,
    }
    committed = {
        "total": 2,
        "topics": ["การลงทุน", "ความเสี่ยง"],
        "topics_line": "การลงทุน | ความเสี่ยง",
        "topic_method": "caller_llm",
        "llm_used": True,
        "llm_used_by_backend": False,
    }
    with mock.patch.object(web_ui.build_jobs, "active_job_snapshot", return_value=None), \
         mock.patch.object(web_ui.rag_index_builder, "prepare_index_context", return_value=prepared) as prepare, \
         mock.patch.object(web_ui.rag_index_builder, "commit_index", return_value=committed) as commit:
        blocked = client.post("/api/index/prepare")
        assert blocked.status_code == 403
        p = client.post("/api/index/prepare", headers=_origin(web_ui))
        c = client.post(
            "/api/index/commit",
            headers=_origin(web_ui),
            json={"topics": ["การลงทุน", "ความเสี่ยง"], "expected_fingerprint": "a" * 64},
        )
    assert p.status_code == 200
    assert p.json()["context"]["expected_fingerprint"] == "a" * 64
    assert p.json()["llm_used_by_backend"] is False
    assert c.status_code == 200
    assert c.json()["index"]["topic_method"] == "caller_llm"
    assert c.json()["llm_used_by_backend"] is False
    prepare.assert_called_once_with()
    commit.assert_called_once_with(["การลงทุน", "ความเสี่ยง"], "a" * 64)


def t_index_prepare_rejects_while_build_is_running():
    web_ui, client = _client()
    with mock.patch.object(
        web_ui.build_jobs,
        "active_job_snapshot",
        return_value={"job_id": "build-123456789abc", "status": "running"},
    ):
        response = client.post("/api/index/prepare", headers=_origin(web_ui))
    assert response.status_code == 409


def t_build_status_cancel_use_persistent_job_manager():
    web_ui, client = _client()
    origin = _origin(web_ui)
    started = "status=started\njob_id=build-123456789abc\nphase=queued\nprogress_percent=0.0"
    running = "status=running\njob_id=build-123456789abc\nphase=chunks\nprogress_percent=42.5"
    cancelling = "status=cancelling\njob_id=build-123456789abc\nphase=chunks\nprogress_percent=42.5"
    with mock.patch.object(web_ui.build_jobs, "start_build", return_value=started), \
         mock.patch.object(web_ui.build_jobs, "build_status", return_value=running), \
         mock.patch.object(web_ui.build_jobs, "cancel_build", return_value=cancelling):
        start = client.post("/api/build", headers=origin)
        status = client.get("/api/jobs/build-123456789abc")
        cancel = client.post("/api/jobs/build-123456789abc/cancel", headers=origin)
    assert start.json()["job"]["job_id"] == "build-123456789abc"
    assert status.json()["job"]["progress_percent"] == "42.5"
    assert cancel.json()["job"]["status"] == "cancelling"


def t_health_declares_no_backend_llm():
    web_ui, client = _client()
    fake = {
        "status": "healthy",
        "registered_files": 3,
        "pipeline_outdated_files": 0,
        "issues": [],
        "ghost_files": [],
        "index_status": "ready",
        "index_files": 3,
        "build_job": None,
        "knowledge_dir": "/tmp/knowledge",
        "index_path": "/tmp/rag_index.json",
        "llm_used": False,
        "retrieval": "multilingual MiniLM + BM25 + RRF",
    }
    with mock.patch.object(web_ui, "_health_payload", return_value=fake):
        response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["llm_used"] is False


r.test("root serves deterministic RAG console", t_root_serves_rag_console)
r.test("search enforces same-origin and no-LLM contract", t_search_is_same_origin_and_no_llm)
r.test("reveal path enforces origin and registry scope", t_reveal_path_requires_same_origin_and_registered_file)
r.test("reveal path rejects unregistered files", t_reveal_path_rejects_unregistered_file)
r.test("reveal path strips source labels", t_reveal_path_strips_source_label_before_registry_lookup)
r.test("index prepare/commit uses caller topics", t_index_prepare_commit_uses_external_topics_only)
r.test("index prepare rejects active build", t_index_prepare_rejects_while_build_is_running)
r.test("build/status/cancel use persistent jobs", t_build_status_cancel_use_persistent_job_manager)
r.test("health declares no backend LLM", t_health_declares_no_backend_llm)

if __name__ == "__main__":
    r.exit()
