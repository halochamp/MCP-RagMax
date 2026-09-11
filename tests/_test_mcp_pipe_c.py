"""Deterministic Pipe C (MCP) -> Pipe B contract tests; no live model required."""

import asyncio
import sys
import threading
from pathlib import Path
from unittest import mock

from _runner import Runner

r = Runner("mcp pipe c")


def _server_module():
    import mcp_server

    return mcp_server


def t_catalog_exposes_nine_bounded_tools():
    server = _server_module()
    tools = asyncio.run(server.mcp.list_tools())
    assert [item.name for item in tools] == [
        "rag_retrieve",
        "rag_list",
        "rag_search_files",
        "rag_read_file",
        "build_kb",
        "build_status",
        "cancel_build",
        "rag_rebuild_index",
        "rag_health",
    ]
    by_name = {item.name: item for item in tools}
    schema = by_name["rag_retrieve"].inputSchema
    assert set(schema["properties"]) == {
        "query",
        "mode",
        "tags",
        "filename_contains",
        "created_after",
        "created_before",
        "source_type",
    }
    assert schema["required"] == ["query"]
    assert by_name["rag_list"].inputSchema.get("required", []) == []
    retrieve_description = by_name["rag_retrieve"].description or ""
    assert "SAME question" in retrieve_description
    assert "preserve one intent" in retrieve_description
    assert "Do not add a new subquestion" in retrieve_description
    assert by_name["rag_search_files"].inputSchema["required"] == ["query"]
    assert by_name["rag_read_file"].inputSchema["required"] == ["filename"]
    assert by_name["build_kb"].inputSchema.get("required", []) == []
    assert by_name["build_status"].inputSchema["required"] == ["job_id"]
    assert by_name["cancel_build"].inputSchema["required"] == ["job_id"]
    rebuild_schema = by_name["rag_rebuild_index"].inputSchema
    assert rebuild_schema.get("required", []) == []
    assert set(rebuild_schema["properties"]) == {"mode", "topics", "expected_fingerprint"}
    assert by_name["rag_health"].inputSchema.get("required", []) == []


def t_pipe_c_does_not_load_pipe_a_or_llm():
    server = _server_module()
    assert server is not None
    assert "main" not in sys.modules
    assert "llm_client" not in sys.modules
    assert "rag_search" not in sys.modules
    assert "ingestor" not in sys.modules
    assert "rag_index_builder" not in sys.modules


def t_pipe_b_module_resolves_inside_this_rag_tree():
    _server_module()
    import rag_retrieve

    expected_root = Path(__file__).resolve().parents[1]
    assert Path(rag_retrieve.__file__).resolve().parent == expected_root


def t_delegates_validated_request_to_pipe_b():
    server = _server_module()
    captured = {}

    def fake_invoke(arguments):
        captured.update(arguments)
        return "PIPE_B_RESULT"

    with mock.patch.object(server, "_invoke_pipe_b", side_effect=fake_invoke):
        result = server.rag_retrieve(
            query=[" Thai query ", "English query"],
            mode=" SOURCE_FIRST ",
            tags=" finance ",
            filename_contains=" report ",
            created_after="2026-01-01",
            created_before="2026-12-31",
            source_type=" md ",
        )

    assert result == "PIPE_B_RESULT"
    assert captured == {
        "query": ["Thai query", "English query"],
        "mode": "source_first",
        "tags": "finance",
        "filename_contains": "report",
        "created_after": "2026-01-01",
        "created_before": "2026-12-31",
        "source_type": "md",
    }


def t_rejects_invalid_requests_before_pipe_b():
    server = _server_module()
    invalid = [
        {"query": ""},
        {"query": "x" * (server._MAX_QUERY_CHARS + 1)},
        {"query": []},
        {"query": ["ok"] * 9},
        {"query": "ok", "mode": "unknown"},
        {"query": "ok", "created_after": "2026-02-30"},
        {"query": "ok", "created_after": "2026-12-31", "created_before": "2026-01-01"},
        {"query": "ok", "tags": "x" * (server._MAX_FILTER_CHARS + 1)},
        {"query": "ok", "tags": 123},
        {"query": "ok", "filename_contains": []},
        {"query": "ok", "created_before": None},
        {"query": 123},
    ]
    with mock.patch.object(server, "_invoke_pipe_b") as invoke:
        for arguments in invalid:
            try:
                server.rag_retrieve(**arguments)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid request: {arguments}")
        invoke.assert_not_called()


def t_accepts_documented_boundary_lengths():
    server = _server_module()
    with mock.patch.object(server, "_invoke_pipe_b", return_value="boundary-ok") as invoke:
        result = server.rag_retrieve(
            query="x" * server._MAX_QUERY_CHARS,
            tags="x" * server._MAX_FILTER_CHARS,
        )
    assert result == "boundary-ok"
    invoke.assert_called_once()


def t_pipe_b_error_becomes_mcp_execution_error():
    server = _server_module()
    with mock.patch.object(server, "_invoke_pipe_b", return_value="[error] index unavailable"):
        try:
            server.rag_retrieve("query")
        except RuntimeError as exc:
            assert str(exc) == "Pipe B rag_retrieve returned an error"
        else:
            raise AssertionError("Pipe B error was returned as a successful result")


def t_pipe_b_exception_is_mapped_without_leaking_details():
    server = _server_module()
    secret_detail = "secret-path-/tmp/private"
    with mock.patch.object(server, "_invoke_pipe_b", side_effect=ValueError(secret_detail)):
        try:
            server.rag_retrieve("query")
        except RuntimeError as exc:
            assert "ValueError" in str(exc)
            assert secret_detail not in str(exc)
            assert exc.__cause__ is None
        else:
            raise AssertionError("Pipe B exception was returned as a successful result")


def t_non_text_pipe_b_result_is_rejected():
    server = _server_module()
    with mock.patch.object(server, "_invoke_pipe_b", return_value={"result": "not text"}):
        try:
            server.rag_retrieve("query")
        except RuntimeError as exc:
            assert str(exc) == "Pipe B rag_retrieve returned a non-text result"
        else:
            raise AssertionError("non-text Pipe B result was returned as success")


def t_oversized_result_is_explicitly_capped():
    server = _server_module()
    with mock.patch.object(server, "_MAX_OUTPUT_CHARS", 256), mock.patch.object(
        server, "_invoke_pipe_b", return_value="x" * 1_000
    ):
        result = server.rag_retrieve("query")
    assert len(result) <= 256
    assert result.startswith("x")
    assert "[truncated]" in result


def t_tiny_cap_never_exceeds_configured_limit():
    server = _server_module()
    with mock.patch.object(server, "_MAX_OUTPUT_CHARS", 32), mock.patch.object(
        server, "_invoke_pipe_b", return_value="x" * 1_000
    ):
        result = server.rag_retrieve("query")
    assert len(result) <= 32
    assert "[truncated]" in result


def t_result_at_or_below_cap_is_unchanged():
    server = _server_module()
    expected = "stable result"
    with mock.patch.object(server, "_MAX_OUTPUT_CHARS", len(expected)), mock.patch.object(
        server, "_invoke_pipe_b", return_value=expected
    ):
        assert server.rag_retrieve("query") == expected


def t_shared_kb_tools_are_deterministic_and_read_only():
    server = _server_module()
    registered = ["/kb/alpha.md", "/kb/beta.md", "/kb/gamma.pdf"]

    with mock.patch.object(server.kb_operations, "registered_paths", return_value=registered):
        listed = server.rag_list(limit=2, offset=1)
    assert "registered_files=3" in listed
    assert "returned=2" in listed
    assert "beta.md" in listed and "gamma.pdf" in listed
    assert "alpha.md" not in listed

    with mock.patch.object(
        server.kb_operations,
        "search_registered_paths",
        return_value=["/kb/alpha.md", "/kb/gamma.pdf"],
    ):
        searched = server.rag_search_files("a", limit=1)
    assert "matches=2" in searched
    assert "returned=1" in searched
    assert "alpha.md" in searched and "gamma.pdf" not in searched

    with mock.patch.object(
        server.kb_operations,
        "read_registered_file",
        return_value=(Path("/kb/alpha.md"), "REGISTERED_BODY"),
    ):
        read = server.rag_read_file("alpha.md")
    assert read == "file=alpha.md\nREGISTERED_BODY"

    with mock.patch.object(
        server.kb_operations,
        "health_snapshot",
        return_value={
            "issues": [],
            "ghost_files": [],
            "registered_files": 3,
            "pipeline_outdated_files": 0,
            "index_status": "ready",
            "index_files": 3,
        },
    ), mock.patch.object(server.build_jobs, "active_job_snapshot", return_value=None):
        health = server.rag_health()
    assert "status=healthy" in health
    assert "issues=0" in health
    assert "ghost_files=0" in health
    assert "registered_files=3" in health
    assert "pipeline_outdated_files=0" in health
    assert "index_status=ready" in health
    assert "index_files=3" in health
    assert "build_job_status=idle" in health


def t_build_tools_delegate_without_loading_ingestor():
    server = _server_module()
    assert "ingestor" not in sys.modules
    with mock.patch.object(server.build_jobs, "start_build", return_value="status=started\njob_id=build-123456789abc") as start:
        out = server.build_kb()
    assert "status=started" in out
    start.assert_called_once_with()

    with mock.patch.object(server.build_jobs, "build_status", return_value="status=running") as status:
        assert server.build_status("build-123456789abc") == "status=running"
    status.assert_called_once_with("build-123456789abc")

    with mock.patch.object(server.build_jobs, "cancel_build", return_value="status=cancelling") as cancel:
        assert server.cancel_build("build-123456789abc") == "status=cancelling"
    cancel.assert_called_once_with("build-123456789abc")

    import types
    prepared = {
        "status": "prepared",
        "expected_fingerprint": "a" * 64,
        "total_files": 2,
        "sampled_files": 2,
        "filenames": ["alpha.md", "beta.pdf"],
        "top_tags": {"finance": 1},
        "source_types": {"md": 1, "pdf": 1},
        "topic_rules": {"guidance": "caller summarizes"},
    }
    committed = {
        "total": 2,
        "sampled": 2,
        "tags": {"finance": 1},
        "source_types": {"md": 1, "pdf": 1},
        "topics_line": "การลงทุน | ความเสี่ยง",
        "registry_fingerprint": "a" * 64,
    }
    fake_builder = types.SimpleNamespace(
        prepare_index_context=mock.Mock(return_value=prepared),
        commit_index=mock.Mock(return_value=committed),
        IndexCommitConflict=type("IndexCommitConflict", (RuntimeError,), {}),
        RAG_INDEX_PATH=Path("/tmp/rag_index.json"),
    )
    with mock.patch.object(server.build_jobs, "active_job_snapshot", return_value=None), \
         mock.patch.dict(sys.modules, {"rag_index_builder": fake_builder}):
        # Generic callers may not know the remote schema on the first call.
        # Premature topics without a fingerprint must safely become prepare,
        # never a write or an error.
        prep = server.rag_rebuild_index(topics=["premature guess"])
        assert "status=prepared" in prep
        assert "expected_fingerprint=" + "a" * 64 in prep
        assert "filenames_json=" in prep
        assert "caller_must_continue=true" in prep
        assert "caller_must_not_answer_before_commit=true" in prep
        assert "next_arguments_json=" in prep
        commit = server.rag_rebuild_index(
            mode="commit",
            topics=["การลงทุน", "ความเสี่ยง"],
            expected_fingerprint="a" * 64,
        )
        assert "status=done" in commit
        assert "topic_method=caller_llm" in commit
        assert "llm_used_by_backend=false" in commit
    fake_builder.prepare_index_context.assert_called_once_with()
    fake_builder.commit_index.assert_called_once_with(["การลงทุน", "ความเสี่ยง"], "a" * 64)
    assert "ingestor" not in sys.modules


def t_shared_kb_tools_validate_scope_and_bounds():
    server = _server_module()
    invalid_calls = [
        lambda: server.rag_list(limit=0),
        lambda: server.rag_list(limit=server._MAX_LIST_LIMIT + 1),
        lambda: server.rag_list(offset=-1),
        lambda: server.rag_search_files(""),
        lambda: server.rag_search_files("x", limit=0),
        lambda: server.rag_read_file(""),
    ]
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid shared-KB request")

    with mock.patch.object(
        server.kb_operations,
        "read_registered_file",
        side_effect=server.kb_operations.RegisteredFileNotFound("/etc/passwd"),
    ):
        try:
            server.rag_read_file("/etc/passwd")
        except ValueError as exc:
            assert "not registered" in str(exc)
        else:
            raise AssertionError("unregistered arbitrary path was readable")

    ambiguous = server.kb_operations.RegisteredFileAmbiguous(
        "report", ["/kb/report-a.md", "/kb/report-b.md"]
    )
    with mock.patch.object(server.kb_operations, "read_registered_file", side_effect=ambiguous):
        try:
            server.rag_read_file("report")
        except ValueError as exc:
            assert "ambiguous" in str(exc)
            assert "report-a.md" in str(exc)
        else:
            raise AssertionError("ambiguous filename was accepted")


def t_pipe_c_import_graph_never_loads_local_llm_modules():
    server = _server_module()
    assert server is not None
    for module_name in ("main", "llm_client", "rag_search", "ingestor", "rag_index_builder"):
        assert module_name not in sys.modules, module_name


def t_pipe_b_calls_are_serialized():
    server = _server_module()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    entered: list[int] = []

    def fake_invoke(arguments):
        call_no = len(entered) + 1
        entered.append(call_no)
        if call_no == 1:
            first_entered.set()
            assert not second_entered.is_set()
            assert release_first.wait(2), "first Pipe B call did not receive release"
        else:
            second_entered.set()
        return f"ok-{call_no}"

    results: list[str] = []

    def call():
        results.append(server.rag_retrieve("query"))

    with mock.patch.object(server, "_invoke_pipe_b", side_effect=fake_invoke):
        first = threading.Thread(target=call)
        second = threading.Thread(target=call)
        try:
            first.start()
            assert first_entered.wait(2), "first Pipe B call did not start"
            second.start()
            assert not second_entered.wait(0.1), "second Pipe B call entered before first left"
        finally:
            release_first.set()
            first.join(2)
            second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert sorted(results) == ["ok-1", "ok-2"]


def t_stdio_handshake_lists_pipe_c_without_starting_a_model():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_path = Path(__file__).resolve().parents[1] / "mcp_server.py"

    async def handshake():
        parameters = StdioServerParameters(command=sys.executable, args=[str(server_path)])
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                return initialized.serverInfo.name, [item.name for item in listed.tools]

    name, tools = asyncio.run(handshake())
    assert name == "MCP-RagMax Pipe C"
    assert tools == [
        "rag_retrieve",
        "rag_list",
        "rag_search_files",
        "rag_read_file",
        "build_kb",
        "build_status",
        "cancel_build",
        "rag_rebuild_index",
        "rag_health",
    ]


r.test("catalog exposes nine bounded tools", t_catalog_exposes_nine_bounded_tools)
r.test("Pipe C does not load Pipe A or its LLM", t_pipe_c_does_not_load_pipe_a_or_llm)
r.test("Pipe B resolves inside this RAG tree", t_pipe_b_module_resolves_inside_this_rag_tree)
r.test("valid request delegates to Pipe B", t_delegates_validated_request_to_pipe_b)
r.test("invalid request is rejected before Pipe B", t_rejects_invalid_requests_before_pipe_b)
r.test("documented boundary lengths are accepted", t_accepts_documented_boundary_lengths)
r.test("Pipe B error becomes MCP execution error", t_pipe_b_error_becomes_mcp_execution_error)
r.test("Pipe B exception details are not leaked", t_pipe_b_exception_is_mapped_without_leaking_details)
r.test("non-text Pipe B result is rejected", t_non_text_pipe_b_result_is_rejected)
r.test("oversized output is explicitly capped", t_oversized_result_is_explicitly_capped)
r.test("tiny output cap never overflows", t_tiny_cap_never_exceeds_configured_limit)
r.test("output at the cap is unchanged", t_result_at_or_below_cap_is_unchanged)
r.test("shared KB tools are deterministic and read-only", t_shared_kb_tools_are_deterministic_and_read_only)
r.test("build tools delegate without loading ingestor", t_build_tools_delegate_without_loading_ingestor)
r.test("shared KB tools validate scope and bounds", t_shared_kb_tools_validate_scope_and_bounds)
r.test("Pipe C import graph never loads local LLM modules", t_pipe_c_import_graph_never_loads_local_llm_modules)
r.test("Pipe B calls are serialized", t_pipe_b_calls_are_serialized)
r.test("stdio handshake lists Pipe C without starting a model", t_stdio_handshake_lists_pipe_c_without_starting_a_model)

if __name__ == "__main__":
    r.exit()
