"""_test_tools.py — rag_search / rag_retrieve tool contract + _to_abs portability"""
import os
import tempfile
from pathlib import Path

from _runner import Runner

r = Runner("tools")


def t26_rag_search_empty():
    from rag_search import rag_search
    result = rag_search.invoke({"query": "machine learning"})
    assert isinstance(result, str)
    assert "[low_quality]" in result or "[error]" in result or "SOURCES:" in result


def t27_rag_retrieve_empty():
    from rag_retrieve import rag_retrieve
    result = rag_retrieve.invoke({"query": "test query"})
    assert isinstance(result, str)
    assert "[error]" in result or "[1]" in result


def t29_tool_returns_str():
    from rag_search import rag_search
    from rag_retrieve import rag_retrieve
    r1 = rag_search.invoke({"query": "test"})
    r2 = rag_retrieve.invoke({"query": "test"})
    assert isinstance(r1, str), f"rag_search returned {type(r1)}"
    assert isinstance(r2, str), f"rag_retrieve returned {type(r2)}"
    assert r1 is not None and r2 is not None


def t34_to_abs_independent_of_cwd():
    """Stored relative keys resolve against the configured knowledge root, not cwd."""
    import config
    import rag_retrieve
    import rag_search
    rel = "_does_not_need_to_exist.md"
    expected = str((config.KNOWLEDGE_DIR / rel).resolve())

    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            assert rag_retrieve._to_abs(rel) == expected
            assert rag_search._to_abs(rel) == expected
    finally:
        os.chdir(cwd)


def t35_to_abs_rejects_absolute_escape():
    """Public port never turns an arbitrary absolute path into a readable source."""
    import rag_retrieve
    import rag_search
    for fn in (rag_retrieve._to_abs, rag_search._to_abs):
        try:
            fn("/tmp/some_absolute_source.md")
        except ValueError:
            pass
        else:
            raise AssertionError("absolute path outside knowledge root was accepted")

def t40_filename_containing_chroma_is_indexable():
    import ingestor

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "chromatic.md"
        path.write_text("supported document")
        assert ingestor._is_indexable_file(path)

        unsupported = root / "image.png"
        unsupported.write_text("not a supported document")
        assert not ingestor._is_indexable_file(unsupported)

        runtime_file = root / "file_hashes.json"
        runtime_file.write_text("{}")
        assert not ingestor._is_indexable_file(runtime_file)

        saved_db_dir = ingestor.store.DB_DIR
        try:
            runtime_root = root / "runtime-data"
            runtime_root.mkdir()
            ingestor.store.DB_DIR = runtime_root / "chroma"
            runtime_doc = runtime_root / "chromatic-runtime.md"
            runtime_doc.write_text("runtime state")
            assert not ingestor._is_indexable_file(runtime_doc)
        finally:
            ingestor.store.DB_DIR = saved_db_dir


def t41_file_status_error_does_not_abort_build():
    import config
    import file_registry
    import ingestor
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    saved = {
        "DATA_DIR": ingestor.DATA_DIR,
        "check": file_registry.check,
        "all_registered": file_registry.all_registered,
        "health_check": ingestor.store.health_check,
        "ghost_files": file_registry.ghost_files,
    }
    try:
        with tempfile.TemporaryDirectory(dir=config.KNOWLEDGE_DIR) as tmp:
            ingestor.DATA_DIR = Path(tmp)
            (ingestor.DATA_DIR / "one.md").write_text("supported document")
            (ingestor.DATA_DIR / "two.md").write_text("second document")

            def fail_check(path):
                raise RuntimeError("probe")

            file_registry.check = fail_check
            file_registry.all_registered = lambda: []
            ingestor.store.health_check = lambda: []
            file_registry.ghost_files = lambda: []
            callbacks = []
            result = ingestor.sync_knowledge_base(
                on_file=lambda idx, total, name, status, n_chunks, err: callbacks.append(
                    (idx, total, name, status, n_chunks, err)
                )
            )

            assert result["total_found"] == 2
            assert [row["status"] for row in result["rows"]] == ["error", "error"]
            assert [event[2] for event in callbacks] == ["one.md", "two.md"]
            assert all(event[3:] == ("error", 0, "probe") for event in callbacks)
    finally:
        ingestor.DATA_DIR = saved["DATA_DIR"]
        file_registry.check = saved["check"]
        file_registry.all_registered = saved["all_registered"]
        ingestor.store.health_check = saved["health_check"]
        file_registry.ghost_files = saved["ghost_files"]


def t55_unregistered_partial_source_is_rebuilt():
    import config
    import file_registry
    import ingestor
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    saved = {
        "DATA_DIR": ingestor.DATA_DIR,
        "check": file_registry.check,
        "all_registered": file_registry.all_registered,
        "has_source": ingestor.store.has_source,
        "delete_file": ingestor.delete_file,
        "ingest_file": ingestor.ingest_file,
        "register": file_registry.register,
        "deregister": file_registry.deregister,
        "health_check": ingestor.store.health_check,
        "ghost_files": file_registry.ghost_files,
    }
    deleted = []
    ingested = []
    registered = []
    deregistered = []
    try:
        with tempfile.TemporaryDirectory(dir=config.KNOWLEDGE_DIR) as tmp:
            ingestor.DATA_DIR = Path(tmp)
            path = ingestor.DATA_DIR / "partial.md"
            path.write_text("supported document")
            file_registry.check = lambda value: "new"
            file_registry.all_registered = lambda: []
            ingestor.store.has_source = lambda source: True
            ingestor.delete_file = lambda value: deleted.append(Path(value).name) or 1
            ingestor.ingest_file = lambda value: ingested.append(Path(value).name) or 1
            file_registry.register = lambda value: registered.append(value)
            file_registry.deregister = lambda value: deregistered.append(value)
            ingestor.store.health_check = lambda: []
            file_registry.ghost_files = lambda: []
            starts = []
            callbacks = []
            result = ingestor.sync_knowledge_base(
                on_file_start=lambda idx, total, name: starts.append((idx, total, name)),
                on_file=lambda idx, total, name, status, n_chunks, err: callbacks.append(
                    (idx, total, name, status, n_chunks, err)
                ),
            )

            assert deleted == ["partial.md"], deleted
            assert ingested == ["partial.md"]
            assert registered == [str(path)]
            assert deregistered == [str(path)]
            assert starts == [(1, 1, "partial.md")]
            assert callbacks == [(1, 1, "partial.md", "changed", 1, None)]
            assert result["rows"] == [{"name": "partial.md", "status": "changed", "n_chunks": 1}]
    finally:
        ingestor.DATA_DIR = saved["DATA_DIR"]
        file_registry.check = saved["check"]
        file_registry.all_registered = saved["all_registered"]
        ingestor.store.has_source = saved["has_source"]
        ingestor.delete_file = saved["delete_file"]
        ingestor.ingest_file = saved["ingest_file"]
        file_registry.register = saved["register"]
        file_registry.deregister = saved["deregister"]
        ingestor.store.health_check = saved["health_check"]
        file_registry.ghost_files = saved["ghost_files"]


def t56_skip_path_and_callbacks_are_consistent():
    """A healthy skip emits a result but never starts an ingest spinner."""
    import config
    import file_registry
    import ingestor
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    saved = {
        "DATA_DIR": ingestor.DATA_DIR,
        "check": file_registry.check,
        "all_registered": file_registry.all_registered,
        "has_source": ingestor.store.has_source,
        "ingest_file": ingestor.ingest_file,
        "register": file_registry.register,
        "health_check": ingestor.store.health_check,
        "ghost_files": file_registry.ghost_files,
    }
    try:
        with tempfile.TemporaryDirectory(dir=config.KNOWLEDGE_DIR) as tmp:
            ingestor.DATA_DIR = Path(tmp)
            (ingestor.DATA_DIR / "skip.md").write_text("already indexed")
            (ingestor.DATA_DIR / "new.md").write_text("needs indexing")

            file_registry.check = lambda value: (
                "skip" if Path(value).name == "skip.md" else "new"
            )
            file_registry.all_registered = lambda: []
            ingestor.store.has_source = lambda source: source.endswith("skip.md")
            ingestor.ingest_file = lambda value: 3
            file_registry.register = lambda value: None
            ingestor.store.health_check = lambda: []
            file_registry.ghost_files = lambda: []
            starts = []
            callbacks = []
            result = ingestor.sync_knowledge_base(
                on_file_start=lambda idx, total, name: starts.append((idx, total, name)),
                on_file=lambda idx, total, name, status, n_chunks, err: callbacks.append(
                    (idx, total, name, status, n_chunks, err)
                ),
            )

            rows = {row["name"]: row for row in result["rows"]}
            assert rows == {
                "new.md": {"name": "new.md", "status": "new", "n_chunks": 3},
                "skip.md": {"name": "skip.md", "status": "skip", "n_chunks": 0},
            }
            assert starts == [(1, 2, "new.md")]
            assert callbacks == [
                (1, 2, "new.md", "new", 3, None),
                (2, 2, "skip.md", "skip", 0, None),
            ]
    finally:
        ingestor.DATA_DIR = saved["DATA_DIR"]
        file_registry.check = saved["check"]
        file_registry.all_registered = saved["all_registered"]
        ingestor.store.has_source = saved["has_source"]
        ingestor.ingest_file = saved["ingest_file"]
        file_registry.register = saved["register"]
        ingestor.store.health_check = saved["health_check"]
        file_registry.ghost_files = saved["ghost_files"]


def t57_stale_registered_sources_are_purged():
    import config
    import file_registry
    import ingestor
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    saved = {
        "DATA_DIR": ingestor.DATA_DIR,
        "all_registered": file_registry.all_registered,
        "deregister": file_registry.deregister,
        "check": file_registry.check,
        "has_source": ingestor.store.has_source,
        "ingest_file": ingestor.ingest_file,
        "register": file_registry.register,
        "delete_file": ingestor.delete_file,
        "health_check": ingestor.store.health_check,
        "ghost_files": file_registry.ghost_files,
    }
    purged = []
    deregistered = []
    try:
        with tempfile.TemporaryDirectory(dir=config.KNOWLEDGE_DIR) as tmp:
            root = Path(tmp)
            ingestor.DATA_DIR = root
            (root / "current.md").write_text("current")
            stale = root / "deleted.md"

            file_registry.all_registered = lambda: [str(stale)]
            ingestor.delete_file = lambda value: purged.append(Path(value).name) or 1
            file_registry.deregister = lambda value: deregistered.append(value)
            file_registry.check = lambda value: "new"
            ingestor.store.has_source = lambda source: False
            ingestor.ingest_file = lambda value: 1
            file_registry.register = lambda value: None
            ingestor.store.health_check = lambda: []
            file_registry.ghost_files = lambda: []

            result = ingestor.sync_knowledge_base()

            assert purged == ["deleted.md"]
            assert deregistered == [str(stale)]
            assert result["total_found"] == 1
            assert result["rows"] == [
                {"name": "current.md", "status": "new", "n_chunks": 1}
            ]
    finally:
        ingestor.DATA_DIR = saved["DATA_DIR"]
        file_registry.all_registered = saved["all_registered"]
        file_registry.deregister = saved["deregister"]
        file_registry.check = saved["check"]
        ingestor.store.has_source = saved["has_source"]
        ingestor.ingest_file = saved["ingest_file"]
        file_registry.register = saved["register"]
        ingestor.delete_file = saved["delete_file"]
        ingestor.store.health_check = saved["health_check"]
        file_registry.ghost_files = saved["ghost_files"]


def t58_main_is_deterministic_web_ui_launcher():
    import main
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "web_ui:app" in source
    assert "127.0.0.1" in source
    assert "llm_client" not in source
    assert "create_react_agent" not in source
    assert "build_llm" not in source


r.test("T26 rag_search returns str on empty DB", t26_rag_search_empty)
r.test("T27 rag_retrieve returns str on empty DB", t27_rag_retrieve_empty)
r.test("T29 tools return str not None/dict/list", t29_tool_returns_str)
r.test("T34 _to_abs independent of cwd", t34_to_abs_independent_of_cwd)
r.test("T35 _to_abs rejects absolute escape", t35_to_abs_rejects_absolute_escape)
r.test("T40 supported filename containing chroma remains indexable", t40_filename_containing_chroma_is_indexable)
r.test("T41 file status error does not abort build", t41_file_status_error_does_not_abort_build)
r.test("T55 unregistered partial source is rebuilt", t55_unregistered_partial_source_is_rebuilt)
r.test("T56 skip path and callbacks are consistent", t56_skip_path_and_callbacks_are_consistent)
r.test("T57 stale registered sources are purged", t57_stale_registered_sources_are_purged)
r.test("T58 main is deterministic web UI launcher", t58_main_is_deterministic_web_ui_launcher)

if __name__ == "__main__":
    r.exit()
