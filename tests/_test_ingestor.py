"""_test_ingestor.py — ingestor file loaders + portability (_rel out-of-tree fallback)"""
import csv
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from _runner import Runner

r = Runner("ingestor")


def t12_txt():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        f.write("hello world machine learning")
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "machine learning" in text


def t13_md():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w") as f:
        f.write("# Title\n\ncontent here")
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "content" in text


def t14_csv():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as f:
        csv.writer(f).writerows([["name", "age"], ["Alice", "30"], ["Bob", "25"]])
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "[columns: name, age]" in text
    assert "Alice" in text


def t15_json():
    from ingestor import load_file
    data = {"users": [{"name": "Alice", "age": 30}], "version": 1}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f)
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "[json schema:" in text
    assert "Alice" in text
    assert "version" in text


def t16_csv_header_every_chunk():
    from ingestor import _load_csv
    from chunker import chunk_document
    rows = [["col1", "col2", "col3"]] + [[f"val{i}a", f"val{i}b", f"val{i}c"] for i in range(20)]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as f:
        csv.writer(f).writerows(rows)
        path = Path(f.name)
    text = _load_csv(path)
    os.unlink(path)
    chunks = chunk_document(text)
    for c in chunks:
        assert "[columns:" in c["child_text"], f"Missing header in chunk: {c['child_text'][:100]}"


def t17_json_header_every_chunk():
    from ingestor import _load_json
    from chunker import chunk_document
    data = {"items": [{"id": i, "val": f"value_{i}" * 5} for i in range(30)]}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f)
        path = Path(f.name)
    text = _load_json(path)
    os.unlink(path)
    chunks = chunk_document(text)
    for c in chunks:
        assert "[json schema:" in c["child_text"], f"Missing header in chunk: {c['child_text'][:100]}"


def t32_rel_rejects_outside_knowledge():
    """Public ingestion never stores an arbitrary out-of-KB source path."""
    from ingestor import _rel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        outside = Path(f.name)
    try:
        try:
            _rel(outside)
        except ValueError:
            pass
        else:
            raise AssertionError("outside source path was accepted")
    finally:
        os.unlink(outside)


def t33_rel_inside_knowledge_is_relative():
    """A source inside configured knowledge is stored as a portable key."""
    import config
    from ingestor import _rel
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    inside = config.KNOWLEDGE_DIR / "_portability_ingestor_test.tmp"
    inside.write_text("x", encoding="utf-8")
    try:
        assert _rel(inside) == inside.name
    finally:
        inside.unlink(missing_ok=True)

def t53_dedup_is_deterministic_without_llm_novelty_gate():
    import ingestor

    original = ingestor.store.dense_search
    try:
        ingestor._chunk_hashes.clear()
        ingestor.store.dense_search = lambda vec, top_k=1: [{"score": 0.80, "child_text": "near"}]
        assert ingestor._should_store("candidate-a", [1.0, 0.0]) is True

        ingestor._chunk_hashes.clear()
        ingestor.store.dense_search = lambda vec, top_k=1: [{"score": 0.96, "child_text": "duplicate"}]
        assert ingestor._should_store("candidate-b", [1.0, 0.0]) is False
        assert not hasattr(ingestor, "_llm_novelty")
        assert not hasattr(ingestor, "_llm_chat")
    finally:
        ingestor.store.dense_search = original
        ingestor._chunk_hashes.clear()


def t54_partial_dual_store_write_is_rolled_back():
    import ingestor

    saved = {
        "encode": ingestor.embedder.encode,
        "should_store": ingestor._should_store,
        "upsert": ingestor.store.upsert_chunk,
        "bm25_add": ingestor.store.bm25_add,
        "bm25_flush": ingestor.store.bm25_flush,
        "delete": ingestor.store.delete_by_source,
        "bm25_delete": ingestor.store.bm25_delete_by_source,
    }
    calls = []
    try:
        import config
        config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.KNOWLEDGE_DIR / "_rollback_probe.txt"
        path.write_text("rollback probe content", encoding="utf-8")
        ingestor.embedder.encode = lambda texts: [[1.0, 0.0] for _ in texts]
        ingestor._should_store = lambda text, vec: True
        ingestor.store.upsert_chunk = lambda **kwargs: "chunk-id"
        ingestor.store.bm25_add = lambda *args, **kwargs: None
        ingestor.store.bm25_flush = lambda: (_ for _ in ()).throw(OSError("disk full"))
        ingestor.store.delete_by_source = lambda source: calls.append(("dense", source)) or 1
        ingestor.store.bm25_delete_by_source = lambda source: calls.append(("bm25", source))
        try:
            ingestor.ingest_file(path)
        except OSError:
            pass
        else:
            raise AssertionError("expected controlled flush failure")
        assert [kind for kind, _ in calls] == ["dense", "bm25"], calls
    finally:
        if "path" in locals() and path.exists():
            path.unlink()
        ingestor.embedder.encode = saved["encode"]
        ingestor._should_store = saved["should_store"]
        ingestor.store.upsert_chunk = saved["upsert"]
        ingestor.store.bm25_add = saved["bm25_add"]
        ingestor.store.bm25_flush = saved["bm25_flush"]
        ingestor.store.delete_by_source = saved["delete"]
        ingestor.store.bm25_delete_by_source = saved["bm25_delete"]


def t55_new_file_cancel_rolls_back_partial_dual_store_write():
    import ingestor

    saved = {
        "chunk_document": ingestor.chunk_document,
        "encode": ingestor.embedder.encode,
        "should_store": ingestor._should_store,
        "upsert": ingestor.store.upsert_chunk,
        "bm25_add": ingestor.store.bm25_add,
        "bm25_flush": ingestor.store.bm25_flush,
        "delete": ingestor.store.delete_by_source,
        "bm25_delete": ingestor.store.bm25_delete_by_source,
    }
    calls = []
    cancel = {"requested": False}
    try:
        import config
        config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.KNOWLEDGE_DIR / "_cancel_rollback_probe.txt"
        path.write_text("cancel rollback probe", encoding="utf-8")
        ingestor.chunk_document = lambda _text: [
            {"child_text": "one", "parent_text": "parent", "parent_index": 0, "child_index": 0},
            {"child_text": "two", "parent_text": "parent", "parent_index": 0, "child_index": 1},
        ]
        ingestor.embedder.encode = lambda texts: [[1.0, 0.0] for _ in texts]
        ingestor._should_store = lambda text, vec: True
        ingestor.store.upsert_chunk = lambda **kwargs: calls.append(("upsert", kwargs["child_idx"])) or f"chunk-{kwargs['child_idx']}"
        ingestor.store.bm25_add = lambda *args, **kwargs: None
        ingestor.store.bm25_flush = lambda: None
        ingestor.store.delete_by_source = lambda source: calls.append(("dense_rollback", source)) or 1
        ingestor.store.bm25_delete_by_source = lambda source: calls.append(("bm25_rollback", source))

        def on_progress(done, total):
            assert total == 2
            if done == 1:
                cancel["requested"] = True

        try:
            ingestor.ingest_file(
                path,
                on_progress=on_progress,
                should_cancel=lambda: cancel["requested"],
            )
        except ingestor.BuildCancelled:
            pass
        else:
            raise AssertionError("expected cooperative cancellation")

        assert ("upsert", 0) in calls
        assert not any(item == ("upsert", 1) for item in calls)
        assert [kind for kind, *_ in calls if "rollback" in kind] == [
            "dense_rollback",
            "bm25_rollback",
        ]
    finally:
        if "path" in locals() and path.exists():
            path.unlink()
        ingestor.chunk_document = saved["chunk_document"]
        ingestor.embedder.encode = saved["encode"]
        ingestor._should_store = saved["should_store"]
        ingestor.store.upsert_chunk = saved["upsert"]
        ingestor.store.bm25_add = saved["bm25_add"]
        ingestor.store.bm25_flush = saved["bm25_flush"]
        ingestor.store.delete_by_source = saved["delete"]
        ingestor.store.bm25_delete_by_source = saved["bm25_delete"]


def t56_changed_file_finishes_before_cancellation_is_honored():
    import file_registry
    import ingestor

    saved_data_dir = ingestor.DATA_DIR
    events = []
    cancel = {"requested": False}
    import config
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.KNOWLEDGE_DIR) as td:
        root = Path(td)
        source = root / "changed.txt"
        source.write_text("changed source", encoding="utf-8")
        try:
            ingestor.DATA_DIR = root
            with mock.patch.object(ingestor, "_is_indexable_file", side_effect=lambda p: p == source), \
                 mock.patch.object(file_registry, "all_registered", return_value=[str(source.resolve())]), \
                 mock.patch.object(file_registry, "check", return_value="changed"), \
                 mock.patch.object(file_registry, "deregister", side_effect=lambda p: events.append("deregister")), \
                 mock.patch.object(file_registry, "register", side_effect=lambda p: events.append("register")), \
                 mock.patch.object(ingestor, "delete_file", side_effect=lambda p: events.append("delete") or 1), \
                 mock.patch.object(ingestor.store, "has_source", return_value=True), \
                 mock.patch.object(ingestor.store, "health_check", return_value=[]), \
                 mock.patch.object(file_registry, "ghost_files", return_value=[]):

                def fake_ingest(path, on_progress=None, should_cancel=None):
                    assert path == source
                    assert should_cancel is None, "changed-file rebuild must not cancel after old rows are removed"
                    events.append("ingest")
                    cancel["requested"] = True
                    return 1

                with mock.patch.object(ingestor, "ingest_file", side_effect=fake_ingest):
                    try:
                        ingestor.sync_knowledge_base(
                            should_cancel=lambda: cancel["requested"]
                        )
                    except ingestor.BuildCancelled:
                        pass
                    else:
                        raise AssertionError("expected cancellation after current changed file")

            assert events[:4] == ["delete", "deregister", "ingest", "register"], events
        finally:
            ingestor.DATA_DIR = saved_data_dir


r.test("T12 txt loader", t12_txt)
r.test("T13 md loader", t13_md)
r.test("T14 csv loader + header", t14_csv)
r.test("T15 json loader + schema", t15_json)
r.test("T16 csv header in every child chunk", t16_csv_header_every_chunk)
r.test("T17 json schema in every child chunk", t17_json_header_every_chunk)
r.test("T32 _rel rejects outside knowledge", t32_rel_rejects_outside_knowledge)
r.test("T33 _rel inside knowledge is relative", t33_rel_inside_knowledge_is_relative)
r.test("T53 deterministic no-LLM dedup", t53_dedup_is_deterministic_without_llm_novelty_gate)
r.test("T54 partial Chroma/BM25 write rolls back", t54_partial_dual_store_write_is_rolled_back)
r.test("T55 new-file cancellation rolls back partial stores", t55_new_file_cancel_rolls_back_partial_dual_store_write)
r.test("T56 changed-file cancellation waits for safe boundary", t56_changed_file_finishes_before_cancellation_is_honored)

if __name__ == "__main__":
    r.exit()
