"""_test_store.py — store.upsert_chunk / dense_search / bm25 / get_chunk_hash"""
from _runner import Runner

r = Runner("store")


def _snapshot(store):
    names = [
        "DB_DIR", "BM25_PATH", "_chroma_client", "_collection", "_bm25",
        "_bm25_corpus", "_bm25_ids", "_bm25_sources", "_bm25_mtime",
    ]
    if hasattr(store, "_bm25_pending"):
        names.append("_bm25_pending")
    return {name: getattr(store, name) for name in names}


def _reset(store, root):
    from pathlib import Path
    store.DB_DIR = Path(root) / "chroma"
    store.BM25_PATH = Path(root) / "bm25.pkl"
    store._chroma_client = None
    store._collection = None
    store._bm25 = None
    store._bm25_corpus = []
    store._bm25_ids = []
    store._bm25_sources = []
    store._bm25_mtime = 0.0
    if hasattr(store, "_bm25_pending"):
        store._bm25_pending = {}


def _restore(store, saved):
    for name, value in saved.items():
        setattr(store, name, value)


def t08_upsert_dense_delete():
    from embedder import encode_one
    from store import upsert_chunk, dense_search, delete_by_source
    vec = encode_one("unit test document about AI")
    upsert_chunk("unit test AI", "parent unit test AI document", vec, "_smoke_.md", 0, 0)
    res = dense_search(encode_one("artificial intelligence"), top_k=5)
    assert any(rr["source"] == "_smoke_.md" for rr in res)
    n = delete_by_source("_smoke_.md")
    assert n >= 1


def t09_bm25_add_search_delete():
    from embedder import encode_one
    from store import upsert_chunk, bm25_add, bm25_delete_by_source, bm25_search, delete_by_source
    vec = encode_one("python programming language")
    cid = upsert_chunk("python programming", "parent python", vec, "_smoke2_.md", 0, 0)
    bm25_add("python programming", cid, "_smoke2_.md")
    for i in range(3):
        v2 = encode_one(f"java c++ golang language {i}")
        cid2 = upsert_chunk(f"java c++ golang {i}", f"parent java {i}", v2, "_smoke2_.md", i + 1, 0)
        bm25_add(f"java c++ golang {i}", cid2, "_smoke2_.md")
    res = bm25_search("python programming", top_k=5)
    assert isinstance(res, list)  # may be empty on tiny corpus, just no crash
    delete_by_source("_smoke2_.md")
    bm25_delete_by_source("_smoke2_.md")


def t10_chunk_hash():
    from store import get_chunk_hash
    h1 = get_chunk_hash("hello")
    h2 = get_chunk_hash("hello")
    h3 = get_chunk_hash("world")
    assert h1 == h2
    assert h1 != h3


def t37_bm25_flush_preserves_cold_disk_index():
    """A zero-store ingest must not erase BM25 loaded only from disk."""
    import pickle
    import tempfile
    from pathlib import Path
    import store

    saved = {
        "BM25_PATH": store.BM25_PATH,
        "_bm25": store._bm25,
        "_bm25_corpus": store._bm25_corpus,
        "_bm25_ids": store._bm25_ids,
        "_bm25_sources": store._bm25_sources,
        "_bm25_mtime": store._bm25_mtime,
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25.pkl"
            expected = {
                "corpus": [["existing", "knowledge"]],
                "ids": ["existing-id"],
                "sources": ["existing.md"],
            }
            path.write_bytes(pickle.dumps(expected))
            store.BM25_PATH = path
            store._bm25 = None
            store._bm25_corpus = []
            store._bm25_ids = []
            store._bm25_sources = []
            store._bm25_mtime = 0.0

            store.bm25_flush()

            actual = pickle.loads(path.read_bytes())
            assert actual["ids"] == expected["ids"]
            assert actual["sources"] == expected["sources"]
    finally:
        for name, value in saved.items():
            setattr(store, name, value)


def t47_bm25_exact_match_works_in_one_document_corpus():
    import tempfile
    import store

    saved = _snapshot(store)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _reset(store, tmp)
            cid = store.upsert_chunk("rareterm", "rareterm parent", [1.0, 0.0, 0.0], "one.md", 0, 0)
            store.bm25_add("rareterm", cid, "one.md")
            results = store.bm25_search("rareterm", top_k=3)
            assert [r["id"] for r in results] == [cid]
            assert store.bm25_search("absentterm", top_k=3) == []
    finally:
        store._chroma_client = None
        store._collection = None
        _restore(store, saved)


def t48_health_reports_bm25_when_chroma_is_empty():
    import tempfile
    import store

    saved = _snapshot(store)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _reset(store, tmp)
            store._bm25_corpus = [["orphan"]]
            store._bm25_ids = ["orphan-id"]
            store._bm25_sources = ["orphan.md"]
            store._rebuild_bm25()
            store._save_bm25()
            issues = store.health_check()
            assert any("orphan" in issue.lower() for issue in issues), issues
    finally:
        store._chroma_client = None
        store._collection = None
        _restore(store, saved)


def t49_flush_merges_external_writer_update():
    import pickle
    import tempfile
    from pathlib import Path
    import store

    saved = _snapshot(store)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _reset(store, tmp)
            path = Path(tmp) / "bm25.pkl"
            base = {"corpus": [["base"]], "ids": ["base-id"], "sources": ["base.md"]}
            path.write_bytes(pickle.dumps(base))
            store._load_bm25()
            store.bm25_add("alpha", "id-a", "a.md", flush=False)

            external = {
                "corpus": [["base"], ["beta"]],
                "ids": ["base-id", "id-b"],
                "sources": ["base.md", "b.md"],
            }
            path.write_bytes(pickle.dumps(external))
            store.bm25_flush()
            final = pickle.loads(path.read_bytes())
            assert set(final["ids"]) == {"base-id", "id-a", "id-b"}, final["ids"]
    finally:
        _restore(store, saved)


r.test("T08 upsert + dense search + delete", t08_upsert_dense_delete)
r.test("T09 bm25 add + search + delete", t09_bm25_add_search_delete)
r.test("T10 chunk hash deterministic", t10_chunk_hash)
r.test("T37 cold BM25 flush preserves persisted index", t37_bm25_flush_preserves_cold_disk_index)
r.test("T47 BM25 exact match works for one-document corpus", t47_bm25_exact_match_works_in_one_document_corpus)
r.test("T48 empty Chroma still reports BM25 orphans", t48_health_reports_bm25_when_chroma_is_empty)
r.test("T49 BM25 flush merges external writer update", t49_flush_merges_external_writer_update)

if __name__ == "__main__":
    r.exit()
